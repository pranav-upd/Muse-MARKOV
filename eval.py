#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Eval driver for Muse Glimmer: GPQA Diamond + BFCL v4 (function calling), base vs finetuned.

* ``gpqa``  EleutherAI GPQA Diamond (zero-shot) on Modal GPU via the deployed ``lm_harness``
            function from starter.py (unchanged behaviour).
* ``bfcl``  Berkeley Function Calling Leaderboard v4, single-turn categories. This script
            generates the model outputs itself by calling your two Modal serve endpoints,
            saves them in BFCL's result format, then scores them with the official
            ``bfcl evaluate`` (deterministic AST matching: no judge model needed).

Why generation is done here and not by ``bfcl generate``
--------------------------------------------------------
BFCL only knows models registered in its own package config, and your endpoints take
``{"prompt", "max_new_tokens", "temperature"}`` rather than an OpenAI-style chat API. So the
results are filed under a *stand-in* BFCL model name (``--bfcl-model``) purely so the official
checker can find and decode them. The model being scored is still your endpoint. The prompt
is this script's own BFCL-style prompt, so numbers are comparable between your two arms but
NOT directly comparable to the public leaderboard.

Usage::

    python -m modal deploy starter.py            # once (GPQA path needs lm_harness)
    pip install bfcl-eval                        # once (BFCL path)

    python eval.py                               # GPQA, both weights (as before)
    python eval.py --bench bfcl --bfcl-limit 20  # smoke test: first 20 cases per category
    python eval.py --bench bfcl --bfcl-sample 60 --bfcl-seed 7   # seeded random 60 per category
    python eval.py --bench bfcl                  # default categories, both weights
    python eval.py --bench bfcl --bfcl-source hub
    python eval.py --bench bfcl --bfcl-score-only   # re-score saved generations
    python eval.py --bench both
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("glimmer-eval")

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
MODAL_APP = "muse-glimmer-grpo"
MODAL_FN = "lm_harness"

SERVE_URLS = {
    "hub": os.environ.get(
        "GLIMMER_FT_URL",
        "https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run",
    ),
    "base": os.environ.get(
        "GLIMMER_BASE_URL",
        "https://pranavupadhyaya52--muse-glimmer-grpo-serve-base.us-east.modal.run",
    ),
}
ARM_LABEL = {"hub": "finetuned", "base": "base"}

DEFAULT_BFCL_CATEGORIES = "simple_python,multiple,parallel,irrelevance"
# Only single-turn categories are supported (they share one data shape and one prompt format).
UNSUPPORTED_CATEGORY_PREFIXES = ("multi_turn", "memory", "web_search", "format_sensitivity")
# A *prompt-mode* (non -FC) model registered in BFCL; see SUPPORTED_MODELS.md in the BFCL repo.
DEFAULT_BFCL_STAND_IN = "meta-llama/Llama-3.1-8B-Instruct"


# --------------------------------------------------------------------------------------
# GPQA (unchanged)
# --------------------------------------------------------------------------------------
def run_lm_eval(tasks: list[str], source: str, limit: int | None, temperature: float = 0.3) -> int:
    """Score base and/or finetuned weights with the LM Evaluation Harness.

    Executes remotely on Modal GPU via the deployed ``lm_harness`` function.
    Prints a per-task comparison table. Returns 0 on success.
    """
    import modal  # local Modal client - lookup only, compute stays remote

    log.info("Looking up deployed lm_harness (deploy starter.py first if missing) ...")
    try:
        fn = modal.Function.from_name(MODAL_APP, MODAL_FN)
    except Exception as exc:
        log.error("Lookup failed: %s. Deploy with: python -m modal deploy starter.py", exc)
        return 2
    log.info("Running lm-eval tasks=%s source=%s limit=%s temperature=%s ...",
             tasks, source, limit, temperature)
    try:
        result = fn.remote(tasks, source, limit, temperature)
    except Exception as exc:
        log.error("Remote lm-eval failed: %s", exc)
        return 1
    sources = result.get("sources", {})
    names = sorted({t for s in sources.values() if isinstance(s, dict) for t in s if t != "error"})
    if result.get("error") or not names:
        blob = json.dumps(result)
        log.error("lm-eval error head: %.800s", blob[:800])
        log.error("lm-eval error tail: %.1500s", blob[-1500:])
        return 1
    header = f"{'task':36s}" + "".join(f"{s:>22s}" for s in sorted(sources))
    print(header)
    for task in names:
        row = f"{task:36s}"
        for src in sorted(sources):
            vals = sources[src].get(task, {})
            if not isinstance(vals, dict) or "error" in vals:
                row += f"{'ERR':>22s}"
                continue
            pick = vals.get("acc,none") or vals.get("acc_norm,none") or vals.get("acc")
            row += f"{(round(float(pick), 4) if pick is not None else 'n/a'):>22}"
        print(row)
    for src in sorted(sources):
        src_vals = sources[src]
        if isinstance(src_vals, dict):
            if "error" in src_vals:
                log.error("%s error: %.500s", src, src_vals["error"])
            for task, vals in src_vals.items():
                if isinstance(vals, dict) and "error" in vals:
                    log.error("%s/%s error: %.400s", src, task, vals["error"])
    return 0


# --------------------------------------------------------------------------------------
# Modal serve endpoint client
# --------------------------------------------------------------------------------------
def call_serve(url: str, prompt: str, max_new_tokens: int, temperature: float,
               timeout: int = 600, retries: int = 2) -> dict:
    """POST {"prompt","max_new_tokens","temperature"} -> {"reply","reasoning"}."""
    body = json.dumps({"prompt": prompt, "max_new_tokens": max_new_tokens,
                       "temperature": temperature}).encode()
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url.rstrip("/") + "/", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            log.warning("serve call failed (attempt %d/%d): %s", attempt + 1, retries + 1, exc)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"serve endpoint failed after {retries + 1} attempts: {last}")


def warmup(arm: str) -> None:
    """Wake the Modal container (cold start can take minutes) and fail fast if it is down."""
    log.info("Warming up %s endpoint (cold start may take a few minutes) ...", ARM_LABEL[arm])
    out = call_serve(SERVE_URLS[arm], "Reply with the single word OK.", 16, 0.0, timeout=900)
    log.info("%s endpoint ready; sample reply: %.60r", ARM_LABEL[arm], out.get("reply"))


# --------------------------------------------------------------------------------------
# BFCL: data loading and prompting
# --------------------------------------------------------------------------------------
BFCL_SYSTEM = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out. You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke.
{functions}
"""


def bfcl_data_dir(override: str | None) -> Path:
    if override:
        return Path(override)
    spec = importlib.util.find_spec("bfcl_eval")
    if not spec or not spec.submodule_search_locations:
        raise SystemExit("bfcl_eval is not installed. Run: pip install bfcl-eval")
    return Path(list(spec.submodule_search_locations)[0]) / "data"


def find_category_file(data_dir: Path, cat: str) -> tuple[Path, str]:
    """Return (path, version) of the newest BFCL_v<N>_<cat>.json (exact category match)."""
    pat = re.compile(rf"^BFCL_v(\d+)_{re.escape(cat)}\.json$")
    hits = [(int(m.group(1)), p) for p in data_dir.iterdir()
            if (m := pat.match(p.name))]
    if not hits:
        raise SystemExit(f"No BFCL data file for category '{cat}' in {data_dir}")
    ver, path = max(hits)
    return path, str(ver)


def load_entries(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        data = json.loads(text)
        return data if isinstance(data, list) else [data]


def build_prompt(entry: dict) -> str:
    q = entry["question"]
    msgs = q[0] if q and isinstance(q[0], list) else q  # single-turn: first (only) turn
    user = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "user")
    sys_extra = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "system")
    system = BFCL_SYSTEM.format(functions=json.dumps(entry.get("function", []), indent=1))
    if sys_extra:
        system += "\n" + sys_extra
    return f"{system}\n\nUser request:\n{user}"


def clean_reply(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.splitlines() if not l.strip().startswith("```")).strip()
    return text


# --------------------------------------------------------------------------------------
# BFCL: generation, scoring, reporting
# --------------------------------------------------------------------------------------
def select_entries(entries: list[dict], cat: str, args) -> list[dict]:
    """Pick the cases to run. --bfcl-sample: seeded random subset (same ids for both arms and
    across reruns, kept in dataset order); --bfcl-limit: first N; otherwise all."""
    if args.bfcl_sample:
        n = min(args.bfcl_sample, len(entries))
        rng = random.Random(f"{args.bfcl_seed}:{cat}")  # str seeds are deterministic
        return [entries[i] for i in sorted(rng.sample(range(len(entries)), n))]
    if args.bfcl_limit:
        return entries[: args.bfcl_limit]
    return entries


def model_dir_name(model: str) -> str:
    return model.replace("/", "_")


def generate_category(arm: str, cat: str, entries: list[dict], version: str, root: Path,
                      model: str, args) -> dict:
    res_dir = root / "result" / model_dir_name(model)
    res_dir.mkdir(parents=True, exist_ok=True)
    res_file = res_dir / f"BFCL_v{version}_{cat}_result.json"

    done: dict[str, dict] = {}
    if res_file.exists():  # resume after a crash / rerun
        for line in res_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    todo = [e for e in entries if e["id"] not in done]
    failures = 0
    lock = threading.Lock()

    def work(entry: dict) -> dict | None:
        t0 = time.time()
        try:
            out = call_serve(SERVE_URLS[arm], build_prompt(entry), args.bfcl_max_new_tokens, 0.0)
        except Exception as exc:
            log.error("[%s/%s] %s failed: %s", ARM_LABEL[arm], cat, entry["id"], exc)
            return None
        return {"id": entry["id"], "result": clean_reply(out.get("reply", "")),
                "latency": round(time.time() - t0, 2)}

    log.info("[%s] %s: generating %d/%d cases (%d already done)",
             ARM_LABEL[arm], cat, len(todo), len(entries), len(done))
    with ThreadPoolExecutor(max_workers=max(1, args.bfcl_concurrency)) as pool:
        futs = [pool.submit(work, e) for e in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            with lock:
                if rec is None:
                    failures += 1
                else:
                    done[rec["id"]] = rec
            if i % 25 == 0:
                log.info("[%s] %s: %d/%d", ARM_LABEL[arm], cat, i, len(todo))

    # Rewrite in dataset order. Failed cases are omitted (NOT written as empty strings, which
    # would be scored as correct abstentions in the 'irrelevance' category).
    ordered = [done[e["id"]] for e in entries if e["id"] in done]
    res_file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in ordered) + "\n",
                        encoding="utf-8")
    empty = sum(1 for r in ordered if not r["result"])
    return {"n_total": len(entries), "n_generated": len(ordered),
            "generation_failures": failures, "empty_replies": empty}


def bfcl_evaluate(root: Path, model: str, cats: list[str]) -> int:
    (root / ".env").touch(exist_ok=True)
    env = dict(os.environ, BFCL_PROJECT_ROOT=str(root))
    if shutil.which("bfcl"):
        cmd = ["bfcl", "evaluate", "--model", model, "--test-category", ",".join(cats),
               "--partial-eval"]
    else:
        cmd = [sys.executable, "-m", "bfcl_eval.eval_checker.eval_runner", "--model", model,
               "--test-category", *cats, "--partial-eval"]
    log.info("%s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, env=env).returncode


def find_score_file(root: Path, cat: str) -> Path | None:
    pat = re.compile(rf"^BFCL_v\d+_{re.escape(cat)}_score\.json$")
    files = sorted(p for p in (root / "score").rglob("*.json") if pat.match(p.name))
    return files[-1] if files else None


def read_scores(root: Path, cats: list[str]) -> dict:
    """Parse BFCL score files: line 1 is a summary {"accuracy","correct_count","total_count"};
    the remaining lines describe the failed cases (each with an "id")."""
    out: dict = {}
    for cat in cats:
        f = find_score_file(root, cat)
        if f is None:
            out[cat] = None
            continue
        try:
            lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            summ = json.loads(lines[0])
            failed = []
            for l in lines[1:]:
                try:
                    rec = json.loads(l)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and "id" in rec:
                    failed.append(rec["id"])
            out[cat] = {"correct": int(summ["correct_count"]), "total": int(summ["total_count"]),
                        "accuracy": float(summ["accuracy"]), "failed_ids": failed}
        except Exception as exc:
            log.error("Could not parse %s (%s). Format may differ in your bfcl-eval version.",
                      f, exc)
            out[cat] = None
    return out


def item_outcomes(root: Path, model: str, cat: str, version: str, score: dict | None):
    """Map case id -> 1 (correct) / 0 (wrong) for the cases that were scored.

    Rebuilt from the result file (ids scored) minus the failed ids in the score file, and
    cross-checked against the summary counts. Returns None if they do not reconcile, so a
    wrong pairing is never reported.
    """
    if not score:
        return None
    rf = root / "result" / model_dir_name(model) / f"BFCL_v{version}_{cat}_result.json"
    if not rf.exists():
        return None
    ids = []
    for line in rf.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                ids.append(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    failed = set(score["failed_ids"]) & set(ids)
    if len(ids) != score["total"] or len(ids) - len(failed) != score["correct"]:
        log.warning("[%s] per-case results do not reconcile with the score summary "
                    "(ids=%d total=%d correct=%d failed-ids=%d); skipping paired analysis for it.",
                    cat, len(ids), score["total"], score["correct"], len(failed))
        return None
    return {i: (0 if i in failed else 1) for i in ids}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def _fmt(correct: int, total: int) -> str:
    return f"{correct / total:.3f} ({correct}/{total})" if total else "n/a"


def print_bfcl_table(results: dict, cats: list[str]) -> None:
    arms = list(results)
    print("\n" + f"{'BFCL v4 (single-turn)':26s}" + "".join(f"{ARM_LABEL[a]:>22s}" for a in arms))
    totals = {a: [0, 0] for a in arms}
    for cat in cats:
        row = f"{cat:26s}"
        for a in arms:
            s = results[a]["scores"].get(cat)
            if not s:
                row += f"{'n/a':>22s}"
                continue
            totals[a][0] += s["correct"]
            totals[a][1] += s["total"]
            row += f"{_fmt(s['correct'], s['total']):>22s}"
        print(row)
    print(f"{'overall (micro)':26s}" + "".join(f"{_fmt(*totals[a]):>22s}" for a in arms))
    ci_cells = []
    for a in arms:
        lo, hi = wilson(*totals[a])
        ci_cells.append(f"{lo:.2f}-{hi:.2f}" if totals[a][1] else "n/a")
    print(f"{'  95% CI (Wilson)':26s}" + "".join(f"{c:>22s}" for c in ci_cells))
    for a in arms:
        gen = results[a]["generation"]
        fails = sum(g["generation_failures"] for g in gen.values())
        empties = sum(g["empty_replies"] for g in gen.values())
        if fails or empties:
            print(f"note [{ARM_LABEL[a]}]: {fails} generation failures (excluded from scoring), "
                  f"{empties} empty replies")


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs (b = ft only, c = base only)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_bootstrap_ci(diffs: list[int], iters: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap CI for the mean paired difference (resampling cases)."""
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(iters))
    return stats[int(0.025 * iters)], stats[int(0.975 * iters) - 1]


def _pair_row(pairs: list[tuple[int, int]]) -> dict:
    n = len(pairs)
    both = sum(1 for h, b in pairs if h and b)
    ft_only = sum(1 for h, b in pairs if h and not b)
    base_only = sum(1 for h, b in pairs if b and not h)
    return {"n": n, "both_ok": both, "ft_only": ft_only, "base_only": base_only,
            "both_wrong": n - both - ft_only - base_only,
            "delta": (ft_only - base_only) / n, "mcnemar_p": mcnemar_exact(ft_only, base_only)}


def paired_compare(hub: dict, base: dict, cats: list[str]) -> dict:
    """Compare finetuned vs base on the SAME cases (hub/base: {cat: {id: 0/1} | None})."""
    rows: dict = {}
    all_pairs: list[tuple[int, int]] = []
    for cat in cats:
        h, b = hub.get(cat), base.get(cat)
        if not h or not b:
            rows[cat] = None
            continue
        ids = sorted(set(h) & set(b))
        if len(ids) != len(h) or len(ids) != len(b):
            log.warning("[%s] arms scored different case sets; pairing the %d shared ids.",
                        cat, len(ids))
        if not ids:
            rows[cat] = None
            continue
        pairs = [(h[i], b[i]) for i in ids]
        rows[cat] = _pair_row(pairs)
        all_pairs += pairs
    overall = None
    if all_pairs:
        overall = _pair_row(all_pairs)
        lo, hi = paired_bootstrap_ci([h - b for h, b in all_pairs])
        overall["delta_ci95"] = [lo, hi]
    return {"categories": rows, "overall": overall}


def print_paired_table(paired: dict, cats: list[str]) -> None:
    print("\nPaired comparison: finetuned vs base on the same cases")
    print(f"{'category':22s}{'n':>5s}{'both ok':>9s}{'ft only':>9s}{'base only':>11s}"
          f"{'both wrong':>12s}{'delta':>9s}{'McNemar p':>11s}")

    def line(name: str, r: dict) -> str:
        return (f"{name:22s}{r['n']:>5d}{r['both_ok']:>9d}{r['ft_only']:>9d}{r['base_only']:>11d}"
                f"{r['both_wrong']:>12d}{r['delta']:>+9.3f}{r['mcnemar_p']:>11.3f}")

    for cat in cats:
        r = paired["categories"].get(cat)
        print(line(cat, r) if r else f"{cat:22s}{'n/a':>5s}")
    o = paired["overall"]
    if not o:
        return
    print(line("overall", o))
    lo, hi = o["delta_ci95"]
    print(f"Overall paired difference (finetuned - base): {o['delta']:+.3f}, 95% bootstrap CI "
          f"{lo:+.3f} to {hi:+.3f}; only {o['ft_only'] + o['base_only']} of {o['n']} cases "
          f"differ between the arms.")
    print("p-values are per comparison and not corrected for testing several categories.")


def run_bfcl(args) -> int:
    cats = [c.strip() for c in args.bfcl_categories.split(",") if c.strip()]
    if args.bfcl_sample and args.bfcl_limit:
        log.error("Use either --bfcl-sample or --bfcl-limit, not both.")
        return 2
    bad = [c for c in cats if c.startswith(UNSUPPORTED_CATEGORY_PREFIXES)]
    if bad:
        log.error("Unsupported (multi-turn/agentic) categories: %s. Single-turn only.", bad)
        return 2
    arms = ["hub", "base"] if args.bfcl_source == "both" else [args.bfcl_source]
    out = Path(args.out).resolve()
    data_dir = bfcl_data_dir(args.bfcl_data_dir)
    files = {c: find_category_file(data_dir, c) for c in cats}

    if not args.bfcl_score_only:
        for arm in arms:
            warmup(arm)

    results: dict = {}
    item_ok: dict = {}
    rc = 0
    for arm in arms:
        root = out / "bfcl" / arm
        (root / "result").mkdir(parents=True, exist_ok=True)
        gen_stats: dict = {}
        if not args.bfcl_score_only:
            for cat in cats:
                path, ver = files[cat]
                entries = select_entries(load_entries(path), cat, args)
                gen_stats[cat] = generate_category(arm, cat, entries, ver, root,
                                                   args.bfcl_model, args)
                if gen_stats[cat]["generation_failures"]:
                    rc = 1
        code = bfcl_evaluate(root, args.bfcl_model, cats)
        if code != 0:
            log.error("[%s] bfcl evaluate exited %d. If it says the model is unknown, pass "
                      "--bfcl-model <a prompt-mode (non -FC) model from BFCL's SUPPORTED_MODELS.md>.",
                      ARM_LABEL[arm], code)
            rc = 1
        scores = read_scores(root, cats)
        item_ok[arm] = {c: item_outcomes(root, args.bfcl_model, c, files[c][1], scores[c])
                        for c in cats}
        results[arm] = {"root": str(root), "generation": gen_stats, "scores": scores}

    # print_bfcl_table only reads counts; failed_ids stay in the JSON for reproducibility.
    print_bfcl_table(results, cats)
    paired = None
    if "hub" in item_ok and "base" in item_ok:
        paired = paired_compare(item_ok["hub"], item_ok["base"], cats)
        print_paired_table(paired, cats)
    else:
        log.info("Paired comparison needs both arms (--bfcl-source both).")
    dest = out / "bfcl_results.json"
    dest.write_text(json.dumps(
        {"categories": cats, "stand_in_model": args.bfcl_model,
         "sampling": {"sample": args.bfcl_sample, "seed": args.bfcl_seed,
                      "limit": args.bfcl_limit},
         "paired": paired, "results": results}, indent=2), encoding="utf-8")
    log.info("Wrote %s", dest)
    return rc


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Eval driver for Muse Glimmer (GPQA + BFCL v4)")
    p.add_argument("--bench", default="gpqa", choices=["gpqa", "bfcl", "both"],
                   help="Which benchmark(s) to run (default: gpqa, as before)")
    p.add_argument("--out", default="eval_out", help="Output dir for BFCL generations/scores")

    g = p.add_argument_group("GPQA (lm-eval)")
    g.add_argument("--lm", action="store_true", help="Kept for compatibility; GPQA is the default.")
    g.add_argument("--lm-tasks", default="gpqa_diamond_zeroshot")
    g.add_argument("--lm-source", default="both", choices=["hub", "base", "both"])
    g.add_argument("--lm-limit", type=int, default=None)
    g.add_argument("--lm-temp", type=float, default=0.3)

    b = p.add_argument_group("BFCL v4")
    b.add_argument("--bfcl-source", default="both", choices=["hub", "base", "both"],
                   help="hub = finetuned LoRA endpoint, base = clean base endpoint")
    b.add_argument("--bfcl-categories", default=DEFAULT_BFCL_CATEGORIES,
                   help="Comma-separated single-turn categories")
    b.add_argument("--bfcl-limit", type=int, default=None,
                   help="Use the FIRST N cases per category (file order; may be unrepresentative)")
    b.add_argument("--bfcl-sample", type=int, default=None,
                   help="Use a seeded RANDOM sample of N cases per category (same cases for both "
                        "arms and across reruns; reuses any already-generated cases)")
    b.add_argument("--bfcl-seed", type=int, default=0, help="Seed for --bfcl-sample (default 0)")
    b.add_argument("--bfcl-concurrency", type=int, default=2,
                   help="Parallel requests per endpoint (Modal may autoscale GPUs)")
    b.add_argument("--bfcl-max-new-tokens", type=int, default=512)
    b.add_argument("--bfcl-model", default=DEFAULT_BFCL_STAND_IN,
                   help="BFCL stand-in model name (prompt-mode) under which results are filed")
    b.add_argument("--bfcl-data-dir", default=None,
                   help="Override path to bfcl_eval/data (default: the installed package)")
    b.add_argument("--bfcl-score-only", action="store_true",
                   help="Skip generation; re-run `bfcl evaluate` on saved results")

    args = p.parse_args(argv)
    rc = 0
    if args.bench in ("gpqa", "both"):
        rc |= run_lm_eval(
            tasks=[x.strip() for x in args.lm_tasks.split(",") if x.strip()],
            source=args.lm_source, limit=args.lm_limit, temperature=args.lm_temp)
    if args.bench in ("bfcl", "both"):
        rc |= run_bfcl(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())