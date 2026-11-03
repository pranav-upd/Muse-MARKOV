# Muse Glimmer Agent FT — GRPO Finetuning + Eval

[![Demo video](https://img.youtube.com/vi/Id0ilSVboPM/0.jpg)](https://youtu.be/Id0ilSVboPM)

Maya is a backend developer racing a deadline: three vendor APIs to wire up before lunch. She pastes the docs into a chat with this finetuned Glimmer, and gets back exact `register_user(...)`-style calls — the kind BFCL's AST checker scores 145/200, against 126/200 for the base model.
GRPO finetuning of Muse Glimmer (30B) into a working software agent:
Modal training/inference (RTX PRO 6000, FP4), Langfuse-tracked eval on
`PersonalAILab/AFM-WebAgent-RL-Dataset`, Muse Wonder as LLM judge,
LoRA published to `pranavupadhyaya52/muse_glimmer_agentFT`, and an
lm-eval harness (`eval.py`) for independent benchmarking.

| File | Role |
|---|---|
| `starter.py` | Training, inference, judge, Langfuse harness, serving endpoints |
| `eval.py` | Eval driver: GPQA Diamond via `lm_harness` on Modal GPU; BFCL v4 via `serve` endpoints + official `bfcl evaluate` |

## Benchmark: GPQA Diamond zero-shot (198 questions)

| Weights | Score |
|---|---|
| Base (`unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit`) | 0.4141 |
| Finetuned (`pranavupadhyaya52/muse_glimmer_agentFT`) | 0.4343 |

Method: EleutherAI harness on a single RTX PRO 6000 via `lm_harness`
(`python eval.py --lm-tasks gpqa_diamond_zeroshot --lm-source both`).
Both arms share prompts, seed, batching, and loader — the only delta is
the LoRA weights, attached onto a clean base load (a double-stacked
adapter bug found mid-project was fixed before these numbers).

The finetuned score is higher by ~0.02 (~4 questions), which hypothetically
suggests a small gain and possibly rules out regression on hard science
reasoning — though at n=198 the gap sits inside statistical noise, so it
is possibly just sampling variation. A second slice would be needed before
reading more into it.

```powershell
python eval.py --lm-tasks gpqa_diamond_zeroshot --lm-source both
```

## Benchmark: BFCL v4 function calling (multiple + parallel, 100 cases each)

| Weights | multiple | parallel | overall (micro) |
|---|---|---|---|
| Base | 55/100 (0.55) | 71/100 (0.71) | 126/200 (0.63) |
| Finetuned | 66/100 (0.66) | 79/100 (0.79) | 145/200 (0.725) |

Paired comparison on the same cases: +0.095 for the finetuned arm
(26 ft-only wins vs 7 base-only, 48 both-wrong), McNemar p ≈ 0.0013,
95% bootstrap CI [+0.04, +0.15].

Method: generations via the Modal `serve` endpoints (both arms, temp 0),
scored with the official `bfcl evaluate` (deterministic AST matching — no
judge model involved). Seeded random sample of 100 per category (seed 7);
results filed under stand-in model `meta-llama/Llama-3.1-8B-Instruct`
(prompt-mode filing only — the generations are Glimmer's, so numbers are
comparable across arms but not against the public leaderboard).

Quickstart:

```powershell
python starter.py --dry-run                                  # validate, no GPU
python -m modal run starter.py --action train                # GRPO on Modal
python -m modal deploy starter.py                            # publish endpoints + lm_harness
python eval.py --lm-limit 10                                 # smoke: 10 samples
python eval.py                                               # GPQA Diamond, both weights
```

---

## 1. Muse Glimmer, Rewired for Action

**The problem.** Base models chat; they don't *do*. An agent that can't reliably act in a terminal, a repo, or a browser is a demo, not a tool. Closing that gap usually takes pretraining-scale budgets — out of reach for nearly everyone in this room.

**What we built.** A complete, reproducible reinforcement-learning loop that teaches a Meta model to act. Muse Glimmer (30B) trains with GRPO on 10,000 agentic trajectories, guided by four reward terms: channel-format, answer-shape, verifiable ground truth, and a live Muse Wonder judge scoring every rollout in real time. Around it, production-grade plumbing: FP4 training on RTX PRO 6000 via Modal, rotating checkpoints, and one-click Hub publishing. Early evidence comes from two head-to-head runs — BFCL function-calling (below) and a GPQA Diamond probe — all base versus finetuned under identical harnesses and seeds, with the finetuned arm ahead in both, though the GPQA gap is small enough to be sampling variation.

**Why it matters.** The weights are the demo; the *loop* is the product. Any team with tasks and a way to check answers can reuse this pipeline to mint their own specialist agents — ops runbooks, research triage, code repair — for the price of a training run, not a pretraining program.

**Why this could be Frontier work.** We didn't prompt a model and hope. Here is the loop — training, judging, checkpointing, publishing, independent scoring — and here are the early results: a finetuned agent that writes valid function calls more often than the base it started from. That's what happens when Meta's models meet the real world: the loop is open, reproducible, and waiting for the next domain.

---

## 2. Built on Muse, End to End

This project runs on three Muse models, each doing what it does best.

**Muse Glimmer (30B)** is the athlete — the finetuning substrate. Through GRPO on 10,000 agentic trajectories, it learns what its pretraining never taught it: to act. Private reasoning on one channel, decisive answers on another, tool calls in live sandboxes instead of chat.

**Muse Wonder** is the referee — called via API inference to judge every rollout in real time. Where answers are verifiable, ground truth rules; everywhere else, Wonder's scores become the reward signal that shapes the policy. No human labels, no waiting — the loop grades itself at machine speed.

**Muse Spark 1.3** is the engineer — pair-programming inside OpenCode through the entire build. It scaffolded the training pipeline, diagnosed dead training steps from raw logs, verified the tokenizer against the model card, wired the Harbor eval harness, and hardened the deployment endpoints. Every debugging turnaround in this project has Spark's fingerprints on it.

Together: Glimmer learns, Wonder judges, Spark builds — a complete agent pipeline where Meta's models don't just participate, they staff every role.

---

## 3. Limitations and next steps

- **One task family scored.** BFCL covers single-turn function calling only; TerminalBench/OSWorld-style multi-step agency is instrumented but unscored.
- **One finetune, one seed family.** Both reported gaps are single runs — the GPQA delta (+0.02) sits inside sampling noise, and even the significant BFCL delta (+0.095, p ≈ 0.0013) deserves a replication with a different sampling seed before it hardens into a claim.
- **30B weights, 96GB bottleneck.** Every stage — training, per-trial inference, parallel eval containers — competes for the same scarce accelerator, so iteration speed and eval scale are capped by hardware access, not by design.
- **Judge coverage is partial.** The Wonder judge scores training rollouts and the AFM eval split, but BFCL verdicts are pure AST matching — judge quality and verifier quality are never cross-checked against each other.
- **Next:** score the remaining BFCL single-turn categories, rerun GPQA with a second seed, then attempt one multi-step TerminalBench slice to see if the channel discipline learned in training survives long-horizon tool use.

---

## Inference

The finetuned model is served from Modal (Hub LoRA attached at request
time — no local GPU needed). Two endpoints, same contract:

| Endpoint | Weights |
|---|---|
| `https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run` | Finetuned (`muse_glimmer_agentFT` LoRA) |
| `https://pranavupadhyaya52--muse-glimmer-grpo-serve-base.us-east.modal.run` | Clean base |

POST `{"prompt", "max_new_tokens"?, "temperature"?}` →
`{"reply", "reasoning"}` (the `to=user` / `to=self` channels).
`temperature: 0` means greedy.

```powershell
$body = @{
  prompt           = "What is 48 + 48/2? Reply with the number only."
  max_new_tokens   = 256
  temperature      = 0.7
} | ConvertTo-Json
Invoke-RestMethod `
  -Uri "https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run/" `
  -Method Post -Body $body -ContentType "application/json"
```

```python
import json, urllib.request

body = json.dumps({
    "prompt": "What is 48 + 48/2? Reply with the number only.",
    "max_new_tokens": 256,
    "temperature": 0.7,
}).encode()
req = urllib.request.Request(
    "https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run/",
    data=body, headers={"Content-Type": "application/json"},
)
print(json.load(urllib.request.urlopen(req, timeout=300)))
```

```bash
curl -X POST \
  https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run/ \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 48 + 48/2? Reply with the number only.","max_new_tokens":256,"temperature":0.7}'
```
