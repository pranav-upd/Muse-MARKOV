<div align="center">

# MARKOV

### Model Assisted Reinforcement and Knowledge Improvement

**A reproducible GRPO loop that turns Muse Glimmer (30B) into a better tool-calling agent.**

![Base model](https://img.shields.io/badge/base-Muse%20Glimmer%2030B-2B44C9)
![Training](https://img.shields.io/badge/training-GRPO-0F7C6B)
![Compute](https://img.shields.io/badge/compute-Modal%20%C2%B7%20RTX%20PRO%206000%20%C2%B7%20FP4-555)
![BFCL](https://img.shields.io/badge/BFCL%20v4-0.63%20%E2%86%92%200.725-brightgreen)
![Significance](https://img.shields.io/badge/McNemar-p%20%E2%89%88%200.0013-brightgreen)
[![Live demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/pranavupadhyaya52/muse_markov-benchtest)

[Results](#results) · [How it works](#how-it-works) · [Quickstart](#quickstart) · [Try the model](#try-the-model) · [Live demo](https://huggingface.co/spaces/pranavupadhyaya52/muse_markov-benchtest) · [Video](https://youtu.be/DikmQtIu3Us) · [Limitations](#limitations-and-next-steps)

</div>

---

> **TL;DR:** Drop in a dataset with checkable answers. Get back a finetuned model with judged rewards, rotating checkpoints, published weights and independent benchmark scores.
>
> On BFCL v4 function calling (200 cases, deterministic scoring), the finetuned model scores **0.725 vs 0.63** for the base model: **+9.5 points**, with 26 finetuned-only wins against 7 base-only wins.
>
> **Live demo:** compare base vs finetuned in your browser on the [Hugging Face Space](https://huggingface.co/spaces/pranavupadhyaya52/muse_markov-benchtest).

<div align="center">

<a href="https://youtu.be/DikmQtIu3Us">
  <img src="https://img.youtube.com/vi/DikmQtIu3Us/maxresdefault.jpg" alt="MARKOV - Model Assisted Reinforcement and Knowledge Improvement (video walkthrough)" width="640">
</a>

<sub>▶ <a href="https://youtu.be/DikmQtIu3Us">Watch the video walkthrough on YouTube</a></sub>

</div>

## Table of contents

- [Why this exists](#why-this-exists)
- [Results](#results)
- [How it works](#how-it-works)
- [Built on Muse](#built-on-muse)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Try the model](#try-the-model)
- [Limitations and next steps](#limitations-and-next-steps)

## Why this exists

Improving a model with a well-designed dataset is still manual, ad hoc and hard to reproduce. As a contract software engineer at Turing.com, I see how much of that loop depends on one-off scripts.

MARKOV is the harness I wished existed: **one pipeline from dataset to trained weights to scored benchmarks**, using Meta's models at every stage.

## Results

All comparisons are **base vs finetuned** under identical harnesses, prompts and seeds. The only difference between the two arms is the LoRA weights.

### BFCL v4 function calling

100 cases per category, seeded random sample (seed 7).

| Weights | multiple | parallel | overall (micro) |
|:--------|:--------:|:--------:|:---------------:|
| Base | 55/100 (0.55) | 71/100 (0.71) | 126/200 (**0.63**) |
| **Finetuned** | **66/100 (0.66)** | **79/100 (0.79)** | **145/200 (0.725)** |

**Paired comparison on the same cases**

| Metric | Value |
|:-------|:------|
| Difference | **+0.095** |
| Finetuned-only wins / base-only wins | 26 / 7 |
| Both wrong | 48 |
| McNemar p-value | ≈ 0.0013 |
| 95% bootstrap CI | [+0.04, +0.15] |

<details>
<summary><b>Method and caveats</b></summary>

- Generations come from the Modal `serve` endpoints at temperature 0.
- Scoring uses the official `bfcl evaluate` (deterministic AST matching, **no judge model**).
- Results are filed under the stand-in model name `meta-llama/Llama-3.1-8B-Instruct` (prompt-mode filing only). The generations are Glimmer's, so the numbers are comparable **across the two arms** but **not against the public BFCL leaderboard**.

</details>

### GPQA Diamond, zero-shot

198 questions.

| Weights | Score |
|:--------|:-----:|
| Base (`unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit`) | 0.4141 |
| Finetuned (`pranavupadhyaya52/muse_glimmer_agentFT`) | 0.4343 |

> [!NOTE]
> The +0.02 gap (about 4 questions) is inside sampling noise at n=198. Read it as "no sign of regression on hard science reasoning", **not** as an improvement.

## How it works

```
dataset ──► GRPO training (Modal, RTX PRO 6000, FP4) ──► rotating checkpoints ──► LoRA on the Hub
                 ▲                                                                    │
        4-term reward                                                                 ▼
 (format, shape, ground truth, Muse Wonder judge)                    served on Modal ──► independent evals
```

| Stage | What happens |
|:------|:-------------|
| **Training** | GRPO on 10,000 agentic trajectories from `PersonalAILab/AFM-WebAgent-RL-Dataset`, FP4 on a single RTX PRO 6000 via Modal |
| **Rewards** | Four terms so no single signal dominates: channel format, answer shape, verifiable ground truth, and a live Muse Wonder judge for rollouts without a checkable answer |
| **Output format** | Private reasoning on the `to=self` channel, the answer on `to=user` |
| **Tracking** | Langfuse for training and eval runs |
| **Continuity** | Rotating checkpoints, with the LoRA pushed to the Hub under a stable name |
| **Evaluation** | `eval.py` runs GPQA Diamond through the EleutherAI `lm_harness` and BFCL v4 through the served endpoints with the official scorer |

## Built on Muse
| Model | Role |
|:------|:-----|
| **Muse Glimmer (30B)** | The model being finetuned |
| **Muse Wonder** | Judge for training rollouts, called through the API |
| **Muse Spark 1.3** | Coding assistant (via OpenCode) for building the pipeline, debugging training runs and hardening the endpoints |

## Repository layout

| File | Role |
|:-----|:-----|
| [`starter.py`](starter.py) | Training, inference, judge, Langfuse harness, serving endpoints |
| [`eval.py`](eval.py) | Eval driver: GPQA Diamond via `lm_harness`; BFCL v4 via the `serve` endpoints and `bfcl evaluate` |
| [`eval_out.zip`](eval_out.zip) | Raw evaluation outputs |

## Quickstart

```bash
python starter.py --dry-run                      # validate config, no GPU
python -m modal run starter.py --action train    # GRPO training on Modal
python -m modal deploy starter.py                # publish the endpoints and lm_harness app
python eval.py --lm-limit 10                     # smoke test: 10 samples
python eval.py                                   # GPQA Diamond, both weights
```

**Requirements**

- A Modal account
- A Hugging Face token (for publishing weights)
- API access to Muse Wonder (for the judge reward)

## Try the model

The finetuned LoRA is attached at request time on Modal, so **no local GPU is needed**.

**Live demo:** the [muse_markov-benchtest Space](https://huggingface.co/spaces/pranavupadhyaya52/muse_markov-benchtest) runs a side-by-side base vs finetuned benchmark comparison through these endpoints using the EleutherAI `lm_eval` harness (generation benchmarks such as GSM8K), with a results table, chart and downloadable Markdown / CSV / JSON reports.

Or call the endpoints directly. Two endpoints share one contract:

| Endpoint | Weights |
|:---------|:--------|
| `https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run` | Finetuned |
| `https://pranavupadhyaya52--muse-glimmer-grpo-serve-base.us-east.modal.run` | Clean base |

**Request:** `POST {"prompt": str, "max_new_tokens"?: int, "temperature"?: float}`
**Response:** `{"reply": str, "reasoning": str}`

`temperature: 0` means greedy decoding. The first request after idle time may be slow while the container starts.

<details open>
<summary><b>curl</b></summary>

```bash
curl -X POST https://pranavupadhyaya52--muse-glimmer-grpo-serve.us-east.modal.run/ \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"What is 48 + 48/2? Reply with the number only.","max_new_tokens":256,"temperature":0.7}'
```

</details>

<details>
<summary><b>Python</b></summary>

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

</details>

## Limitations and next steps

| Limitation | Detail |
|:-----------|:-------|
| **One task family scored** | BFCL covers single-turn function calling. Multi-step agency (TerminalBench, OSWorld style) is instrumented but not yet scored. |
| **One finetune, one seed family** | The GPQA delta is within noise. The BFCL delta is statistically significant but should be replicated with a different sampling seed before it is treated as settled. |
| **Judge and verifier not cross-checked** | The Wonder judge scores training rollouts and the AFM eval split, while BFCL verdicts come from AST matching only. |
| **Hardware-bound** | Training, 30B inference and parallel eval containers compete for the same GPU budget, which limits iteration speed and eval scale. |

**Next**

- [ ] Score the remaining BFCL single-turn categories
- [ ] Rerun GPQA with a second seed
- [ ] Replicate BFCL with a different sampling seed
- [ ] Try one multi-step TerminalBench slice to see whether the learned channel discipline holds over long-horizon tool use
