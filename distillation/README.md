# On-policy distillation (OPD)

Trains **Gemma 4 E4B** (student) on its own AgentClinic-style patient interviews,
coached token by token by **Gemma 4 26B-A4B** (teacher). The two models share the
same 262,144-token vocabulary, which exact per-token distillation needs.

| File | What it does |
|---|---|
| `prepare_opd_cases.py` | Step 1: turns the 1,309 kept SFT cases (see [`../data`](../data)) into `opd_cases.jsonl`, the case facts the student interviews. |
| `train_opd.py` | Step 2: the training loop (rollout → teacher grade → reverse-KL loss, LoRA on the student). Self-contained [uv script](https://docs.astral.sh/uv/guides/scripts/) for Hugging Face Jobs. |
| `test_train_opd.py` | CPU checks for `train_opd.py` (tokenizers plus a tiny random Gemma 4 smoke test). |

## Loss, in short

- Exact full-vocabulary reverse KL, KL(student ‖ teacher), on every token the student generated as the doctor.
- A small supervised anchor (weight 0.05) on the final `DIAGNOSIS READY:` turn of successful interviews.
- Failed interviews: the loss stops right after `DIAGNOSIS READY:`, so the wrong diagnosis gets no loss.
- Success is the teacher's YES/NO grade of the final diagnosis against the ground truth.

The full design is in the docstring at the top of `train_opd.py`.

## Run

```bash
python3 prepare_opd_cases.py --push-to-hub USER/opd-cases
hf jobs uv run --flavor h200 --timeout 24h --secrets HF_TOKEN --detach \
    train_opd.py --cases-repo USER/opd-cases --push-to-hub USER/gemma4-e4b-opd
```

Needs one H200 (about 77 GB of bf16 weights for student, teacher and simulator), or two GPUs with `--teacher-device cuda:1`.

## Not yet in this repo

`prepare_opd_cases.py` and `test_train_opd.py` import helpers that sit next to them in
the original working folder: `agentclinic_eval.py` (in [`../agentclinic`](../agentclinic)),
`generate_sft_dialogues.py` and `agentclinic_style_sft.py`. `prepare_opd_cases.py` also
reads `sft_cases_structured.jsonl`. Those SFT pipeline scripts aren't published yet, so
run these two from a folder that has them. `train_opd.py` itself needs none of them.
