# ai-diagnosis

Experiments evaluating open LLMs on interactive medical diagnosis.

## AgentClinic baseline evaluation

`agentclinic/agentclinic_eval.py` runs a doctor model through the
[AgentClinic](https://github.com/SamuelSchmidgall/AgentClinic) MedQA
scenarios (Schmidgall et al., 2024) using Hugging Face Inference Providers.
Each case is a multi-turn simulated encounter: the doctor model asks the
patient questions, orders tests (`REQUEST TEST: ...`), and commits to a
diagnosis (`DIAGNOSIS READY: ...`). The patient and measurement agents are
played by a fixed simulator model (`Qwen/Qwen3-8B` by default).

### Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...   # https://huggingface.co/settings/tokens
```

Gated models (e.g. Gemma 3, MedGemma) require accepting the license on the
model's Hugging Face page first.

### Run

```bash
cd agentclinic
python3 agentclinic_eval.py                                   # 10 scenarios, Qwen3-8B doctor
python3 agentclinic_eval.py --doctor-model google/gemma-3-27b-it
python3 agentclinic_eval.py --doctor-model google/gemma-3-27b-it --num-scenarios all
```

The dataset (`agentclinic_medqa.jsonl`) downloads on first run. Outputs are
`agentclinic_<model>_results.csv` (one row per case) and
`agentclinic_<model>_transcripts.jsonl` (full dialogues). Re-running the same
command resumes an interrupted run; pass `--fresh` to start over. See
`python3 agentclinic_eval.py --help` for all options.

### Grading

`agentclinic/agentclinic_judge_prompt.py` holds the 5-axis LLM-as-judge rubric
(semantically correct, matches details, substantive addition, grounded in
evidence, critical error) used to grade the transcripts.

## Multi-turn SFT dataset

`data/sft_multiturn_kept.jsonl` holds 1,309 AgentClinic-style doctor/patient/measurement
dialogues (Gemma-3-27B doctor) built from MedQA and MedCaseReasoning, filtered to
correct, grounded diagnoses. See [data/README.md](data/README.md) for how it was built,
the keep rule, counts and licenses.

*Research code only — not for clinical use.*
