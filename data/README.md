# Multi-turn SFT data (reproduction): data card

Doctor (teacher): Gemma-3-27B-it. Patient and measurement simulators: Gemma-3-4B-it.
Grader: Opus 5.5, every dialogue graded (no sampling), rubric `agentclinic_judge_prompt.py`.

Keep rule: ground truth is a diagnosis, AND semantically correct, AND grounded in evidence,
AND no critical error, AND either
- no substantive addition ("strict"), or
- the substantive addition is only a hedge ("hedged"): the answer qualifies itself or lists
  candidates ("likely X", "X or Y") but adds no unsupported cause, subtype or finding.
  Opus re-checked all 156 correct-but-flagged dialogues and passed 72.

Simulator problems are allowed.

## Counts by stage

| Stage | MedQA | MedCaseReasoning | Total |
|---|---:|---:|---:|
| Eligible after filtering and decontamination | 727 | 11600 | 12327 |
| Dropped at case splitting | 33 | 525 | 558 |
| Dialogues generated | 695 | 11075 | 11770 |
| Graded by Opus | 695 | 11075 | 11770 |
| Semantically correct | 275 (39.6%) | 1383 (12.5%) | 1658 (14.1%) |
| **Kept** | **214** (30.8%) | **1095** (9.9%) | **1309** (11.1%) |
| of which hedged-correct | 6 | 66 | 72 |
| Kept with a simulator problem | 119 | 497 | 616 |

## Mix compared with the external run

| | This run | External run |
|---|---:|---:|
| Kept dialogues | 1309 | ~4,600 |
| MedQA share | 16.3% | 27.8% |
| MedCaseReasoning share | 83.7% | 72.2% |

Median doctor turns in kept dialogues: 11.

## Share of graded dialogues where each axis is true

| Axis | MedQA | MedCaseReasoning |
|---|---:|---:|
| ground_truth_is_diagnosis | 98.6% | 98.8% |
| semantically_correct | 39.6% | 12.5% |
| matches_details | 31.7% | 7.7% |
| substantive_addition | 18.4% | 25.9% |
| grounded_in_evidence | 77.7% | 77.8% |
| critical_error | 33.4% | 39.1% |
| simulator_problem | 60.4% | 56.8% |

## Files

- `sft_multiturn_kept.jsonl`: the training set, one dialogue per line, `messages` in chat format
  (doctor = assistant; patient and measurement replies = user).
- `sft_multiturn_grades.csv`: every graded dialogue with all seven grades, the keep decision and the grader's analysis.
- `sft_multiturn_flags.csv`: 2 kept dialogues whose `final_diagnosis` field looks malformed.
  The grader read the full transcript, so these are still kept; check them by hand if needed.

## Known limits

- MedCaseReasoning cases are rare-disease case reports; the 27B doctor rarely names them exactly,
  so the MCR keep rate is far below MedQA's and the mix differs from the external run.
- Grades come from one Opus pass with no second grader; no human spot check has been done yet.

## Sources and licenses

- MedQA (USMLE 4-option): [`GBaker/MedQA-USMLE-4-options`](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), CC-BY-4.0.
  Original: Jin et al., 2021, "What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams".
- MedCaseReasoning: [`zou-lab/MedCaseReasoning`](https://huggingface.co/datasets/zou-lab/MedCaseReasoning), MIT.
- Dialogues were generated with Gemma 3 models and are subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
- MedQA cases appearing in the extended AgentClinic set were excluded (decontamination).

Synthetic research data only: not medical advice and not for clinical use.
