"""
Step 1 of on-policy distillation (OPD): turn the final multi-turn SFT set into
the case file the student interviews during training.

The student does not train on the 27B transcripts. It runs its own interviews
of the same 1,309 cases (the kept set from assemble_sft_dataset.py), so only
the case facts are needed here: demographics, the patient's facts, the
exam/test findings and the ground-truth diagnosis.

doctor_prompt is built by agentclinic_eval.build_doctor_instructions(), the
same instructions used for SFT generation and the AgentClinic eval, so it
cannot drift from them.

Input:  sft_multiturn_kept.jsonl, sft_cases_structured.jsonl
Output: opd_cases.jsonl   one case per line
Run:    python3 prepare_opd_cases.py [--push-to-hub USER/REPO]
        --push-to-hub uploads opd_cases.jsonl to a PRIVATE dataset repo, so the
        HF Job running train_opd.py can download it.
"""

import argparse
import json
import os
import re

from huggingface_hub import get_token

p = argparse.ArgumentParser()
p.add_argument("--push-to-hub", default=None, help="Private dataset repo to upload opd_cases.jsonl to")
args, _ = p.parse_known_args()

# agentclinic_eval builds InferenceClients at import and exits without HF_TOKEN.
# No API call is made here; it is only imported for its prompt builder and flatten().
os.environ.setdefault("HF_TOKEN", get_token() or "")
import agentclinic_eval as ac  # noqa: E402


def split_camel(s):
    """928 MedCaseReasoning diagnoses are CamelCase-joined ("DiffuseLargeBCellLymphoma").
    The teacher grades against this string, so restore the spaces."""
    if " " in s.strip():
        return s
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    return re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)


kept = {(r["source"], r["index"]) for r in map(json.loads, open("sft_multiturn_kept.jsonl"))}
out, n_split = [], 0
for line in open("sft_cases_structured.jsonl"):
    c = json.loads(line)
    if (c["source"], c["index"]) not in kept:
        continue
    o = c["OSCE_Examination"]
    pa = o["Patient_Actor"]
    findings = {}
    findings.update(ac.flatten(o.get("Physical_Examination_Findings", {})))
    findings.update(ac.flatten(o.get("Test_Results", {})))
    dx = split_camel(o["Correct_Diagnosis"])
    n_split += dx != o["Correct_Diagnosis"]
    demographics = pa.get("Demographics", "Unknown")
    out.append({
        "source": c["source"], "index": c["index"],
        "demographics": demographics, "patient_actor": pa, "findings": findings,
        "correct_diagnosis": dx,
        "doctor_prompt": ac.build_doctor_instructions(demographics, first_turn=True),
    })

if len(out) != len(kept):
    raise SystemExit(f"found {len(out)} of {len(kept)} kept cases in sft_cases_structured.jsonl")
with open("opd_cases.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote opd_cases.jsonl: {len(out)} cases "
      f"({sum(r['source'] == 'medqa' for r in out)} MedQA), {n_split} CamelCase diagnoses split")

if args.push_to_hub:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.push_to_hub, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj="opd_cases.jsonl", path_in_repo="opd_cases.jsonl",
                    repo_id=args.push_to_hub, repo_type="dataset")
    print(f"uploaded to {args.push_to_hub} (private)")
