# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.6",
#   "transformers>=5.16.1",
#   "peft>=0.18",
#   "accelerate>=1.10",
#   "huggingface_hub>=1.0",
# ]
# ///
"""
Step 2 of on-policy distillation (OPD): train Gemma 4 E4B (the student, or
"resident") on its OWN patient interviews, coached token by token by Gemma 4
26B-A4B (the teacher).

Why Gemma 4 26B-A4B as the teacher: per-token distillation over the whole
vocabulary needs the student and teacher to share a tokenizer. Gemma 4 26B-A4B
has exactly the same 262,144-token vocabulary as Gemma 4 E4B. Gemma 3 27B does
not: 6,206 token ids differ, and its chat-turn tokens are different.

One training step:
1. Rollout. The student interviews --rollouts-per-step cases from opd_cases.jsonl
   using the AgentClinic protocol (ask a question / REQUEST TEST: / DIAGNOSIS READY:),
   sampling at temperature 1 with no top-k/top-p, so its tokens follow its own
   distribution exactly. The patient and measurement simulators are Gemma-3-4B
   (greedy) with the same prompts as SFT generation. After --max-turns doctor
   turns without a diagnosis, one forced-diagnosis turn is asked for.
2. Grade. The teacher reads the ground truth and the student's final diagnosis
   and answers YES/NO (same definition as the rubric's "semantically correct").
   success = P(YES) > P(NO). An interview with no diagnosis is a failure.
3. Coach. The teacher scores the exact token sequence the student produced. On
   every student-generated token, the loss is the exact reverse KL over the full
   vocabulary:
       KL(student || teacher) = sum_v p_s(v) * (log p_s(v) - log p_t(v))
   Reverse KL is mode-seeking: it pulls the student onto the teacher's single
   highest-confidence choice rather than spreading mass over all of the
   teacher's options.
4. Masks.
   - Successful interviews: KL on every doctor turn, plus a small supervised
     "anchor" (cross-entropy on the student's own tokens, weight --anchor-weight)
     on the final DIAGNOSIS READY turn only.
   - Failed interviews: KL on every doctor turn, but in the final turn the loss
     stops right after the "DIAGNOSIS READY:" string, so the wrong diagnosis text
     (and anything after it) gets no loss. If the student never wrote a
     diagnosis line there is nothing to cut.
   Patient, measurement and prompt tokens never get loss.

The student is trained with LoRA (same r/alpha/target modules as the earlier SFT
recipe, restricted to the language model). The teacher and simulator are frozen.

Hardware: student 8.0B + teacher 25.8B + simulator 4.3B params in bf16 need
about 77 GB of weights plus activations. Use one H200 (141 GB), or two GPUs with
--teacher-device cuda:1.

Run (HF Jobs; the cases file is downloaded from the private dataset repo):
  hf jobs uv run --flavor h200 --timeout 24h --secrets HF_TOKEN --detach \\
      train_opd.py --cases-repo USER/opd-cases --push-to-hub USER/gemma4-e4b-opd
Outputs in --output-dir: adapter checkpoints, metrics.jsonl (one line per
step) and rollouts.jsonl (every interview with its grade).
"""

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--student", default="google/gemma-4-E4B-it")
    p.add_argument("--teacher", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--simulator", default="google/gemma-3-4b-it")
    p.add_argument("--cases", default="opd_cases.jsonl", help="Local case file from prepare_opd_cases.py")
    p.add_argument("--cases-repo", default=None, help="Private HF dataset repo holding opd_cases.jsonl (for HF Jobs)")
    p.add_argument("--student-device", default="cuda:0")
    p.add_argument("--teacher-device", default="cuda:0")
    p.add_argument("--simulator-device", default="cuda:0")
    p.add_argument("--epochs", type=float, default=1.0, help="Passes over the cases (sets the step count)")
    p.add_argument("--rollouts-per-step", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--anchor-weight", type=float, default=0.05, help="Weight of the supervised anchor on successful DIAGNOSIS READY turns")
    p.add_argument("--max-turns", type=int, default=20, help="Doctor turns before the forced-diagnosis turn (AgentClinic default 20)")
    p.add_argument("--doctor-max-new-tokens", type=int, default=256)
    p.add_argument("--simulator-max-new-tokens", type=int, default=200)
    p.add_argument("--max-seq-len", type=int, default=8192, help="An interview that would grow past this ends as a failure")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-regex", default=r".*language_model.*layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--output-dir", default="gemma4-e4b-opd")
    p.add_argument("--push-to-hub", default=None, help="Private model repo for checkpoints and logs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# AgentClinic protocol: copied from agentclinic_eval.py so this file runs on its
# own in an HF Job. test_train_opd.py checks that every copy still matches.
# ---------------------------------------------------------------------------

TEST_RE = re.compile(r"REQUEST TEST:\s*(.+)", re.IGNORECASE)
DIAGNOSIS_RE = re.compile(
    r"(?:(?:DIAGNOSIS READY|FINAL DIAGNOSIS)\s*:\s*|(?:MY DIAGNOSIS IS|THE DIAGNOSIS IS)\s*:?\s*)(.+)",
    re.IGNORECASE,
)
FREE_TEXT_DIAGNOSIS_RE = re.compile(
    r"\b(?:most likely has|most likely is|(?:was|is|has been) diagnosed with|"
    r"diagnosed with|consistent with a diagnosis of)\s+(.+?)(?:[.\n]|$)",
    re.IGNORECASE,
)
HEDGE_RE = re.compile(
    r"\b(?:suspicious for|suspect|differential|possible|possibly|may have|"
    r"might be|could be|rule out|considering)\b",
    re.IGNORECASE,
)
LEAKED_TAG_RE = re.compile(r"\[(?:doctor|patient|measurement)\]", re.IGNORECASE)
RESULTS_PREFIX_RE = re.compile(r"^results:\s*", re.IGNORECASE)
LEADING_LABEL_RE = re.compile(r"^\[\w+\]\s*")
QUOTE_CHARS = "\"'“”‘’ "
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
LABEL_SPLIT_RE = re.compile(r"\s*[,;]\s*|\s+and\s+")


def find_diagnosis(text):
    """Same order as agentclinic_eval.run_scenario: an explicit diagnosis line wins;
    otherwise the narrow free-text fallback, only if the turn has no test order,
    question or hedge. Returns (diagnosis, char offset where the diagnosis text
    starts) or (None, None)."""
    m = DIAGNOSIS_RE.search(text)
    if m:
        return m.group(1).strip(), m.start(1)
    if TEST_RE.search(text) or "?" in text or HEDGE_RE.search(text):
        return None, None
    m = FREE_TEXT_DIAGNOSIS_RE.search(text)
    if m:
        return m.group(1).strip(), m.start(1)
    return None, None


def normalize_for_comparison(text):
    text = LEADING_LABEL_RE.sub("", text.strip())
    return re.sub(r"\s+", " ", text.strip(QUOTE_CHARS)).lower()


def looks_like_echo(reply, source_text):
    return normalize_for_comparison(reply) == normalize_for_comparison(source_text)


def visible_messages(history, speaker):
    messages = []
    for entry in history:
        if speaker not in entry["visible_to"]:
            continue
        role = "assistant" if entry["speaker"] == speaker else "user"
        content = entry["content"] if role == "assistant" else f"[{entry['speaker']}] {entry['content']}"
        messages.append({"role": role, "content": content})
    return messages


def patient_messages(history, patient_actor):
    facts = "\n".join(f"{k}: {v}" for k, v in patient_actor.items() if k != "Demographics")
    instructions = (
        "You are a patient at a doctor's office, played for a medical training simulation. "
        "Answer in 1-3 sentences of first-person dialogue, staying strictly consistent with "
        "the facts below. Don't reveal anything the doctor hasn't asked about, and never "
        "mention or guess a diagnosis. Respond with the dialogue only -- no reasoning, no "
        "narration. /no_think\n\nYour facts:\n" + facts
    )
    messages = visible_messages(history, "patient")
    messages[0] = {"role": "user", "content": instructions + "\n\n" + messages[0]["content"]}
    return messages


def measurement_messages(requested_test, findings):
    findings_text = "\n".join(f"{k}: {v}" for k, v in findings.items()) or "(none recorded)"
    instructions = (
        "You are a clinical measurement reader for a medical training simulation. You are "
        "given a fixed set of recorded findings below. Respond with ONLY the matching "
        "finding(s), formatted as `RESULTS: <finding>`. If nothing below matches the request, "
        "respond exactly `RESULTS: Not documented for this case.` Never invent a finding that "
        f"isn't listed. /no_think\n\nRecorded findings:\n{findings_text}\n\nRequested: {requested_test}"
    )
    return [{"role": "user", "content": instructions}]


def _label_key(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def measurement_label_fix(reply, findings):
    """From generate_sft_dialogues.py: Gemma-3-4B sometimes answers with a finding's
    name instead of its value; substitute the recorded value."""
    body = RESULTS_PREFIX_RE.sub("", reply.strip()).strip().strip(".")
    by_label = {}
    for full_key, value in findings.items():
        by_label[_label_key(full_key)] = (full_key, value)
        by_label[_label_key(full_key.split(" > ")[-1])] = (full_key, value)
    parts = [p for p in LABEL_SPLIT_RE.split(body) if p]
    if parts and all(_label_key(p) in by_label for p in parts):
        hits = [by_label[_label_key(p)] for p in parts]
        return "RESULTS: " + "; ".join(f"{k.split(' > ')[-1].replace('_', ' ')}: {v}" for k, v in hits)
    return reply


# The forced-diagnosis request goes into the same conversation as a user turn, so
# that the whole interview is one token sequence the teacher can score in one pass.
# Wording follows agentclinic_eval.force_diagnosis_turn().
FORCE_PROMPT = (
    "[system] You have used up all your allotted turns. You must now state your final "
    "diagnosis. Do not ask any question. Do not order any test. Respond with only this, "
    "filled in, and nothing else:\nDIAGNOSIS READY: <diagnosis>"
)

GRADE_PROMPT = (
    "You are grading a simulated doctor's final diagnosis against the ground truth.\n\n"
    "Ground truth diagnosis: {gt}\n"
    "Doctor's final diagnosis: {fd}\n\n"
    "Answer YES if the doctor's final diagnosis names the same underlying disease or condition "
    "as the ground truth (synonyms, different wording, or extra correct detail are fine). "
    "Otherwise answer NO. Reply with exactly one word: YES or NO."
)


# ---------------------------------------------------------------------------
# Token bookkeeping: the interview is kept as the exact token ids the student saw
# and sampled, so the teacher scores precisely the student's own tokens.
# ---------------------------------------------------------------------------

class ChatFormat:
    """Builds token ids for the student's chat template piece by piece. Checked
    in the test to give the same ids as rendering the whole conversation."""

    def __init__(self, tok):
        self.tok = tok
        u = [{"role": "user", "content": "U"}]
        ua = u + [{"role": "assistant", "content": "A"}]
        g0 = self.render(u, gen=True)
        self.s_ua = self.render(ua)
        assert self.s_ua.startswith(g0 + "A"), "chat template is not prefix-stable"
        self.close_ids = self.enc(self.s_ua[len(g0) + 1:])  # e.g. "<turn|>\n"
        self.end_turn_id = self.close_ids[0]
        self.stop_ids = sorted({self.end_turn_id, tok.eos_token_id})
        self._ua = ua

    def render(self, msgs, gen=False):
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

    def enc(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def start(self, prompt):
        return self.enc(self.render([{"role": "user", "content": prompt}], gen=True))

    def after_doctor(self, sampled):
        """Ids that close a doctor turn the student just sampled."""
        if sampled and sampled[-1] == self.end_turn_id:
            return self.close_ids[1:]
        return self.close_ids

    def user_turn(self, content):
        s = self.render(self._ua + [{"role": "user", "content": content}], gen=True)
        assert s.startswith(self.s_ua)
        return self.enc(s[len(self.s_ua):])


@dataclass
class Episode:
    case: dict
    ids: list
    spans: list = field(default_factory=list)   # (start, end) of each sampled doctor turn in ids
    texts: list = field(default_factory=list)   # decoded doctor turns (special tokens kept)
    history: list = field(default_factory=list)
    done: bool = False
    forced: bool = False
    truncated: bool = False
    final_diagnosis: str = None
    diag_char: int = None    # where the diagnosis text starts in texts[-1]
    p_yes: float = None
    success: bool = False


def tokens_before_char(tok, ids, char_pos):
    """How many leading tokens of `ids` decode to text ending at or before char_pos."""
    k = 0
    for i in range(len(ids)):
        if len(tok.decode(ids[: i + 1], skip_special_tokens=False)) <= char_pos:
            k = i + 1
        else:
            break
    return k


def loss_positions(ep, tok):
    """Token positions (indices into ep.ids) that get the KL loss, and those that
    get the supervised anchor."""
    kl, anchor = [], []
    if not ep.spans:
        return kl, anchor
    for s, e in ep.spans[:-1]:
        kl.extend(range(s, e))
    s, e = ep.spans[-1]
    if ep.success:
        kl.extend(range(s, e))
        if DIAGNOSIS_RE.search(ep.texts[-1]):
            anchor.extend(range(s, e))
    elif ep.diag_char is not None:
        kl.extend(range(s, s + tokens_before_char(tok, ep.ids[s:e], ep.diag_char)))
    else:
        kl.extend(range(s, e))
    return kl, anchor


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, seqs, pad_id, stop_ids, max_new, device, sample):
    """Left-padded batch generation. Returns the new ids per row, cut after the
    first stop token (inclusive)."""
    width = max(map(len, seqs))
    inp = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        inp[i, width - len(s):] = torch.tensor(s)
        att[i, width - len(s):] = 1
    kw = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0) if sample else dict(do_sample=False)
    out = model.generate(input_ids=inp.to(device), attention_mask=att.to(device), max_new_tokens=max_new,
                         eos_token_id=stop_ids, pad_token_id=pad_id, **kw)
    res = []
    for row in out[:, width:].tolist():
        cut = next((j + 1 for j, t in enumerate(row) if t in stop_ids), len(row))
        res.append(row[:cut])
    return res


def simulate(sim, tok_sim, requests, args):
    if not requests:
        return []
    stops = [tok_sim.eos_token_id, tok_sim.convert_tokens_to_ids("<end_of_turn>")]
    seqs = [tok_sim(tok_sim.apply_chat_template(m, tokenize=False, add_generation_prompt=True),
                    add_special_tokens=False)["input_ids"] for m in requests]
    outs = generate(sim, seqs, tok_sim.pad_token_id, stops, args.simulator_max_new_tokens,
                    args.simulator_device, sample=False)
    return [THINK_RE.sub("", tok_sim.decode(o, skip_special_tokens=True)).strip() for o in outs]


def rollout(cases, student, tok_s, fmt, sim, tok_sim, args):
    eps = [Episode(case=c, ids=fmt.start(c["doctor_prompt"])) for c in cases]
    for turn in range(args.max_turns + 1):
        active = [e for e in eps if not e.done]
        if not active:
            break
        if turn == args.max_turns:
            for e in active:
                e.ids += fmt.user_turn(FORCE_PROMPT)
                e.forced = True
        for e in active:
            if len(e.ids) + args.doctor_max_new_tokens > args.max_seq_len:
                e.done, e.truncated = True, True
        active = [e for e in active if not e.done]
        if not active:
            break

        student.eval()
        gens = generate(student, [e.ids for e in active], tok_s.pad_token_id, fmt.stop_ids,
                        args.doctor_max_new_tokens, args.student_device, sample=True)

        requests, targets = [], []
        for e, g in zip(active, gens):
            start = len(e.ids)
            e.ids += g
            e.spans.append((start, len(e.ids)))
            e.ids += fmt.after_doctor(g)
            content = g[:-1] if g and g[-1] in fmt.stop_ids else g
            raw = tok_s.decode(content, skip_special_tokens=False)
            e.texts.append(raw)
            msg = raw.strip()

            diag, char = find_diagnosis(raw)
            if diag is not None or e.forced:
                e.done, e.final_diagnosis, e.diag_char = True, diag, char
                e.history.append({"speaker": "doctor", "content": msg, "visible_to": ["doctor"]})
                continue
            tm = TEST_RE.search(msg)
            if tm:
                e.history.append({"speaker": "doctor", "content": msg, "visible_to": ["doctor", "measurement"]})
                requests.append(measurement_messages(tm.group(1).strip(), e.case["findings"]))
                targets.append((e, "measurement", tm.group(1).strip()))
            else:
                e.history.append({"speaker": "doctor", "content": msg, "visible_to": ["doctor", "patient"]})
                requests.append(patient_messages(e.history, e.case["patient_actor"]))
                targets.append((e, "patient", msg))

        for (e, kind, src), reply in zip(targets, simulate(sim, tok_sim, requests, args)):
            if kind == "patient":
                if not reply or looks_like_echo(reply, src) or LEAKED_TAG_RE.search(reply):
                    reply = "Okay, doctor."
            else:
                stripped = RESULTS_PREFIX_RE.sub("", reply.strip())
                if not reply or looks_like_echo(stripped, src) or looks_like_echo(reply, src):
                    reply = "RESULTS: Not documented for this case."
                reply = measurement_label_fix(reply, e.case["findings"])
            vis = ["doctor", kind]
            e.history.append({"speaker": kind, "content": reply, "visible_to": vis})
            e.ids += fmt.user_turn(f"[{kind}] {reply}")
    return eps


@torch.no_grad()
def grade(teacher, tok_t, eps, args):
    todo = [e for e in eps if e.final_diagnosis]
    if not todo:
        return
    yes = tok_t("YES", add_special_tokens=False)["input_ids"][0]
    no = tok_t("NO", add_special_tokens=False)["input_ids"][0]
    seqs = [tok_t(tok_t.apply_chat_template(
                [{"role": "user", "content": GRADE_PROMPT.format(gt=e.case["correct_diagnosis"], fd=e.final_diagnosis)}],
                tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"] for e in todo]
    for i in range(0, len(seqs), 8):
        chunk = seqs[i:i + 8]
        width = max(map(len, chunk))
        inp = torch.full((len(chunk), width), tok_t.pad_token_id, dtype=torch.long)
        att = torch.zeros_like(inp)
        for j, s in enumerate(chunk):
            inp[j, width - len(s):] = torch.tensor(s)
            att[j, width - len(s):] = 1
        logits = teacher(input_ids=inp.to(args.teacher_device), attention_mask=att.to(args.teacher_device),
                         logits_to_keep=1).logits[:, -1].float()
        p = torch.softmax(logits[:, [yes, no]], -1)[:, 0].tolist()
        for e, py in zip(todo[i:i + 8], p):
            e.p_yes, e.success = py, py > 0.5


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def reverse_kl(s_logits, t_logits):
    """Exact KL(student || teacher) per position, summed over the full vocabulary."""
    ls = F.log_softmax(s_logits.float(), -1)
    lt = F.log_softmax(t_logits.float(), -1)
    return (ls.exp() * (ls - lt)).sum(-1)


def train_on(eps, student, teacher, tok_s, args):
    """Backpropagates the step's loss episode by episode. The loss is
    mean reverse KL over all KL tokens + anchor_weight * mean CE over all anchor tokens."""
    plans = [(e, *loss_positions(e, tok_s)) for e in eps]
    n_kl = sum(len(k) for _, k, _ in plans)
    n_an = sum(len(a) for _, _, a in plans)
    kl_sum = ce_sum = 0.0
    student.train()
    for e, kl, an in plans:
        pos = sorted(set(kl) | set(an))
        if not pos:
            continue
        ids = torch.tensor([e.ids])
        keep = torch.tensor(pos) - 1          # the logits at i-1 predict token i
        with torch.no_grad():
            t_logits = teacher(input_ids=ids.to(args.teacher_device),
                               logits_to_keep=keep.to(args.teacher_device)).logits[0].to(args.student_device)
        s_logits = student(input_ids=ids.to(args.student_device),
                           logits_to_keep=keep.to(args.student_device)).logits[0]
        kl_set, an_set = set(kl), set(an)
        kl_mask = torch.tensor([p in kl_set for p in pos], device=args.student_device)
        an_mask = torch.tensor([p in an_set for p in pos], device=args.student_device)
        loss = s_logits.new_zeros((), dtype=torch.float32)
        if kl_mask.any():
            kl_tok = reverse_kl(s_logits[kl_mask], t_logits[kl_mask])
            loss = loss + kl_tok.sum() / n_kl
            kl_sum += kl_tok.sum().item()
        if an_mask.any():
            targets = torch.tensor(pos, device=args.student_device)[an_mask]
            targets = torch.tensor(e.ids, device=args.student_device)[targets]
            ce = F.cross_entropy(s_logits[an_mask].float(), targets, reduction="sum")
            loss = loss + args.anchor_weight * ce / n_an
            ce_sum += ce.item()
        loss.backward()
        del s_logits, t_logits, loss
    return {"kl_tokens": n_kl, "anchor_tokens": n_an,
            "mean_kl": kl_sum / max(n_kl, 1), "anchor_ce": ce_sum / max(n_an, 1)}


# ---------------------------------------------------------------------------
# Setup and main loop
# ---------------------------------------------------------------------------

def build_models(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    def load(name, device):
        m = AutoModelForImageTextToText.from_pretrained(name, dtype=torch.bfloat16, device_map=device)
        m.requires_grad_(False)
        return m

    tok_s = AutoTokenizer.from_pretrained(args.student)
    tok_t = AutoTokenizer.from_pretrained(args.teacher)
    tok_sim = AutoTokenizer.from_pretrained(args.simulator)
    student = load(args.student, args.student_device)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student = get_peft_model(student, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_regex, task_type="CAUSAL_LM"))
    teacher = load(args.teacher, args.teacher_device).eval()
    sim = load(args.simulator, args.simulator_device).eval()
    return student, tok_s, teacher, tok_t, sim, tok_sim


def check_vocab(tok_s, tok_t):
    if tok_s.get_vocab() != tok_t.get_vocab():
        raise SystemExit("student and teacher tokenizers differ; full-vocabulary KL needs the same vocabulary")


def load_cases(args):
    path = args.cases
    if args.cases_repo:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(args.cases_repo, "opd_cases.jsonl", repo_type="dataset")
    return [json.loads(l) for l in open(path)]


def save(student, args, step, final=False):
    d = os.path.join(args.output_dir, "final" if final else f"step_{step:05d}")
    student.save_pretrained(d)
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, private=True, exist_ok=True)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.push_to_hub,
                          commit_message=f"step {step}")


def main(args, models=None):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    student, tok_s, teacher, tok_t, sim, tok_sim = models or build_models(args)
    check_vocab(tok_s, tok_t)
    fmt = ChatFormat(tok_s)

    cases = load_cases(args)
    steps = math.ceil(len(cases) * args.epochs / args.rollouts_per_step)
    order = []
    while len(order) < steps * args.rollouts_per_step:
        order += random.sample(cases, len(cases))

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(args.warmup_steps, 1)))
    print(f"{len(cases)} cases, {steps} steps x {args.rollouts_per_step} interviews; "
          f"{sum(p.numel() for p in params) / 1e6:.1f}M trainable params", flush=True)

    with open(os.path.join(args.output_dir, "metrics.jsonl"), "a") as mf, \
         open(os.path.join(args.output_dir, "rollouts.jsonl"), "a") as rf:
        for step in range(1, steps + 1):
            t0 = time.time()
            batch = order[(step - 1) * args.rollouts_per_step: step * args.rollouts_per_step]
            eps = rollout(batch, student, tok_s, fmt, sim, tok_sim, args)
            t_roll = time.time() - t0
            grade(teacher, tok_t, eps, args)
            stats = train_on(eps, student, teacher, tok_s, args)
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            m = {"step": step, "success_rate": sum(e.success for e in eps) / len(eps),
                 "diagnosis_given": sum(e.final_diagnosis is not None for e in eps) / len(eps),
                 "forced": sum(e.forced for e in eps), "truncated": sum(e.truncated for e in eps),
                 "mean_turns": sum(len(e.spans) for e in eps) / len(eps),
                 **stats, "grad_norm": gnorm, "lr": sched.get_last_lr()[0],
                 "rollout_s": round(t_roll, 1), "step_s": round(time.time() - t0, 1)}
            mf.write(json.dumps(m) + "\n")
            mf.flush()
            print(json.dumps(m), flush=True)
            for e in eps:
                rf.write(json.dumps({"step": step, "source": e.case["source"], "index": e.case["index"],
                                     "correct_diagnosis": e.case["correct_diagnosis"],
                                     "final_diagnosis": e.final_diagnosis, "p_yes": e.p_yes,
                                     "success": e.success, "forced": e.forced, "truncated": e.truncated,
                                     "history": e.history}, ensure_ascii=False) + "\n")
            rf.flush()
            if step % args.save_every == 0:
                save(student, args, step)
    save(student, args, steps, final=True)


if __name__ == "__main__":
    main(parse_args())
