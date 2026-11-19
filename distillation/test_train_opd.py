"""
Checks for train_opd.py. Runs on a laptop CPU (no big models): tokenizers only,
plus a tiny random Gemma 4 for the end-to-end smoke test.

Run: python3 test_train_opd.py   (needs peft; downloads tokenizers only)
"""

import copy
import json
import os
import tempfile

import torch
from huggingface_hub import get_token

os.environ.setdefault("HF_TOKEN", get_token() or "")
import agentclinic_eval as ac  # noqa: E402
import generate_sft_dialogues as gsd  # noqa: E402
import train_opd as T  # noqa: E402
from transformers import AutoConfig, AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
fmt = T.ChatFormat(tok)
cases = [json.loads(l) for l in open("opd_cases.jsonl")]
ok = lambda name: print("ok  ", name)


# 1. Copied protocol matches agentclinic_eval / generate_sft_dialogues ----------
for n in ["TEST_RE", "DIAGNOSIS_RE", "FREE_TEXT_DIAGNOSIS_RE", "HEDGE_RE", "LEAKED_TAG_RE",
          "RESULTS_PREFIX_RE", "LEADING_LABEL_RE", "THINK_RE"]:
    assert getattr(T, n).pattern == getattr(ac, n).pattern, n
assert T.LABEL_SPLIT_RE.pattern == gsd.LABEL_SPLIT_RE.pattern


class Capture:
    def __init__(self, reply):
        self.reply, self.messages = reply, None

    def chat_completion(self, messages, max_tokens):
        self.messages = messages
        return {"choices": [{"message": {"content": self.reply}}]}


case = cases[0]
hist = [{"speaker": "doctor", "content": "What brings you in?", "visible_to": ["doctor", "patient"]},
        {"speaker": "patient", "content": "I'm tired.", "visible_to": ["doctor", "patient"]},
        {"speaker": "doctor", "content": "REQUEST TEST: CBC", "visible_to": ["doctor", "measurement"]},
        {"speaker": "measurement", "content": "RESULTS: normal", "visible_to": ["doctor", "measurement"]},
        {"speaker": "doctor", "content": "How long?", "visible_to": ["doctor", "patient"]}]
cap = Capture("For months.")
ac.simulator_client = cap
ac.message_text = lambda r: r["choices"][0]["message"]["content"]
ac.patient_reply(hist, case["patient_actor"])
assert cap.messages == T.patient_messages(hist, case["patient_actor"])
ac.measurement_read("CBC", case["findings"])
assert cap.messages == T.measurement_messages("CBC", case["findings"])
assert T.visible_messages(hist, "doctor") == ac.visible_messages(hist, "doctor")
key = next(iter(case["findings"]))
for reply in ["RESULTS: " + key.split(" > ")[-1], "RESULTS: something else"]:
    gsd._original_measurement_read = lambda *_: reply
    assert gsd.measurement_read_with_label_fix("x", case["findings"]) == T.measurement_label_fix(reply, case["findings"])
for c in cases[:50]:
    assert c["doctor_prompt"] == ac.build_doctor_instructions(c["demographics"], first_turn=True)
for text in ["DIAGNOSIS READY: Sarcoidosis", "Final diagnosis: Lupus nephritis", "Any fever?",
             "REQUEST TEST: CBC", "She most likely has Graves disease.", "This could be X; she most likely has Y.",
             "I think the diagnosis is: gout"]:
    m = ac.DIAGNOSIS_RE.search(text)
    want = m.group(1).strip() if m else (None if ac.TEST_RE.search(text) else ac.extract_free_text_diagnosis(text))
    assert T.find_diagnosis(text)[0] == want, text
ok("prompts, regexes, label fix and diagnosis parsing match agentclinic_eval / generate_sft_dialogues")


# 2. Piecewise token ids == rendering the whole conversation ------------------------
doctor_turns = ["What brings you in today?", "REQUEST TEST: CBC", "DIAGNOSIS READY: Iron deficiency anemia"]
replies = ["[patient] I'm tired.", "[measurement] RESULTS: Hb 9 g/dL"]
ids = fmt.start(case["doctor_prompt"])
msgs = [{"role": "user", "content": case["doctor_prompt"]}]
for i, d in enumerate(doctor_turns):
    sampled = fmt.enc(d) + [fmt.end_turn_id]
    ids += sampled + fmt.after_doctor(sampled)
    msgs.append({"role": "assistant", "content": d})
    if i < len(replies):
        ids += fmt.user_turn(replies[i])
        msgs.append({"role": "user", "content": replies[i]})
assert ids == fmt.enc(fmt.render(msgs)), "piecewise ids differ from the chat template"
assert fmt.stop_ids == sorted({tok.convert_tokens_to_ids("<turn|>"), tok.eos_token_id})
ok("piecewise token ids equal the Gemma 4 chat template for a full interview")


# 3. Loss masks -------------------------------------------------------------------
def make_episode(turns, success):
    e = T.Episode(case=case, ids=fmt.start("P"))
    for d in turns:
        g = fmt.enc(d) + [fmt.end_turn_id]
        s = len(e.ids)
        e.ids += g
        e.spans.append((s, len(e.ids)))
        e.ids += fmt.after_doctor(g)
        e.texts.append(tok.decode(g[:-1], skip_special_tokens=False))
        e.ids += fmt.user_turn("[patient] ok")
    e.final_diagnosis, e.diag_char = T.find_diagnosis(e.texts[-1])
    e.success = success
    return e


turns = ["Any fever?", "REQUEST TEST: CBC", "DIAGNOSIS READY: Acute myeloid leukemia"]
dec = lambda e, pos: tok.decode([e.ids[p] for p in pos], skip_special_tokens=False)

good = make_episode(turns, True)
kl, an = T.loss_positions(good, tok)
assert kl == [p for s, e_ in good.spans for p in range(s, e_)]           # every sampled token
assert dec(good, an) == "DIAGNOSIS READY: Acute myeloid leukemia<turn|>"  # anchor = final turn only

bad = make_episode(turns, False)
kl, an = T.loss_positions(bad, tok)
s, _ = bad.spans[-1]
assert an == []
assert dec(bad, [p for p in kl if p >= s]) == "DIAGNOSIS READY:"          # cut right after the string
assert dec(bad, [p for p in kl if p < s]) == dec(bad, [p for s_, e_ in bad.spans[:-1] for p in range(s_, e_)])

lead = make_episode(["Any fever?", "Based on this, DIAGNOSIS READY: Gout"], False)
kl, _ = T.loss_positions(lead, tok)
assert dec(lead, [p for p in kl if p >= lead.spans[-1][0]]) == "Based on this, DIAGNOSIS READY:"

none = make_episode(["Any fever?", "Do you have pain?"], False)      # forced turn, no diagnosis line
none.final_diagnosis, none.diag_char = None, None
kl, an = T.loss_positions(none, tok)
assert kl == [p for s_, e_ in none.spans for p in range(s_, e_)] and an == []
ok("masks: success = KL everywhere + anchor on DIAGNOSIS READY turn; failure = KL cut after 'DIAGNOSIS READY:'")


# 4. Full-vocabulary reverse KL -----------------------------------------------------
torch.manual_seed(0)
a, b = torch.randn(5, 262144), torch.randn(5, 262144)
assert T.reverse_kl(a, a).abs().max() < 1e-4
ps, pt = torch.softmax(a.double(), -1), torch.softmax(b.double(), -1)
manual = (ps * (ps.log() - pt.log())).sum(-1)
assert torch.allclose(T.reverse_kl(a, b).double(), manual, rtol=1e-3)
assert (T.reverse_kl(a, b) >= 0).all()
ok("reverse KL over all 262,144 tokens matches the closed form, is 0 for equal logits")


# 5. End-to-end smoke test with tiny random Gemma 4 models ----------------------------
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Gemma4ForCausalLM  # noqa: E402

tc = AutoConfig.from_pretrained("google/gemma-4-E4B-it").text_config
tc = copy.deepcopy(tc)
small = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
             num_key_value_heads=1, head_dim=16, hidden_size_per_layer_input=8)
for k, v in small.items():
    setattr(tc, k, v)
for k in ["global_head_dim", "num_global_key_value_heads"]:
    if hasattr(tc, k) and getattr(tc, k) is not None:
        setattr(tc, k, 16 if k == "global_head_dim" else 1)
tc.layer_types = ["sliding_attention", "full_attention"]
if hasattr(tc, "num_kv_shared_layers"):
    tc.num_kv_shared_layers = 0
if isinstance(getattr(tc, "intermediate_size", None), list):
    tc.intermediate_size = [64, 64]


def tiny(seed=1):
    torch.manual_seed(seed)
    return Gemma4ForCausalLM(tc).eval().requires_grad_(False)


student = get_peft_model(tiny(), LoraConfig(r=4, lora_alpha=8, target_modules=r".*layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"))
before = {n: p.detach().clone() for n, p in student.named_parameters() if p.requires_grad}
tok_sim = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
with tempfile.TemporaryDirectory() as d:
    args = T.parse_args(["--cases", "opd_cases.jsonl", "--student-device", "cpu", "--teacher-device", "cpu",
                         "--simulator-device", "cpu", "--epochs", str(4 / len(cases)), "--rollouts-per-step", "2",
                         "--max-turns", "2", "--doctor-max-new-tokens", "6", "--simulator-max-new-tokens", "4",
                         "--warmup-steps", "1", "--lr", "1e-2", "--output-dir", d, "--save-every", "1"])
    T.main(args, models=(student, tok, tiny(2), tok, tiny(3), tok_sim))
    metrics = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    rolls = [json.loads(l) for l in open(os.path.join(d, "rollouts.jsonl"))]
    assert len(metrics) == 2 and len(rolls) == 4 and all(m["kl_tokens"] > 0 for m in metrics)
    assert all(r["forced"] for r in rolls) and metrics[0]["mean_kl"] > 0          # random model never diagnoses -> forced turn used
    assert os.path.exists(os.path.join(d, "final", "adapter_model.safetensors"))
changed = sum(not torch.equal(before[n], p) for n, p in student.named_parameters() if p.requires_grad)
assert changed > 0
ok(f"end-to-end: 2 steps of rollout -> grade -> KL backward -> AdamW ran; {changed}/{len(before)} LoRA tensors updated")

# train_on with the hand-built episodes from section 3 (success + failure), so the anchor path runs too
student.zero_grad()
stats = T.train_on([good, bad], student, tiny(2), tok, args)
n_kl = sum(len(T.loss_positions(e, tok)[0]) for e in (good, bad))
n_an = len(T.loss_positions(good, tok)[1])
assert stats["kl_tokens"] == n_kl and stats["anchor_tokens"] == n_an and stats["anchor_ce"] > 0 and stats["mean_kl"] > 0
assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters() if p.requires_grad)
ok(f"train_on: {n_kl} KL tokens + {n_an} anchor tokens, mean KL {stats['mean_kl']:.3f}, anchor CE {stats['anchor_ce']:.2f}, gradients reach LoRA")
