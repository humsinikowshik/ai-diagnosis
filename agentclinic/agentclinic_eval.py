"""
Run an open model through the AgentClinic benchmark (Schmidgall et al., 2024)
using only Hugging Face's hosted Inference API -- no Anthropic API key
needed. AgentClinic isn't single-turn Q&A: each "case" is a simulated
patient encounter where the doctor model has to ask questions, order tests,
and eventually commit to a diagnosis.

This script plays two Hugging Face-hosted roles per case, every turn:
  - Doctor              (DOCTOR_MODEL)     <- the model being evaluated
  - Patient             (SIMULATOR_MODEL)  <- role-plays the case's symptoms/history
  - Measurement reader  (SIMULATOR_MODEL)  <- reads out exam/lab/imaging findings,
                                               strictly from the case data, whenever
                                               the doctor orders a test

DOCTOR_MODEL and SIMULATOR_MODEL are currently both Qwen/Qwen3-8B, i.e. the
same model plays doctor, patient, and measurement reader -- each role is
still a separate, stateless API call that only ever sees its own filtered
view of the conversation (see visible_messages()), so there's no literal
"the doctor can see its own answer key" leakage, but it does mean this run
isn't testing the doctor model against an independent simulator the way
the earlier google/gemma-3-4b-it run was (see agentclinic_gemma3_results.csv
for that comparison point -- same 10 scenarios, same simulator/turn budget,
different doctor model).

This script does NOT grade the results -- there's no LLM-as-judge call
here, so no Anthropic API key is needed to run it. Grading happens as a
separate step: once you have a transcripts file, ask Claude Code (this
assistant) to grade it against the rubric in agentclinic_judge_prompt.py.
Claude Code reads the rubric and the transcripts directly and writes the
graded CSV itself -- Claude Sonnet 5 *is* the judge here, just invoked
through this chat session instead of a paid API call.

Only the AgentClinic-MedQA scenarios are handled here (text-only). The
NEJM scenarios are multimodal (they include images) and would need a
vision-capable doctor call; out of scope for this script.

Setup:
  pip install -r requirements.txt
  export HF_TOKEN=hf_...                 # https://huggingface.co/settings/tokens
  # Gated models (Gemma, MedGemma) need you to click "Agree and access
  # repository" on the model's HF page, logged in as the account that owns
  # HF_TOKEN, before this script will work for them. Qwen3 and Baichuan-M2
  # are ungated.

Run (DOCTOR_MODEL defaults to Qwen/Qwen3-8B; output filenames are derived
from whatever doctor model you pass, so different doctor runs never
overwrite each other's results):
  python3 agentclinic_eval.py
  python3 agentclinic_eval.py --doctor-model google/gemma-3-27b-it
  python3 agentclinic_eval.py --doctor-model baichuan-inc/Baichuan-M2-32B --doctor-max-tokens 1000

For a full run over the whole MedQA dataset (107 scenarios) instead of the
default 10:
  python3 agentclinic_eval.py --doctor-model google/gemma-3-27b-it --num-scenarios all

If a run gets interrupted (billing pause, crash, network outage), just run
the exact same command again -- it resumes from whatever's already saved
in that doctor model's output files instead of re-running (and re-paying
for) completed scenarios. Pass --fresh to force a clean restart instead.
A scenario that fails on what looks like a transient provider error (busy,
timeout, 502/503/504) is automatically retried a few times before giving up
and moving on.

Then, in Claude Code, ask something like:
  "Grade agentclinic_qwen3-8b_transcripts.jsonl using agentclinic_judge_prompt.py"
"""

import argparse
import csv
import json
import os
import re
import sys
import time

import requests
from huggingface_hub import InferenceClient

# When stdout is redirected to a file (e.g. a background run) Python fully
# buffers it instead of flushing per line, so progress can sit invisible in
# a buffer for a long time. Force line buffering so `print()` below shows
# up immediately either way.
sys.stdout.reconfigure(line_buffering=True)

# --- Settings you can tweak (or override with --doctor-model / --doctor-max-tokens) ---
# parse_known_args() rather than parse_args() so this still works when the
# module is imported for interactive testing (e.g. `python3 -c "import
# agentclinic_eval"`), where sys.argv isn't this script's own CLI args.
_parser = argparse.ArgumentParser()
_parser.add_argument("--doctor-model", default="Qwen/Qwen3-8B", help="HF model ID for the doctor agent (the model under evaluation)")
_parser.add_argument("--doctor-max-tokens", type=int, default=800, help="max_tokens for the doctor's turn -- raise this for models with 'thinking' modes (Qwen3, Baichuan-M2) if you see truncated/empty replies")
_parser.add_argument("--provider", default=None, help="HF Inference Provider to force for the doctor model (e.g. 'featherless-ai'). Needed when a model is only served by one provider and HF's automatic routing doesn't find it -- seen with baichuan-inc/Baichuan-M2-32B and google/medgemma-27b-text-it, both Featherless-AI-only, even with that provider enabled in account settings. Check the model's HF page for its provider name if you hit 'not supported by any provider you have enabled'.")
_parser.add_argument("--num-scenarios", default="10", help="How many scenarios to run, or 'all' for the whole dataset (107 for MedQA). Default 10.")
_parser.add_argument("--fresh", action="store_true", help="Ignore any existing results for this doctor model and start over from scenario 0, instead of resuming past whatever's already saved.")
_parser.add_argument("--run-tag", default=None, help="Name used in the output filenames instead of one derived from --doctor-model. Use it when --doctor-model is a dedicated Inference Endpoint URL (otherwise the files get named after the long endpoint hostname), e.g. --run-tag gemma-4-e4b-it.")
_parser.add_argument("--simulator-model", default="Qwen/Qwen3-8B", help="Model for the patient + measurement agents. Keep this Qwen3-8B for comparability with earlier runs; pass a dedicated Inference Endpoint URL serving Qwen/Qwen3-8B when the Inference Providers route is down (Featherless, its only provider, was failing in Nov 2026).")
_args, _ = _parser.parse_known_args()

DOCTOR_MODEL = _args.doctor_model           # the model under evaluation
DOCTOR_MAX_TOKENS = _args.doctor_max_tokens  # generous by default: reasoning models need room for
                                             # both reasoning and the actual reply, or the reply
                                             # comes back empty (see message_text() below)
DOCTOR_PROVIDER = _args.provider            # None = let huggingface_hub auto-detect (works for most)
SIMULATOR_MODEL = _args.simulator_model     # plays patient + measurement reader (fixed across doctor-model comparisons: always Qwen3-8B)
SIMULATOR_MAX_TOKENS = 400                 # Qwen3 "thinks" before answering, so give it headroom
DATASET_URL = "https://raw.githubusercontent.com/SamuelSchmidgall/AgentClinic/main/agentclinic_medqa.jsonl"
DATASET_FILE = "agentclinic_medqa.jsonl"   # cached locally after first download
NUM_SCENARIOS = None if _args.num_scenarios.strip().lower() == "all" else int(_args.num_scenarios)
TOTAL_TURNS = 20                           # doctor's turn budget per case (AgentClinic's default)
FRESH_START = _args.fresh                  # False (default) = resume past whatever's already in the output files
MAX_SCENARIO_RETRIES = 3                   # extra attempts for a scenario that fails on a transient provider error
MAX_FORCE_DIAGNOSIS_RETRIES = 3            # attempts to get a properly-formatted DIAGNOSIS READY out of the
                                             # end-of-turns forced call, before giving up and recording whatever
                                             # raw (possibly non-diagnosis) text came back
RETRY_DELAY_SECONDS = 20                   # pause between retries -- immediate back-to-back retries all fail
                                             # the same way during a real outage (seen for real: a sustained
                                             # network drop burned all 3 retries instantly, losing 100+
                                             # scenarios that a delay would likely have ridden out)
TRANSIENT_ERROR_MARKERS = (
    "busy", "timeout", "timed out", "temporarily", "502", "503", "504", "overloaded",
    # local network blips (Wi-Fi drop, DNS hiccup) -- distinct from a provider-side
    # error, but just as worth retrying rather than permanently skipping a scenario.
    # Seen for real: a multi-scenario run lost 267 scenarios to a DNS outage because
    # none of these markers were recognized as transient at the time.
    "nodename nor servname", "name or service not known", "network is unreachable",
    "connection reset", "connection aborted", "failed to establish a new connection",
)


def model_tag(model_id):
    """Turn an HF model id into a filesystem-safe tag, e.g.
    "google/gemma-3-27b-it" -> "gemma-3-27b-it", so each doctor model's
    output files are automatically distinct and nothing gets overwritten.
    """
    return re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")


RUN_TAG = model_tag(_args.run_tag) if _args.run_tag else model_tag(DOCTOR_MODEL)
OUTPUT_CSV = f"agentclinic_{RUN_TAG}_results.csv"
OUTPUT_JSONL = f"agentclinic_{RUN_TAG}_transcripts.jsonl"  # full dialogue, ungraded -- see docstring above

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise SystemExit("Set the HF_TOKEN environment variable first (see comment above).")

REQUEST_TIMEOUT = 60  # seconds -- without this, a stalled provider call hangs forever
doctor_client = InferenceClient(model=DOCTOR_MODEL, token=hf_token, timeout=REQUEST_TIMEOUT, provider=DOCTOR_PROVIDER)
simulator_client = InferenceClient(model=SIMULATOR_MODEL, token=hf_token, timeout=REQUEST_TIMEOUT)

TEST_RE = re.compile(r"REQUEST TEST:\s*(.+)", re.IGNORECASE)
# Keyword-colon phrasings, all functionally equivalent to "DIAGNOSIS READY:" --
# added after seeing "FINAL DIAGNOSIS:", "MY DIAGNOSIS IS:", and "THE DIAGNOSIS
# IS:" repeatedly in real transcripts (the doctor was only ever told to use
# "DIAGNOSIS READY:", but reliably paraphrases it). Low false-positive risk:
# each of these is an explicit, deliberate declarative keyword, not something
# said in passing.
DIAGNOSIS_RE = re.compile(
    r"(?:(?:DIAGNOSIS READY|FINAL DIAGNOSIS)\s*:\s*|(?:MY DIAGNOSIS IS|THE DIAGNOSIS IS)\s*:?\s*)(.+)",
    re.IGNORECASE,
)
# Best-effort fallback for a confidently-stated free-text conclusion that never
# uses any DIAGNOSIS_RE keyword at all. Confirmed for real (gemma-3-4b-sft,
# case 0): a doctor turn flowed straight from a confirmatory lab result into
# "I believe our patient most likely has myasthenia gravis" -- the correct,
# well-reasoned answer, followed by an appropriate treatment plan and goodbye
# -- yet matched no DIAGNOSIS_RE keyword, so the scenario kept running past its
# natural end and eventually hallucinated a brand-new consultation once the
# turn budget forced another reply out of an already-finished conversation.
# Deliberately narrow and gated (see extract_free_text_diagnosis below): a
# hedge like "I am suspicious for X" earlier in that same transcript must NOT
# trigger this, since it's a mid-workup hypothesis, not the final answer --
# only a confident assertion verb, with no hedge word and no question mark or
# test order in the same message, counts.
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
THINK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def extract_free_text_diagnosis(text):
    """Best-effort fallback when none of DIAGNOSIS_RE's explicit keywords
    appear at all -- see FREE_TEXT_DIAGNOSIS_RE above for why and how this is
    gated. Returns the matched diagnosis text, or None if nothing sufficiently
    confident was found.
    """
    if "?" in text or TEST_RE.search(text) or HEDGE_RE.search(text):
        return None
    match = FREE_TEXT_DIAGNOSIS_RE.search(text)
    return match.group(1).strip() if match else None


def strip_thinking(text):
    """Qwen3 sometimes puts its reasoning in <think>...</think> before the
    real answer. Strip it so it never leaks into patient dialogue or a
    "test result" that the doctor sees.
    """
    return THINK_RE.sub("", text).strip()


LEADING_LABEL_RE = re.compile(r"^\[\w+\]\s*")
QUOTE_CHARS = "\"'“”‘’ "


def normalize_for_comparison(text):
    text = LEADING_LABEL_RE.sub("", text.strip())
    return re.sub(r"\s+", " ", text.strip(QUOTE_CHARS)).lower()


def looks_like_echo(reply, source_text):
    """True if `reply` is just `source_text` parroted back (allowing for
    a leaked "[speaker] " label or different quote marks around it).

    Observed twice in a 10-scenario run: Qwen3, playing the patient, would
    occasionally respond to the doctor's line with that exact line instead
    of an in-character answer -- once even echoing back the internal
    "[doctor] " role label used in visible_messages(), which should never
    appear in dialogue at all. Both times it followed a doctor statement
    ("Let's check your vital signs") rather than a direct question.
    """
    return normalize_for_comparison(reply) == normalize_for_comparison(source_text)


def message_text(response):
    """Pull the actual reply out of a chat_completion response.

    "Thinking" models (Qwen3, Baichuan-M2) can spend their whole
    `max_tokens` budget on internal reasoning and never get to write
    `content` at all, which comes back as None (`finish_reason ==
    "length"`). That crashes a naive `.content.strip()`. Appending
    "/no_think" to the prompt (used for the patient/measurement roles
    below) makes this rare in practice, but this still falls back to the
    reasoning text rather than crashing a whole scenario over one flaky
    call. Different models name the reasoning field differently --
    Qwen3 uses `reasoning_content`, Baichuan-M2 uses `reasoning` -- so
    check both rather than assuming one.
    """
    content = response.choices[0].message.content
    if content is None:
        message = response.choices[0].message
        content = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) or ""
    return content


# --- Dataset loading -----------------------------------------------------

def download_dataset():
    if not os.path.exists(DATASET_FILE):
        print(f"Downloading AgentClinic-MedQA scenarios from {DATASET_URL} ...")
        # requests (via certifi) rather than urllib -- urllib's stdlib ssl
        # context has no CA bundle on some python.org macOS builds.
        response = requests.get(DATASET_URL, timeout=30)
        response.raise_for_status()
        with open(DATASET_FILE, "wb") as f:
            f.write(response.content)


def flatten(value, prefix=""):
    """AgentClinic's exam/test findings are nested dicts of arbitrary depth,
    e.g. Test_Results -> Blood_Tests -> Acetylcholine_Receptor_Antibodies.
    Flatten them into {"Blood_Tests > Acetylcholine...": "Present ..."} so
    they're easy to read and to hand to an LLM as plain text.
    """
    flat = {}
    if isinstance(value, dict):
        for key, sub_value in value.items():
            path = f"{prefix} > {key}" if prefix else key
            flat.update(flatten(sub_value, path))
    elif isinstance(value, list):
        flat[prefix] = "; ".join(str(item) for item in value)
    else:
        flat[prefix] = str(value)
    return flat


def load_scenarios():
    scenarios = []
    with open(DATASET_FILE) as f:
        for line in f:
            record = json.loads(line)["OSCE_Examination"]
            patient_actor = record["Patient_Actor"]
            findings = {}
            findings.update(flatten(record.get("Physical_Examination_Findings", {})))
            findings.update(flatten(record.get("Test_Results", {})))
            scenarios.append({
                "demographics": patient_actor.get("Demographics", "Unknown"),
                "patient_actor": patient_actor,
                "findings": findings,
                "correct_diagnosis": record["Correct_Diagnosis"],
            })
    return scenarios


# --- The three simulation agents -----------------------------------------
#
# Note: none of these calls use a "system" role message. Chat templates for
# open models served this way have historically had spotty support for a
# system role (medqa_gemma_eval.py in this same project only ever sends
# "user" messages for that reason), so instructions for every agent are
# folded into the leading "user" message instead.

def visible_messages(history, speaker):
    """Build a plain user/assistant message list from one agent's point of
    view: entries not tagged as visible to `speaker` are dropped, that
    agent's own past turns become "assistant" messages, everyone else's
    becomes "user" (labelled with [speaker] so it's clear who said what).
    """
    messages = []
    for entry in history:
        if speaker not in entry["visible_to"]:
            continue
        role = "assistant" if entry["speaker"] == speaker else "user"
        content = entry["content"] if role == "assistant" else f"[{entry['speaker']}] {entry['content']}"
        messages.append({"role": role, "content": content})
    return messages


LEAKED_TAG_RE = re.compile(r"\[(?:doctor|patient|measurement)\]", re.IGNORECASE)
META_INSTRUCTION_RE = re.compile(r"^(?:ask|tell|inform|request)\b.{0,80}\bpatient\b", re.IGNORECASE)


def looks_broken_doctor_output(text):
    """Catch two failure modes seen when a reasoning model (Qwen3) plays the
    doctor: (1) writing a third-person stage direction ("Ask the patient
    if...") instead of actually asking, and (2) far worse, going on to
    fabricate the *patient's own answer* in a later turn -- generating a
    fake "[patient] ..." line itself rather than waiting for the real
    patient agent. Once seen, this observed history: it kept happening and
    the model built false "history" on top of its own fabrication, which
    directly produced a wrong diagnosis in testing.

    A message that's already a valid REQUEST TEST / DIAGNOSIS READY action
    is exempted -- don't second-guess a well-formed protocol action just
    because incidental wording elsewhere in it looks similar.
    """
    if TEST_RE.search(text) or DIAGNOSIS_RE.search(text):
        return False
    return bool(LEAKED_TAG_RE.search(text)) or bool(META_INSTRUCTION_RE.match(text.strip()))


# The categories of clinically purposeful question AgentClinic's own scenarios
# are built around -- every question to the patient should be in service of
# one of these, not generic paperwork/rapport filler (occupation, insurance,
# scheduling, "have a great day"). Listing them concretely (rather than just
# saying "ask relevant questions") gives the model something specific to
# pattern-match onto, which matters for a checkpoint that otherwise drifts
# into demographic small talk and non-sequiturs.
INQUIRY_CATEGORIES = (
    "- What is the cause of symptom X?\n"
    "- What is the dose of drug X?\n"
    "- How should I manage disease or finding X?\n"
    "- What is the drug of choice for condition X?\n"
    "- What test is indicated in situation X?\n"
    "- How accurate/good is test X in situation Y?\n"
    "- What are the side effects of drug X?\n"
    "- What is the prognosis for condition X?"
)


def build_doctor_instructions(demographics, extra_instruction=None, first_turn=False):
    instructions = (
        "You are a doctor named Dr. Agent conducting a patient consultation. "
        "Respond only with what you would actually say or do next -- do not narrate your reasoning, "
        "and do not describe what to ask in the third person (never write things like 'Ask the "
        "patient if...' -- ask them directly, in first person, as dialogue). Never write a line for "
        "anyone but yourself; wait for the patient's or measurement reader's actual reply instead of "
        "guessing what they would say.\n\n"
        f"A new patient has arrived. Demographics: {demographics}. Treat these demographics (age, "
        "sex, and anything else given) as clinically relevant evidence -- weigh them, along with "
        "everything else you learn, when deciding what to ask or test next and when reaching your "
        "diagnosis.\n\n"
        "Your goal is to reach the correct diagnosis. Every question or test you choose must "
        "directly serve that goal -- never ask about anything else (no occupation, insurance, "
        "scheduling, small talk, or other non-clinical filler). Each patient question should pursue "
        "one of these lines of inquiry:\n" + INQUIRY_CATEGORIES + "\n\n"
        f"You have at most {TOTAL_TURNS} turns total before you must give a final diagnosis -- pace "
        "your questions and tests so you reach a confident diagnosis within that budget, rather than "
        "using every turn regardless of need. Each turn, do exactly ONE of the following:\n"
        "1. Ask the patient one question, as plain dialogue, pursuing one of the lines of inquiry above.\n"
        "2. Order a physical exam or test with exactly this format:\n   REQUEST TEST: <name>\n"
        "3. Give your final diagnosis with exactly this format:\n   DIAGNOSIS READY: <diagnosis>\n\n"
        "Use DIAGNOSIS READY as soon as you are confident, and always before your turns run out."
    )
    if extra_instruction:
        instructions += "\n\n" + extra_instruction
    if first_turn:
        instructions += "\n\nBegin by asking the patient your first question."
    return instructions


def doctor_turn(history, demographics, extra_instruction=None):
    instructions = build_doctor_instructions(demographics, extra_instruction, first_turn=not history)
    messages = [{"role": "user", "content": instructions}] + visible_messages(history, "doctor")

    reply = ""
    for attempt in range(3):
        response = doctor_client.chat_completion(messages=messages, max_tokens=DOCTOR_MAX_TOKENS)
        reply = strip_thinking(message_text(response).strip())
        if reply and not looks_broken_doctor_output(reply):
            return reply
    return reply  # best effort after exhausting retries -- let run_scenario's classification handle it


def force_diagnosis_turn(history, demographics, blunt=False):
    """Ask the doctor for a final diagnosis only, with none of the normal
    turn options (ask a question / order a test) present in the prompt at
    all -- unlike layering an extra_instruction on top of the full menu in
    build_doctor_instructions(), which a model with weak instruction-following
    tends to ignore in favor of the far more heavily-reinforced menu items.
    Confirmed for real: with the menu-based approach, 0/24 gemma-3-4b-sft
    scenarios that ran out of turns were ever rescued across 3 retries each --
    every single one kept ordering a test anyway, because the prompt still
    listed "order a test" as a valid numbered option right above the override.
    """
    instructions = (
        "You are a doctor named Dr. Agent. You have just finished a consultation with a "
        f"patient (demographics: {demographics}) and have used up all your allotted turns. "
        "You must now state your final diagnosis. Do not ask any question. Do not order "
        "any test. Respond with only this, filled in, and nothing else:\n"
        "DIAGNOSIS READY: <diagnosis>"
    )
    if blunt:
        instructions += (
            "\n\nYour previous reply did not follow this format -- it asked a question or "
            "ordered a test instead. That is not allowed anymore. Reply with only the line "
            "above, with your diagnosis filled in, and absolutely nothing else."
        )
    messages = [{"role": "user", "content": instructions}] + visible_messages(history, "doctor")
    response = doctor_client.chat_completion(messages=messages, max_tokens=DOCTOR_MAX_TOKENS)
    return strip_thinking(message_text(response).strip())


def patient_reply(history, patient_actor):
    facts = "\n".join(f"{k}: {v}" for k, v in patient_actor.items() if k != "Demographics")
    instructions = (
        "You are a patient at a doctor's office, played for a medical training simulation. "
        "Answer in 1-3 sentences of first-person dialogue, staying strictly consistent with "
        "the facts below. Don't reveal anything the doctor hasn't asked about, and never "
        "mention or guess a diagnosis. Respond with the dialogue only -- no reasoning, no "
        "narration. /no_think\n\nYour facts:\n" + facts
    )
    # patient_reply is only ever called right after the doctor's question has
    # been appended to history, so this is never empty and its first entry
    # is always role "user" (the doctor speaking) -- merge the instructions
    # into that message instead of adding a separate leading one, otherwise
    # we'd get two "user" messages in a row, which breaks strict
    # user/assistant alternation in most chat templates.
    messages = visible_messages(history, "patient")
    messages[0] = {"role": "user", "content": instructions + "\n\n" + messages[0]["content"]}
    doctor_said = history[-1]["content"]  # what the patient must actually respond to, not repeat

    for attempt in range(3):
        response = simulator_client.chat_completion(messages=messages, max_tokens=SIMULATOR_MAX_TOKENS)
        reply = strip_thinking(message_text(response).strip())
        # reject an echo, or the patient generating a fake "[doctor] ..." line of its
        # own -- seen when a garbled doctor turn (e.g. leaked reasoning) confuses the
        # patient about whose turn it is.
        if reply and not looks_like_echo(reply, doctor_said) and not LEAKED_TAG_RE.search(reply):
            return reply
    return "Okay, doctor."  # generic in-character fallback if it keeps echoing


RESULTS_PREFIX_RE = re.compile(r"^results:\s*", re.IGNORECASE)


def looks_like_measurement_echo(reply, requested_test):
    """True if the measurement reader just parroted the requested test name
    back as the "result" instead of reporting a finding or saying "Not
    documented for this case." patient_reply() has had echo-retry protection
    (looks_like_echo(), below) since early in this project; measurement_read()
    never did, and it showed: a doctor call went on to fabricate a positive
    lab result ("the repeat fungal culture grew Mucor species") out of an
    echoed non-answer and used it to justify a diagnosis -- found while
    reading a "match"-graded SFT record, meaning the diagnosis-correctness
    check alone doesn't catch this at all.
    """
    stripped = RESULTS_PREFIX_RE.sub("", reply.strip())
    return looks_like_echo(stripped, requested_test) or looks_like_echo(reply, requested_test)


def measurement_read(requested_test, findings):
    findings_text = "\n".join(f"{k}: {v}" for k, v in findings.items()) or "(none recorded)"
    instructions = (
        "You are a clinical measurement reader for a medical training simulation. You are "
        "given a fixed set of recorded findings below. Respond with ONLY the matching "
        "finding(s), formatted as `RESULTS: <finding>`. If nothing below matches the request, "
        "respond exactly `RESULTS: Not documented for this case.` Never invent a finding that "
        f"isn't listed. /no_think\n\nRecorded findings:\n{findings_text}\n\nRequested: {requested_test}"
    )
    for attempt in range(3):
        response = simulator_client.chat_completion(
            messages=[{"role": "user", "content": instructions}], max_tokens=SIMULATOR_MAX_TOKENS,
        )
        reply = strip_thinking(message_text(response).strip())
        if reply and not looks_like_measurement_echo(reply, requested_test):
            return reply
    return "RESULTS: Not documented for this case."  # safe, honest fallback if it keeps echoing


# --- Running one scenario --------------------------------------------------

def run_scenario(scenario):
    history = []
    final_diagnosis = None
    turns_used = 0

    for _ in range(TOTAL_TURNS):
        doctor_msg = doctor_turn(history, scenario["demographics"])
        turns_used += 1

        diagnosis_match = DIAGNOSIS_RE.search(doctor_msg)
        test_match = TEST_RE.search(doctor_msg)
        free_text_diagnosis = None if (diagnosis_match or test_match) else extract_free_text_diagnosis(doctor_msg)

        if diagnosis_match or free_text_diagnosis:
            final_diagnosis = diagnosis_match.group(1).strip() if diagnosis_match else free_text_diagnosis
            history.append({"speaker": "doctor", "content": doctor_msg, "visible_to": ["doctor"]})
            break

        if test_match:
            history.append({"speaker": "doctor", "content": doctor_msg, "visible_to": ["doctor", "measurement"]})
            result = measurement_read(test_match.group(1).strip(), scenario["findings"])
            history.append({"speaker": "measurement", "content": result, "visible_to": ["doctor", "measurement"]})
        else:
            history.append({"speaker": "doctor", "content": doctor_msg, "visible_to": ["doctor", "patient"]})
            reply = patient_reply(history, scenario["patient_actor"])
            history.append({"speaker": "patient", "content": reply, "visible_to": ["doctor", "patient"]})

    forced_diagnosis_noncompliant = False
    if final_diagnosis is None:
        # A single forced attempt isn't enough for a model that regularly ignores
        # format instructions (observed heavily in fine-tuned checkpoints): it
        # just asks another question instead of emitting DIAGNOSIS READY, and
        # that raw question used to get recorded as the "final_diagnosis"
        # verbatim. Retry with an increasingly blunt instruction, same as
        # doctor_turn()'s own retry loop for broken output, but checking
        # specifically for DIAGNOSIS_RE compliance rather than just "not broken".
        forced = ""
        for attempt in range(MAX_FORCE_DIAGNOSIS_RETRIES):
            forced = force_diagnosis_turn(history, scenario["demographics"], blunt=attempt > 0)
            if DIAGNOSIS_RE.search(forced) or extract_free_text_diagnosis(forced):
                break
        else:
            forced_diagnosis_noncompliant = True
        # Append this forced turn to history same as a voluntary diagnosis --
        # otherwise final_diagnosis (returned below) describes a call that
        # never actually happened in the recorded conversation. Confirmed for
        # real: 12/400 scenarios in one SFT-generation batch hit exactly this,
        # leaving `messages` ending on an unrelated ordinary turn while
        # final_diagnosis held disconnected text from this forced call --
        # silently wrong for any consumer that assumes the two agree (a
        # transcript reader, or downstream training-data code that assumes
        # the last turn is always the diagnosis).
        history.append({"speaker": "doctor", "content": forced, "visible_to": ["doctor"]})
        diagnosis_match = DIAGNOSIS_RE.search(forced)
        if diagnosis_match:
            final_diagnosis = diagnosis_match.group(1).strip()
        else:
            final_diagnosis = extract_free_text_diagnosis(forced) or forced.strip()

    return {
        "history": history,
        "final_diagnosis": final_diagnosis,
        "turns_used": turns_used,
        "forced_diagnosis_noncompliant": forced_diagnosis_noncompliant,
    }


# --- Main --------------------------------------------------------------
#
# No grading happens here -- see the module docstring. This just runs the
# simulation and saves what happened, ungraded.

def format_transcript(history):
    return "\n".join(f"{entry['speaker'].upper()}: {entry['content']}" for entry in history)


def is_transient_error(exc):
    """True if this looks like a temporary provider hiccup worth retrying
    (a busy/overloaded backend, a timeout) rather than a real problem that
    retrying won't fix (bad request, model not found, payment required)."""
    text = str(exc).lower()
    return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)


def load_existing_results():
    """Read whatever this doctor model has already saved, so a full run can
    resume after an interruption (billing pause, crash, transient outage)
    instead of re-running -- and re-paying for -- scenarios already done.
    Returns (records, csv_rows, done_indices), all empty if starting fresh.
    """
    if FRESH_START or not os.path.exists(OUTPUT_JSONL):
        return [], [], set()
    with open(OUTPUT_JSONL) as f:
        records = [json.loads(line) for line in f]
    csv_rows = [
        {"index": r["index"], "correct_diagnosis": r["correct_diagnosis"],
         "final_diagnosis": r["final_diagnosis"], "turns_used": r["turns_used"],
         "forced_diagnosis_noncompliant": r.get("forced_diagnosis_noncompliant", False)}
        for r in records
    ]
    done_indices = {r["index"] for r in records}
    if done_indices:
        print(f"Resuming {DOCTOR_MODEL}: {len(done_indices)} scenario(s) already done, skipping those.")
    return records, csv_rows, done_indices


def save_results(records, csv_rows):
    records.sort(key=lambda r: r["index"])
    with open(OUTPUT_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    csv_rows.sort(key=lambda r: r["index"])
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "correct_diagnosis", "final_diagnosis", "turns_used", "forced_diagnosis_noncompliant"])
        writer.writeheader()
        writer.writerows(csv_rows)


def main():
    download_dataset()
    scenarios = load_scenarios()
    if NUM_SCENARIOS is not None:
        scenarios = scenarios[:NUM_SCENARIOS]

    records, csv_rows, done_indices = load_existing_results()

    for i, scenario in enumerate(scenarios):
        if i in done_indices:
            continue
        print(f"\n=== Scenario {i + 1}/{len(scenarios)}: ground truth = {scenario['correct_diagnosis']!r} ===")

        run = None
        for attempt in range(1 + MAX_SCENARIO_RETRIES):
            try:
                run = run_scenario(scenario)
                break
            except Exception as e:
                if attempt < MAX_SCENARIO_RETRIES and is_transient_error(e):
                    print(f"  .. transient error ({e}), retrying in {RETRY_DELAY_SECONDS}s ({attempt + 1}/{MAX_SCENARIO_RETRIES})")
                    time.sleep(RETRY_DELAY_SECONDS)
                    continue
                print(f"  !! skipped due to error: {e}")
                break
        if run is None:
            continue

        print(f"Doctor's final diagnosis: {run['final_diagnosis']!r} (used {run['turns_used']} turns)")

        records.append({
            "index": i,
            "correct_diagnosis": scenario["correct_diagnosis"],
            "final_diagnosis": run["final_diagnosis"],
            "turns_used": run["turns_used"],
            "forced_diagnosis_noncompliant": run["forced_diagnosis_noncompliant"],
            "history": run["history"],
        })
        csv_rows.append({
            "index": i,
            "correct_diagnosis": scenario["correct_diagnosis"],
            "final_diagnosis": run["final_diagnosis"],
            "turns_used": run["turns_used"],
            "forced_diagnosis_noncompliant": run["forced_diagnosis_noncompliant"],
        })
        save_results(records, csv_rows)  # save after every scenario, so an interruption loses at most one

    print(f"\n{len(records)}/{len(scenarios)} scenarios completed.")
    print(f"Ungraded summary saved to {OUTPUT_CSV}")
    print(f"Full transcripts saved to {OUTPUT_JSONL}")
    if len(records) < len(scenarios):
        print(f"({len(scenarios) - len(records)} scenario(s) still missing -- rerun the same command to resume.)")
    print(
        "\nTo grade: ask Claude Code to grade "
        f"{OUTPUT_JSONL} using the rubric in agentclinic_judge_prompt.py"
    )


if __name__ == "__main__":
    main()
