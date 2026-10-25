"""Part C: support-ticket triage with guardrails, redaction, retries, validation, budget and routing."""
import json
import os
import random
import re
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402
from mock_llm import TransientLLMError, est_tokens  # noqa: E402

OUT = os.path.join(ROOT, "outputs")
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATS = ["billing", "delivery", "technical", "fraud"]
PII = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
       ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
       ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]

ZERO_SHOT = ("You CLASSIFY Instamart customer support tickets. Allowed categories: billing, delivery, technical, fraud. "
             'Reply with JSON only: {"category": "...", "reason": "..."}')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: Someone used my card for an order I never placed. -> {"category": "fraud", "reason": "unauthorized transaction"}
Ticket: The app freezes when I open the cart. -> {"category": "technical", "reason": "app error"}
Ticket: I was charged twice for one order. -> {"category": "billing", "reason": "double charge"}
Ticket: The rider has not arrived after an hour. -> {"category": "delivery", "reason": "late delivery"}"""


def redact(text):
    counts = {}
    for name, pattern, placeholder in PII:
        text, counts[name] = pattern.subn(placeholder, text)
    return text, counts


def call_with_retry(ticket_id, messages, log):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as err:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            log.append({"ticket_id": ticket_id, "retry": n, "wait_s": round(wait, 4), "error": str(err)})
            print(f"  retry ticket={ticket_id} n={n} wait={wait:.4f}s ({err})")
            time.sleep(wait)


def validate(text):
    try:
        reply = json.loads(text)
        if isinstance(reply, dict) and reply.get("category") in CATS:
            return "classified_json", reply["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        found = {w for w in re.findall(r"[a-z]+", (text or "").lower()) if w in CATS}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""


def run(tickets, system_prompt, token_budget=None):
    mock_llm.reset()
    random.seed(42)
    rows, retry_log, prompts, total = [], [], [], 0
    pii_counts = {name: 0 for name, _, _ in PII}
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "category": "", "tokens_used": 0}
        if not t.text.strip():
            rows.append({**row, "status": "skipped_empty"}); continue
        if est_tokens(t.text) > MAX_INPUT_TOKENS:
            rows.append({**row, "status": "flagged_long"}); continue
        clean, counts = redact(t.text)
        for k, v in counts.items():
            pii_counts[k] += v
        if token_budget is not None and total >= token_budget:
            rows.append({**row, "status": "budget_exhausted"}); continue
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": clean}]
        prompts.append(clean)
        resp = call_with_retry(t.ticket_id, messages, retry_log)
        used = resp.usage.total_tokens
        total += used
        status, category = validate(resp.choices[0].message.content)
        rows.append({**row, "status": status, "category": category, "tokens_used": used})
    for text in prompts:
        for name, pattern, _ in PII:
            assert not pattern.search(text), f"{name} leaked into a prompt"
    return pd.DataFrame(rows), retry_log, total, pii_counts


def summarise(name, df, total):
    done = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((done["category"] == done["true_category"]).sum())
    print(f"\n== {name}: status counts {df['status'].value_counts().to_dict()}")
    print(f"classified {len(done)}, accuracy {correct}/{len(done)} = {correct / len(done):.1%}, "
          f"total tokens {total}, tokens per classified ticket {total / len(done):.1f}")
    return done


tickets = pd.read_csv(os.path.join(ROOT, "data", "raw", "tickets.csv"), keep_default_na=False)
long_tokens = sorted({est_tokens(x) for x in tickets["text"] if est_tokens(x) > MAX_INPUT_TOKENS})
print(f"tickets {len(tickets)}; empty {int((tickets['text'].str.strip() == '').sum())}; "
      f"long {int((tickets['text'].map(est_tokens) > MAX_INPUT_TOKENS).sum())} (est_tokens {long_tokens})")

all_counts = {name: 0 for name, _, _ in PII}
for text in tickets["text"]:
    redacted, counts = redact(text)
    for k, v in counts.items():
        all_counts[k] += v
    assert not any(p.search(redacted) for _, p, _ in PII)
print(f"redactions across all 164 tickets: {all_counts}, total {sum(all_counts.values())}; 0 matches left")
print(f"prompt est_tokens: zero-shot {est_tokens(ZERO_SHOT)}, few-shot {est_tokens(FEW_SHOT)}")

print("\n-- zero-shot run")
zs, zs_log, zs_total, _ = run(tickets, ZERO_SHOT)
print(f"transient errors recovered: {len(zs_log)}")
assert len(zs_log) == 19
zs_done = summarise("zero-shot", zs, zs_total)
print("zero-shot confusion (rows true, cols predicted)\n" +
      pd.crosstab(zs_done["true_category"], zs_done["category"]).to_string())

print("\n-- few-shot run")
fs, fs_log, fs_total, _ = run(tickets, FEW_SHOT)
print(f"transient errors recovered: {len(fs_log)}")
summarise("few-shot", fs, fs_total)

print("\n-- budget demo (zero-shot, TOKEN_BUDGET=6000)")
demo, _, demo_total, _ = run(tickets, ZERO_SHOT, token_budget=6000)
n_exhausted = int((demo["status"] == "budget_exhausted").sum())
print(f"budget demo: {n_exhausted} tickets budget_exhausted, tokens spent {demo_total}")
assert n_exhausted >= 1
demo.drop(columns="true_category").to_csv(os.path.join(OUT, "triage_budget_demo.csv"), index=False)

print("\n-- final run (few-shot, TOKEN_BUDGET=100000)")
final, final_log, final_total, _ = run(tickets, FEW_SHOT, token_budget=100000)
seg = pd.read_csv(os.path.join(OUT, "customer_segments.csv")).set_index("customer_id")
final["segment_name"] = final["customer_id"].map(seg["segment_name"]).fillna("")
premium = final["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)
human = final["status"].isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"])
fraud = ~human & (final["category"] == "fraud")
final["route"] = "support_queue"
final.loc[~human & ~fraud & premium, "route"] = "premium_desk"
final.loc[fraud, "route"] = "security_team"
final.loc[human, "route"] = "human_review"
final["priority"] = "NORMAL"
final.loc[final["route"].isin(["security_team", "premium_desk"]), "priority"] = "HIGH"
cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
final[cols].to_csv(os.path.join(OUT, "triage_results.csv"), index=False)
print(f"final status counts {final['status'].value_counts().to_dict()}; tokens {final_total}")
print("routes\n" + final.groupby(["route", "priority"]).size().to_string())
print(f"wrote outputs/triage_results.csv ({len(final)} rows) and outputs/triage_budget_demo.csv")
