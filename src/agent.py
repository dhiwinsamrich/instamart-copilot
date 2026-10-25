"""Part D: refund agent - tools, ReAct loop with safety limits, checkpointing and resume."""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import mock_llm  # noqa: E402
from rag import retrieve  # noqa: E402

DB = os.path.join(ROOT, "data", "instamart.db")
OUT = os.path.join(ROOT, "outputs")
STATE = os.path.join(ROOT, "state")
LEDGER = os.path.join(OUT, "refund_ledger.jsonl")
RUNS = os.path.join(OUT, "agent_runs.jsonl")
SYSTEM = ("You are the Instamart refund AGENT. Use the tools to look up the order, check the refund policy, "
          "calculate the refund and then issue it or escalate it. Finish with STATUS: explanation.")


# ---------------- tools ----------------
def get_order(order_id):
    with sqlite3.connect(DB) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(zip(["order_id", "status", "order_amount", "delivery_minutes"], row))


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0.0}


def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


def _spec(name, description, **params):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {k: {"type": t} for k, t in params.items()},
                           "required": list(params)}}


TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the order database.", order_id="string"),
    _spec("search_policy", "Retrieve the three policy passages most relevant to a query.", query="string"),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.",
          order_amount="number", delivery_minutes="number"),
    _spec("issue_refund", "Issue an approved refund of at most INR 1000 and record it in the ledger.",
          order_id="string", refund_inr="number"),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead for approval.",
          order_id="string", refund_inr="number"),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ---------------- agent loop ----------------
class SimulatedCrash(Exception):
    pass


def _save(state):
    os.makedirs(STATE, exist_ok=True)
    with open(os.path.join(STATE, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1)


def _fallback(state, status, reason):
    done = ", ".join(state["tools_executed"]) or "none"
    summary = f"{status}: handing to a human agent. {reason} Tools completed: {done}."
    return _finish(state, status, summary)


def _finish(state, status, final_answer):
    state.update(status=status, final_answer=final_answer)
    _save(state)
    return {k: state[k] for k in ["run_id", "scenario_id", "status", "final_answer", "steps", "llm_calls",
                                   "tokens_used", "tools_executed"]}


def _loop(state, crash_after_tool=None):
    lim = state["limits"]
    started = time.monotonic()
    while True:
        if time.monotonic() - started >= lim["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT", f"Time limit of {lim['time_limit_seconds']} s reached.")
        if state["llm_calls"] >= lim["max_iterations"]:
            return _fallback(state, "FALLBACK_MAX_ITERATIONS",
                             f"No final answer after {lim['max_iterations']} model calls.")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > lim["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT",
                             f"Token limit of {lim['token_limit']} exceeded ({state['tokens_used']} used).")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return _finish(state, msg.content.split(":")[0], msg.content)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        observation = TOOL_MAP[tc.name](**tc.arguments)
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                  "content": json.dumps(observation)})
        state["step"] += 1
        state["tools_executed"].append(tc.name)
        state["steps"].append({"step": state["step"], "tool": tc.name, "arguments": tc.arguments,
                               "observation": observation})
        _save(state)
        if crash_after_tool is not None and len(state["tools_executed"]) == crash_after_tool:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tool} ({tc.name})")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tool=None):
    state = {"run_id": run_id or scenario_id, "scenario_id": scenario_id,
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds},
             "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": []}
    return _loop(state, crash_after_tool)


def resume_agent(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        state = json.load(f)
    return _loop(state)


def refund_amount(result):
    for s in reversed(result["steps"]):
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0.0


def log_run(result, run_type):
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"run_type": run_type, **result}) + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(STATE, exist_ok=True)
    for path in (LEDGER, RUNS):
        open(path, "w").close()
    for fn in os.listdir(STATE):
        if fn.endswith(".json"):
            os.remove(os.path.join(STATE, fn))

    scenarios = pd.read_csv(os.path.join(ROOT, "data", "raw", "agent_scenarios.csv"))
    rows = []
    for sc in scenarios.itertuples(index=False):
        res = run_agent(sc.message, sc.scenario_id)
        log_run(res, "scenario")
        amt = refund_amount(res)
        ok = (res["status"] == sc.expected_outcome or
              (sc.expected_outcome == "FALLBACK" and res["status"].startswith("FALLBACK"))) and \
            abs(amt - sc.expected_refund_inr) <= 0.01
        rows.append({"scenario": sc.scenario_id, "expected": sc.expected_outcome, "status": res["status"],
                     "refund_inr": amt, "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
                     "tools": " > ".join(res["tools_executed"]), "match": ok})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)
    print("== scenario results\n" + table.to_string(index=False))
    assert table["match"].all()
    assert table.loc[table["scenario"] == "S7", "status"].item() == "FALLBACK_MAX_ITERATIONS"
    assert table.loc[table["scenario"] == "S1", "llm_calls"].item() == 5

    s1 = scenarios.set_index("scenario_id").loc["S1", "message"]
    tok = run_agent(s1, "S1", token_limit=200, run_id="S1-token-limit-demo")
    log_run(tok, "token_limit_demo")
    print(f"\n== S1 with token_limit=200: {tok['status']} after {tok['llm_calls']} calls, "
          f"{tok['tokens_used']} tokens, tools {tok['tools_executed']}\n   {tok['final_answer']}")
    assert tok["status"] == "FALLBACK_TOKEN_LIMIT"
    tmo = run_agent(s1, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
    log_run(tmo, "timeout_demo")
    print(f"== S1 with time_limit_seconds=0: {tmo['status']} after {tmo['llm_calls']} calls\n   {tmo['final_answer']}")
    assert tmo["status"] == "FALLBACK_TIMEOUT" and tmo["llm_calls"] == 0

    run_id = "S1-crash-resume-demo"
    try:
        run_agent(s1, "S1", run_id=run_id, crash_after_tool=2)
    except SimulatedCrash as err:
        with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
            before = json.load(f)["tools_executed"]
        print(f"\n== crash: {err}; checkpoint state/{run_id}.json holds tools {before}")
    resumed = resume_agent(run_id)
    log_run(resumed, "crash_resume_demo")
    after = resumed["tools_executed"][len(before):]
    print(f"resume executed {after}; all tools in order {resumed['tools_executed']}")
    print(f"final {resumed['status']} INR {refund_amount(resumed):.2f}, llm_calls {resumed['llm_calls']}")
    assert resumed["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["status"] == "REFUND" and abs(refund_amount(resumed) - 128.25) <= 0.01

    print("\n== refund ledger")
    print(open(LEDGER, encoding="utf-8").read().strip())
    print(f"\nwrote outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl and {len(os.listdir(STATE))} checkpoints in state/")
