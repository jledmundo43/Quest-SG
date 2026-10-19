"""AI guide: calls the OpenAI Chat Completions API if OPENAI_API_KEY is set."""
import os
import requests

import sim

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "You are a friendly, patient guide inside an investing simulation game for a total beginner in Singapore. "
    "The player starts with SGD 50,000 of play money and invests in fictional products (money market funds, "
    "bonds, stocks, mutual funds, ETFs, structured notes, options, futures and FX). Markets are simulated. "
    "Answer questions in plain, warm language with short answers (2-5 sentences, or a few bullet points), "
    "avoid jargon or explain it simply, and use the player's actual portfolio and recent events when relevant. "
    "Do not lecture or teach unprompted. Never give real financial advice or recommend real-world products; "
    "if asked about real investing, give general education and suggest speaking to a licensed adviser. "
    "All products and companies in the game are fictional."
)


def enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def context(state: dict) -> str:
    v = sim.view(state)
    m = v["market"]
    lines = [
        f"Game date: {v['date']}. Cash SGD {v['cash']:,.2f}. Total value SGD {v['total']:,.2f} "
        f"(started with SGD {v['start']:,.0f}, gain/loss SGD {v['gain']:,.2f}).",
        f"Market: STI index {m['sti']:,.0f}, SGD short rate {m['rate']}%, 10Y gov yield {m['y10']}%, "
        f"inflation {m['infl']}%, USD/SGD {m['usd']}, high-yield spread {m['hy']}%.",
        "Active market events: " + (", ".join(f"{e['label']} ({e['headline']})" for e in m["events"]) or "none"),
        "Holdings:" if v["positions"] else "Holdings: none yet (all cash).",
    ]
    for p in v["positions"]:
        lines.append(f"- {p['name']} [{p['cls']}, risk {p['risk']}/5]: value SGD {p['value']:,.2f}, cost SGD "
                     f"{p['cost']:,.2f}, income received SGD {p['income']:,.2f}, P&L SGD {p['pnl']:,.2f} "
                     f"({p['pnl_pct']}%). {p['details']}")
    lines.append("Recent log (newest last):")
    for e in v["log"][-15:]:
        lines.append(f"- {e['date']}: {e.get('headline') or ''} {e['text']}")
    return "\n".join(lines)


def ask(state: dict, messages: list[dict]) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "Current game context:\n" + context(state)}]
    for mm in messages[-12:]:
        if mm.get("role") in ("user", "assistant") and isinstance(mm.get("content"), str):
            msgs.append({"role": mm["role"], "content": mm["content"][:2000]})
    r = requests.post("https://api.openai.com/v1/chat/completions",
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": MODEL, "messages": msgs, "temperature": 0.5, "max_tokens": 400},
                      timeout=45)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def explain_jump(state: dict, summary: dict) -> str:
    ev = "; ".join(f"{e['date']}: {e.get('headline') or e['text']}" for e in summary.get("events", [])) or "no major events"
    pos = "; ".join(f"{p['name']}: {p['change']:+,.2f}" for p in summary.get("positions", [])) or "no positions held"
    q = (f"I just moved time forward from {summary['from_date']} to {summary['to_date']}. My portfolio went from "
         f"SGD {summary['start_total']:,.2f} to SGD {summary['end_total']:,.2f}. The STI changed "
         f"{summary['sti_change_pct']}%. Events: {ev}. Change per position (incl. income): {pos}. "
         "In 3-5 short sentences, explain simply what happened and why my investments moved the way they did.")
    return ask(state, [{"role": "user", "content": q}])
