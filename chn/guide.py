"""AI 向导：若设置了 OPENAI_API_KEY，则调用 OpenAI Chat Completions API。"""
import os
import requests

import sim

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SYSTEM_PROMPT = (
    "你是一款新加坡投资模拟游戏里的友好、耐心向导，面向完全的新手。"
    "玩家用 SGD 50,000 虚拟资金，投资虚构产品（货币基金、国债、公司债、股票、共同基金、ETF、"
    "结构化票据、期权、期货和外汇）。市场是模拟的。"
    "请用简洁、温暖的简体中文回答（2–5 句话，或几条要点），少用术语，必要时用白话解释，"
    "并结合玩家的真实持仓与近期事件。"
    "不要主动说教。不要给出真实理财建议或推荐现实世界产品；若被问到真实投资，"
    "只做一般性科普，并建议咨询持牌顾问。"
    "游戏中的产品和公司均为虚构。"
    "始终用简体中文回答。"
)


def enabled() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def context(state: dict) -> str:
    v = sim.view(state)
    m = v["market"]
    lines = [
        f"游戏日期：{v['date']}。现金 SGD {v['cash']:,.2f}。总资产 SGD {v['total']:,.2f} "
        f"（起始 SGD {v['start']:,.0f}，盈亏 SGD {v['gain']:,.2f}）。",
        f"市场：STI 指数 {m['sti']:,.0f}，新元短期利率 {m['rate']}%，10年期国债收益率 {m['y10']}%，"
        f"通胀 {m['infl']}%，美元/新元 {m['usd']}，高收益利差 {m['hy']}%。",
        "进行中的市场事件：" + ("、".join(f"{e['label']}（{e['headline']}）" for e in m["events"]) or "无"),
        "持仓：" if v["positions"] else "持仓：暂无（全部是现金）。",
    ]
    for p in v["positions"]:
        lines.append(f"- {p['name']} [{p['cls']}，风险 {p['risk']}/5]：市值 SGD {p['value']:,.2f}，成本 SGD "
                     f"{p['cost']:,.2f}，已收收益 SGD {p['income']:,.2f}，盈亏 SGD {p['pnl']:,.2f} "
                     f"（{p['pnl_pct']}%）。{p['details']}")
    lines.append("近期动态（由旧到新）：")
    for e in v["log"][-15:]:
        lines.append(f"- {e['date']}：{e.get('headline') or ''} {e['text']}")
    return "\n".join(lines)


def ask(state: dict, messages: list[dict]) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "当前游戏情境：\n" + context(state)}]
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
    ev = "；".join(f"{e['date']}：{e.get('headline') or e['text']}" for e in summary.get("events", [])) or "无重大事件"
    pos = "；".join(f"{p['name']}：{p['change']:+,.2f}" for p in summary.get("positions", [])) or "无持仓"
    q = (f"我刚把时间从 {summary['from_date']} 推进到 {summary['to_date']}。组合从 "
         f"SGD {summary['start_total']:,.2f} 变为 SGD {summary['end_total']:,.2f}。STI 变动 "
         f"{summary['sti_change_pct']}%。事件：{ev}。各持仓变动（含收益）：{pos}。"
         "请用 3–5 句简短白话，解释发生了什么，以及我的投资为何这样变动。请用简体中文回答。")
    return ask(state, [{"role": "user", "content": q}])
