"""FastAPI server for the SGD investing game. Stateless: the browser sends its game state with each call."""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import guide
import sim

app = FastAPI(title="Invest Game")
STATIC = Path(__file__).parent / "static"


class StateReq(BaseModel):
    state: dict


class TradeReq(StateReq):
    action: str                 # "buy" or "sell"
    product_id: str | None = None
    position_id: int | None = None
    amount: float | None = None  # SGD; for sell, None = sell everything


class AdvanceReq(StateReq):
    months: int


class ChatReq(StateReq):
    messages: list[dict]


class ExplainReq(StateReq):
    summary: dict


def _check(state: dict):
    if not isinstance(state, dict) or state.get("version") != 1 or "market" not in state:
        raise HTTPException(400, "Invalid or outdated game state. Please start a new game.")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "guide_enabled": guide.enabled()}


@app.post("/api/new")
def new_game():
    st = sim.new_game()
    return {"state": st, "view": sim.view(st)}


@app.post("/api/view")
def get_view(req: StateReq):
    _check(req.state)
    return {"state": req.state, "view": sim.view(req.state)}


@app.post("/api/trade")
def trade(req: TradeReq):
    st = req.state
    _check(st)
    try:
        if req.action == "buy":
            if st["cash"] < 0:
                raise sim.TradeError("You owe the broker money (negative cash). Sell something first.")
            sim.buy(st, req.product_id, req.amount)
        elif req.action == "sell":
            sim.sell(st, req.position_id, req.amount)
        else:
            raise sim.TradeError("Unknown action.")
    except sim.TradeError as e:
        raise HTTPException(400, str(e))
    st["history"][-1] = sim._snapshot(st)
    return {"state": st, "view": sim.view(st)}


@app.post("/api/advance")
def advance(req: AdvanceReq):
    st = req.state
    _check(st)
    try:
        summary = sim.advance(st, req.months)
    except sim.TradeError as e:
        raise HTTPException(400, str(e))
    return {"state": st, "view": sim.view(st), "summary": summary}


GUIDE_OFF = ("The AI guide will be switched on soon! For now, keep exploring: try different investments "
             "and move time forward to see what happens.")


@app.post("/api/chat")
def chat(req: ChatReq):
    _check(req.state)
    if not guide.enabled():
        return {"enabled": False, "reply": GUIDE_OFF}
    try:
        return {"enabled": True, "reply": guide.ask(req.state, req.messages)}
    except Exception:
        return {"enabled": True, "reply": "Sorry, I couldn't reach the guide just now. Please try again in a moment."}


@app.post("/api/explain")
def explain(req: ExplainReq):
    _check(req.state)
    if not guide.enabled():
        return {"enabled": False, "reply": GUIDE_OFF}
    try:
        return {"enabled": True, "reply": guide.explain_jump(req.state, req.summary)}
    except Exception:
        return {"enabled": True, "reply": "Sorry, I couldn't reach the guide just now. Please try again in a moment."}


app.mount("/static", StaticFiles(directory=STATIC), name="static")
