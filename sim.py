"""
Market simulation engine for the SGD investing game.

Everything is simulated in monthly steps. The whole game state is a plain
JSON-serialisable dict, so the browser can keep it in localStorage and send
it back with each request (the server is stateless).

Units: interest rates, yields and spreads are in percentage points (2.5 = 2.5%).
Prices are in SGD unless stated otherwise.
"""
from __future__ import annotations

import math
import random
import uuid
from copy import deepcopy

import numpy as np

START_YEAR = 2027
START_CASH = 50_000.0
CASH_RATE = 0.05          # % p.a. paid on idle cash in a bank savings account
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def month_label(t: int) -> str:
    return f"{MONTHS[t % 12]} {START_YEAR + t // 12}"


# ---------------------------------------------------------------------------
# Market factors and their correlations
# ---------------------------------------------------------------------------
FACTORS = ["sg", "gl", "r", "L", "ig", "hy", "USD", "EUR", "JPY", "AUD", "infl"]
# Annual volatility of each factor. Equity/FX: log-return vol. Rates/spreads/inflation: pp per year.
FACTOR_VOL = {"sg": 0.15, "gl": 0.15, "r": 0.6, "L": 0.7, "ig": 0.3, "hy": 1.2,
              "USD": 0.05, "EUR": 0.07, "JPY": 0.10, "AUD": 0.10, "infl": 0.8}
_CORR_PAIRS = {
    ("sg", "gl"): 0.75, ("sg", "r"): 0.10, ("gl", "r"): 0.10, ("sg", "L"): 0.15, ("gl", "L"): 0.20,
    ("r", "L"): 0.60, ("sg", "ig"): -0.35, ("gl", "ig"): -0.35, ("sg", "hy"): -0.55, ("gl", "hy"): -0.60,
    ("ig", "hy"): 0.70, ("USD", "sg"): -0.30, ("USD", "gl"): -0.20, ("USD", "hy"): 0.20,
    ("EUR", "sg"): 0.10, ("EUR", "gl"): 0.10, ("EUR", "USD"): 0.30,
    ("JPY", "sg"): -0.30, ("JPY", "gl"): -0.30, ("JPY", "USD"): 0.30, ("JPY", "EUR"): 0.30, ("JPY", "hy"): 0.20,
    ("AUD", "sg"): 0.40, ("AUD", "gl"): 0.45, ("AUD", "USD"): 0.10, ("AUD", "EUR"): 0.30,
    ("AUD", "JPY"): 0.10, ("AUD", "hy"): -0.35, ("infl", "r"): 0.20, ("infl", "L"): 0.30,
}


def _build_cholesky() -> np.ndarray:
    n = len(FACTORS)
    c = np.eye(n)
    for (a, b), v in _CORR_PAIRS.items():
        i, j = FACTORS.index(a), FACTORS.index(b)
        c[i, j] = c[j, i] = v
    # make sure the matrix is positive definite (clip tiny negative eigenvalues)
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 1e-4, None)
    c = v @ np.diag(w) @ v.T
    d = np.sqrt(np.diag(c))
    c = c / np.outer(d, d)
    return np.linalg.cholesky(c)


CHOL = _build_cholesky()

# Long-run means for mean-reverting variables and monthly mean-reversion speed
MEAN = {"r": 2.5, "L": 3.0, "ig": 1.0, "hy": 4.0, "infl": 2.5}
KAPPA = {"r": 0.04, "L": 0.03, "ig": 0.08, "hy": 0.08, "infl": 0.06}
FLOOR = {"r": 0.0, "L": 0.2, "ig": 0.3, "hy": 1.5, "infl": -1.0}
FOREIGN_RATE = {"EUR": 2.0, "JPY": 0.5, "AUD": 3.6}   # USD rate = SGD short rate + 1.0


def usd_rate(m):
    return m["r"] + 1.0


def foreign_rate(m, ccy):
    return usd_rate(m) if ccy == "USD" else FOREIGN_RATE[ccy]


def curve(m, T: float) -> float:
    """SGD government yield (%) for maturity T years: blends the short rate r and long rate L."""
    T = max(T, 1e-6)
    x = T / 2.0
    return m["L"] + (m["r"] - m["L"]) * (1 - math.exp(-x)) / x


# ---------------------------------------------------------------------------
# Market events
# Effects are applied EVERY month while the event is active.
#   equity keys (sg, gl, tech): extra monthly log return
#   r, L, ig, hy, infl: change in percentage points per month
#   fx: extra monthly log return of the foreign currency vs SGD
#   vol: volatility multiplier, hazard: default-probability multiplier
# ---------------------------------------------------------------------------
EVENTS = {
    "recession": dict(
        prob=0.010, dur=(3, 6), label="Recession / market crash",
        headlines=["Global recession fears: stock markets tumble",
                   "Market crash! Economy shrinks as companies cut jobs",
                   "Banking scare sparks a global sell-off"],
        text="Economies are slowing. Shares fall hard, investors rush to safe government bonds, "
             "riskier company bonds suffer and central banks start cutting rates.",
        eff=dict(sg=-0.055, gl=-0.055, tech=-0.01, r=-0.25, L=-0.12, ig=0.15, hy=0.6, infl=-0.15,
                 fx={"USD": 0.008, "EUR": -0.004, "JPY": 0.015, "AUD": -0.02}, vol=1.8, hazard=2.5)),
    "hike": dict(
        prob=0.018, dur=(3, 3), label="Central bank rate hike",
        headlines=["US Fed raises interest rates; Singapore rates follow",
                   "Central banks turn hawkish: borrowing costs climb"],
        text="Higher interest rates make existing bonds less attractive (their prices fall), "
             "money market funds earn more, and growth stocks usually dip.",
        eff=dict(sg=-0.006, gl=-0.008, tech=-0.01, r=0.25, L=0.08, fx={"USD": 0.004}, vol=1.1)),
    "cut": dict(
        prob=0.018, dur=(3, 3), label="Central bank rate cut",
        headlines=["US Fed cuts interest rates; SGD rates ease",
                   "Central banks cut rates to support growth"],
        text="Lower interest rates push existing bond prices up, money market funds earn less, "
             "and shares usually get a lift.",
        eff=dict(sg=0.008, gl=0.010, tech=0.01, r=-0.25, L=-0.06, fx={"USD": -0.004})),
    "inflation": dict(
        prob=0.008, dur=(6, 6), label="Inflation spike",
        headlines=["Inflation spikes as energy and food prices soar",
                   "Prices rising at fastest pace in years"],
        text="Fast-rising prices erode the value of cash. Rates and bond yields rise, so both bonds and "
             "shares can fall together. MAS lets the Singapore dollar strengthen to fight inflation.",
        eff=dict(sg=-0.010, gl=-0.012, tech=-0.01, r=0.15, L=0.15, infl=0.5,
                 fx={"USD": -0.003, "EUR": -0.003, "JPY": -0.003, "AUD": -0.003}, vol=1.2)),
    "bull": dict(
        prob=0.012, dur=(6, 9), label="Bull run",
        headlines=["Bull market! Stocks hit record highs",
                   "Tech boom lifts markets worldwide"],
        text="Investors are optimistic. Shares rise steadily, riskier bonds do well and volatility is low.",
        eff=dict(sg=0.020, gl=0.025, tech=0.015, ig=-0.03, hy=-0.12, fx={"AUD": 0.004}, vol=0.85)),
    "fx": dict(
        prob=0.010, dur=(2, 2), label="Currency shock",
        variants=[
            ("US dollar surges against Asian currencies", {"USD": 0.03, "JPY": -0.01, "AUD": -0.01}),
            ("Singapore dollar jumps after surprise MAS move", {"USD": -0.025, "EUR": -0.025, "JPY": -0.025, "AUD": -0.025}),
            ("Euro slides on European debt worries", {"EUR": -0.035}),
            ("Japanese yen soars as investors seek safety", {"JPY": 0.04}),
            ("Australian dollar plunges as commodity prices fall", {"AUD": -0.04}),
        ],
        text="Currencies can move sharply. Anything you hold in a foreign currency (FX deposits, "
             "overseas funds) changes value in SGD terms.",
        eff=dict(vol=1.1)),
    "credit": dict(
        prob=0.008, dur=(2, 2), label="Credit event",
        headlines=["Credit scare: a major property developer misses a bond payment",
                   "Corporate debt worries rattle bond markets"],
        text="Investors worry that companies may not repay their debts. High-yield bonds fall sharply, "
             "safe government bonds gain, and shares wobble.",
        eff=dict(sg=-0.015, gl=-0.012, ig=0.2, hy=0.8, L=-0.03, vol=1.3, hazard=4.0)),
}
EXCLUSIVE = {"recession": {"bull"}, "bull": {"recession"}, "hike": {"cut"}, "cut": {"hike"}}


def _event_drift(key: str) -> float:
    """Average yearly equity log-drift that events add; base drifts are adjusted so long-run returns match targets."""
    tot = 0.0
    for e in EVENTS.values():
        mean_dur = (e["dur"][0] + e["dur"][1]) / 2
        tot += e["prob"] * 12 * mean_dur * e["eff"].get(key, 0.0)
    return tot


# Target long-run expected returns (annual, arithmetic, local currency)
SG_PRICE_RET = 0.035     # STI price return (plus ~3.5% dividends = ~7% total)
GL_TOTAL_RET = 0.08      # global equities total return in USD
CALIBRATION = 0.006  # small top-up found by Monte Carlo (events cannot overlap, fee drag, etc.)
SG_DRIFT = SG_PRICE_RET - _event_drift("sg") + CALIBRATION
GL_DRIFT = GL_TOTAL_RET - _event_drift("gl") + CALIBRATION
TECH_EVENT = _event_drift("tech")

# ---------------------------------------------------------------------------
# Product catalogue
# ---------------------------------------------------------------------------
CLASSES = ["Money market funds", "Government bonds", "Corporate bonds", "Stocks", "Mutual funds",
           "ETFs", "Structured notes", "Options", "Futures", "FX"]

STOCKS = {  # beta to factor, idiosyncratic vol, alpha (annual), dividend yield, rate sensitivity
    "MERLION": dict(factor="sg", beta=1.0, idio=0.14, alpha=0.0, div=0.05, rate_sens=0.0),
    "HARBOUR": dict(factor="sg", beta=0.5, idio=0.12, alpha=-0.01, div=0.06, rate_sens=0.0),
    "ORCHID": dict(factor="gl", beta=1.4, idio=0.35, alpha=0.01, div=0.0, rate_sens=-4.0),
}


def stock_vol(sym):
    s = STOCKS[sym]
    return math.sqrt((s["beta"] * FACTOR_VOL[s["factor"]]) ** 2 + s["idio"] ** 2)


P = []  # list of product dicts


def _p(**kw):
    kw.setdefault("min", 100.0)
    P.append(kw)


_p(id="MMF_SGD", cls="Money market funds", name="Lion SGD Money Market Fund", kind="fund", risk=1,
   fee=0.3, buy_cost=0.0, sell_cost=0.0,
   desc="Pools your money into very short-term bank deposits and government bills. Earns close to the "
        "current SGD interest rate, is very stable and you can withdraw any time. Fee 0.3% a year.")
_p(id="MMF_USD", cls="Money market funds", name="Lion USD Money Market Fund", kind="fund", risk=2,
   fee=0.3, buy_cost=0.0, sell_cost=0.0, ccy="USD",
   desc="Same idea, but invested in US dollars. US rates are usually higher than SGD rates, but your "
        "returns in SGD also go up and down with the USD/SGD exchange rate. Fee 0.3% a year.")
for pid, name, T, risk in [("SGS_2Y", "Singapore Government Bond 2Y", 2, 1),
                           ("SGS_10Y", "Singapore Government Bond 10Y", 10, 2),
                           ("SGS_30Y", "Singapore Government Bond 30Y", 30, 3)]:
    _p(id=pid, cls="Government bonds", name=name, kind="bond", risk=risk, tenor=T, spread=None,
       min=1000.0, sell_cost=0.003,
       desc=f"You lend money to the Singapore government for {T} years. It pays a fixed interest "
            f"(coupon) every 6 months and returns your money at the end. Practically no default risk, "
            f"but if interest rates rise before maturity, the bond's market price falls"
            f"{' a lot, because it is so long' if T >= 30 else ''}.")
_p(id="CORP_IG", cls="Corporate bonds", name="Raffles Utilities 5Y Bond (A-rated)", kind="bond", risk=2,
   tenor=5, spread="ig", issuer="Raffles Utilities", recovery=0.5, hazard=0.002, min=1000.0, sell_cost=0.005,
   desc="A loan to a strong, investment-grade company. Pays a bit more interest than government bonds. "
        "Default is rare but not impossible.")
_p(id="CORP_HY", cls="Corporate bonds", name="Lion City Resorts 4Y Bond (high-yield, BB-)", kind="bond",
   risk=4, tenor=4, spread="hy", issuer="Lion City Resorts", recovery=0.4, hazard=0.015, min=1000.0, sell_cost=0.01,
   desc="A loan to a riskier company, so it pays a much higher coupon. In bad times its price can drop "
        "sharply and the company could default, returning only part of your money.")
_p(id="MERLION", cls="Stocks", name="Merlion Bank (blue-chip)", kind="fund", risk=3, buy_cost=0.001,
   sell_cost=0.001, div=0.05,
   desc="A large, established Singapore bank. Pays a steady dividend (about 5% a year, every quarter). "
        "The share price moves with the local market.")
_p(id="HARBOUR", cls="Stocks", name="Harbour Telecom (defensive)", kind="fund", risk=3, buy_cost=0.001,
   sell_cost=0.001, div=0.06,
   desc="A telecom company. People pay phone bills in good times and bad, so it moves less than the "
        "market and pays a high dividend, but grows slowly.")
_p(id="ORCHID", cls="Stocks", name="Orchid Tech (growth)", kind="fund", risk=4, buy_cost=0.001,
   sell_cost=0.001, div=0.0,
   desc="A fast-growing technology company. No dividend: it reinvests everything. The share price can "
        "soar or crash and is sensitive to interest rates.")
_p(id="MF_GLOBAL", cls="Mutual funds", name="Evergreen Global Equity Fund (actively managed)", kind="fund",
   risk=3, fee=1.75, buy_cost=0.03, sell_cost=0.0, ccy="USD",
   desc="A professional manager picks shares around the world. Diversified, but you pay a 3% sales charge "
        "when you buy plus 1.75% a year in fees. Invested in foreign currencies (unhedged).")
_p(id="MF_BAL", cls="Mutual funds", name="Kopi Balanced Fund (50% shares / 50% bonds)", kind="fund",
   risk=2, fee=1.5, buy_cost=0.03, sell_cost=0.0,
   desc="Half Singapore shares, half high-quality bonds. Smoother ride than pure shares. 3% sales charge "
        "plus 1.5% a year in fees.")
_p(id="ETF_STI", cls="ETFs", name="STI Index ETF (Straits 30)", kind="fund", risk=3, fee=0.3,
   buy_cost=0.001, sell_cost=0.001, div=0.035,
   desc="Buys all 30 big companies in Singapore's main stock index in one go. Low fee (0.3% a year), "
        "traded like a share, pays dividends every 6 months.")
_p(id="ETF_WORLD", cls="ETFs", name="Global World Equity ETF (USD)", kind="fund", risk=3, fee=0.2,
   buy_cost=0.001, sell_cost=0.001, ccy="USD",
   desc="Thousands of companies around the world in a single, very cheap fund (0.2% a year). Priced in US "
        "dollars, so the USD/SGD exchange rate also affects you. Dividends are reinvested.")
_p(id="ETF_SGBOND", cls="ETFs", name="SG Government Bond ETF", kind="fund", risk=2, fee=0.25,
   buy_cost=0.001, sell_cost=0.001,
   desc="A basket of Singapore government bonds (average maturity about 7 years). Earns interest; the "
        "price falls when rates rise and rises when rates fall. Interest is reinvested.")
_p(id="NOTE_CPN", cls="Structured notes", name="3Y Capital-Protected Note on the STI", kind="cpn",
   risk=2, tenor=36, min=5000.0, sell_cost=0.02,
   desc="Hold it for 3 years and you get at least 100% of your money back, plus a share (the "
        "'participation rate') of any rise in the STI index. No interest meanwhile. Selling early "
        "means accepting the market price, which can be below what you paid.")
_p(id="NOTE_RC_MERLION", cls="Structured notes", name="1Y Reverse Convertible on Merlion Bank",
   kind="rc", risk=4, tenor=12, und="MERLION", coupon=8.0, barrier=0.70, min=5000.0, sell_cost=0.02,
   desc="Pays a high 8% a year coupon (quarterly). After 1 year you get your money back IF Merlion Bank "
        "never fell 30% or more (the 'knock-in barrier', checked monthly). If it did and ends below its "
        "starting price, you are given Merlion shares instead, which are worth less.")
_p(id="NOTE_RC_ORCHID", cls="Structured notes", name="1Y Reverse Convertible on Orchid Tech",
   kind="rc", risk=5, tenor=12, und="ORCHID", coupon=16.0, barrier=0.65, min=5000.0, sell_cost=0.02,
   desc="An even juicier 16% a year coupon, because the underlying stock (Orchid Tech) is very volatile. "
        "If it ever drops 35% or more and finishes below its starting price, you receive Orchid shares "
        "instead of your money back.")
_p(id="OPT_STI_CALL", cls="Options", name="STI 6-month Call (strike +5%)", kind="option", risk=5,
   und="sg", otype="call", tenor=6, moneyness=1.05, sell_cost=0.03,
   desc="A bet that the STI index will be MORE than 5% higher in 6 months. You pay a small premium up "
        "front. If the index ends above the strike you earn the difference (can be many times your "
        "premium); otherwise the option expires worthless and you lose the whole premium.")
_p(id="OPT_STI_PUT", cls="Options", name="STI 6-month Put (strike -5%)", kind="option", risk=5,
   und="sg", otype="put", tenor=6, moneyness=0.95, sell_cost=0.03,
   desc="Pays off if the STI index falls MORE than 5% in 6 months, like insurance against a crash. If "
        "the market doesn't fall that far, it expires worthless and you lose the premium.")
_p(id="OPT_ORCH_CALL", cls="Options", name="Orchid Tech 1-year Call (at-the-money)", kind="option",
   risk=5, und="ORCHID", otype="call", tenor=12, moneyness=1.0, sell_cost=0.03,
   desc="Pays the amount by which Orchid Tech's share price ends above today's price in 1 year. Big upside "
        "if the stock soars; worthless if it ends lower.")
for side, pid in [(1, "FUT_STI_LONG"), (-1, "FUT_STI_SHORT")]:
    _p(id=pid, cls="Futures", name=f"STI Index Futures - {'Long (profit if market rises)' if side > 0 else 'Short (profit if market falls)'}",
       kind="future", risk=5, side=side, leverage=10.0, maint=0.05, sell_cost=0.0005, min=1000.0,
       desc="Leveraged contract on the STI index. The money you put in is only a 10% deposit (margin), so "
            "you control 10x as much: a 5% market move = a 50% gain or loss on your money. If you lose half "
            "your deposit you get a MARGIN CALL: more cash is taken from your account automatically, "
            "or the position is closed. You can lose more than you put in.")
for ccy, name, risk in [("USD", "US Dollar (USD/SGD)", 2), ("EUR", "Euro (EUR/SGD)", 3),
                        ("JPY", "Japanese Yen (JPY/SGD, per 100 yen)", 3), ("AUD", "Australian Dollar (AUD/SGD)", 3)]:
    _p(id=f"FX_{ccy}", cls="FX", name=f"{name} deposit", kind="fund", risk=risk, ccy=ccy,
       buy_cost=0.0025, sell_cost=0.0025,
       desc=f"Convert SGD into {ccy} and keep it in a {ccy} deposit earning {ccy} interest. You gain if "
            f"{ccy} strengthens against SGD and lose if it weakens. 0.25% conversion spread each way.")

PRODUCTS = {p["id"]: p for p in P}


# ---------------------------------------------------------------------------
# Pricing helpers
# ---------------------------------------------------------------------------
def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, r, q, vol, otype):
    """Black-Scholes price. r, q as decimals, T in years."""
    if T <= 1e-9 or vol <= 1e-9:
        return max(S - K, 0.0) if otype == "call" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    if otype == "call":
        return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


def bond_price(face, coupon, yld, months_left, months_since_issue):
    """Value of a bond with semi-annual coupons. coupon/yld in %."""
    if months_left <= 0:
        return face
    y = yld / 100 / 2
    c = face * coupon / 100 / 2
    # next coupon in m0 months, then every 6 months
    m0 = 6 - (months_since_issue % 6)
    pv, m = 0.0, m0
    while m <= months_left + 1e-9:
        pv += c / (1 + y) ** (m / 6)
        m += 6
    pv += face / (1 + y) ** (months_left / 6)
    return pv


def vol_mult(m):
    v = 1.0
    for e in m["events"]:
        v *= e["eff"].get("vol", 1.0)
    return v


def underlying_price(m, und):
    return m["sg"] if und == "sg" else m["px"][und]


def underlying_vol(und):
    return FACTOR_VOL["sg"] if und == "sg" else stock_vol(und)


def underlying_div(und):
    return 0.035 if und == "sg" else STOCKS[und]["div"]


IMPLIED_VOL_PREMIUM = 1.2  # options are priced with a bit more volatility than the model uses


def option_value(pos, m, t):
    S = underlying_price(m, pos["und"])
    T = max(pos["expiry_t"] - t, 0) / 12
    iv = underlying_vol(pos["und"]) * vol_mult(m) * IMPLIED_VOL_PREMIUM
    return pos["qty"] * bs_price(S, pos["strike"], T, m["r"] / 100, underlying_div(pos["und"]), iv, pos["otype"])


def cpn_unit_value(m, S_rel, months_left, part):
    T = months_left / 12
    if T <= 0:
        return 1 + part * max(S_rel - 1, 0)
    y = (curve(m, T) + 0.5) / 100
    iv = FACTOR_VOL["sg"] * vol_mult(m) * IMPLIED_VOL_PREMIUM
    return (1 + y) ** (-T) + part * bs_price(S_rel, 1.0, T, m["r"] / 100, 0.035, iv, "call")


_MC_Z = np.random.default_rng(1234).standard_normal((1000, 36))
_MC_Z = np.vstack([_MC_Z, -_MC_Z])  # antithetic paths for stable estimates


def rc_value(pos, m, t):
    """Mark-to-market of a reverse convertible via a small Monte Carlo."""
    n = pos["mat_t"] - t
    S = m["px"][pos["und"]]
    s0, prin = pos["s0"], pos["principal"]
    if n <= 0:
        return prin * (S / s0 if pos["ki"] and S < s0 else 1.0)
    vol = stock_vol(pos["und"]) * vol_mult(m)
    r = m["r"] / 100
    q = STOCKS[pos["und"]]["div"]
    dt = 1 / 12
    z = _MC_Z[:, :n]
    logp = math.log(S / s0) + np.cumsum((r - q - 0.5 * vol * vol) * dt + vol * math.sqrt(dt) * z, axis=1)
    ki = np.full(z.shape[0], bool(pos["ki"])) | (logp.min(axis=1) <= math.log(pos["barrier"]))
    final = np.exp(logp[:, -1])
    payoff = np.where(ki & (final < 1.0), final, 1.0)
    disc_rate = (curve(m, n / 12) + 0.5) / 100
    pv = prin * float(payoff.mean()) / (1 + disc_rate) ** (n / 12)
    # remaining quarterly coupons
    for k in range(1, 5):
        cm = pos["issue_t"] + 3 * k
        if t < cm <= pos["mat_t"]:
            pv += prin * pos["coupon"] / 100 / 4 / (1 + disc_rate) ** ((cm - t) / 12)
    return pv


def bond_yield(m, prod, months_left):
    y = curve(m, months_left / 12)
    if prod["spread"]:
        y += m[prod["spread"]]
    return y


def position_value(pos, m, t):
    prod = PRODUCTS[pos["pid"]]
    k = pos["kind"]
    if k == "fund":
        return pos["units"] * m["px"][pos["pid"]]
    if k == "bond":
        left = pos["mat_t"] - t
        return bond_price(pos["face"], pos["coupon"], bond_yield(m, prod, left), left, t - pos["issue_t"])
    if k == "option":
        return option_value(pos, m, t)
    if k == "future":
        return pos["margin"]
    if k == "cpn":
        return pos["principal"] * cpn_unit_value(m, m["sg"] / pos["s0"], pos["mat_t"] - t, pos["part"])
    if k == "rc":
        return rc_value(pos, m, t)
    raise ValueError(k)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------
def new_game(seed: int | None = None) -> dict:
    seed = seed if seed is not None else random.randrange(1, 2**31)
    m = dict(r=2.5, L=3.0, ig=1.0, hy=4.0, infl=2.5, cpi=100.0, sg=3500.0, gl=1000.0,
             fx={"USD": 1.35, "EUR": 1.47, "JPY": 0.90, "AUD": 0.88}, events=[],
             px={"MMF_SGD": 1.0, "MMF_USD": 1.35, "MERLION": 35.0, "HARBOUR": 2.50, "ORCHID": 25.0,
                 "MF_GLOBAL": 1.0, "MF_BAL": 1.0, "ETF_STI": 3.50, "ETF_WORLD": 135.0, "ETF_SGBOND": 1.0,
                 "FX_USD": 1.35, "FX_EUR": 1.47, "FX_JPY": 0.90, "FX_AUD": 0.88})
    st = dict(version=1, id=uuid.uuid4().hex[:12], seed=seed, t=0, cash=START_CASH, market=m,
              positions=[], log=[], history=[], next_id=1, defaults={}, invested=START_CASH)
    _log(st, "info", f"Welcome! You have SGD {START_CASH:,.0f} in cash. Pick some investments, then move time forward.")
    st["history"].append(_snapshot(st))
    return st


def _log(st, kind, text, headline=None):
    st["log"].append(dict(t=st["t"], date=month_label(st["t"]), kind=kind, text=text, headline=headline))


def total_value(st):
    return st["cash"] + sum(position_value(p, st["market"], st["t"]) for p in st["positions"])


def _snapshot(st):
    m = st["market"]
    return dict(t=st["t"], date=month_label(st["t"]), total=round(total_value(st), 2),
                cash=round(st["cash"], 2), sti=round(m["sg"], 1), rate=round(m["r"], 2),
                infl=round(m["infl"], 2), cpi=round(m["cpi"], 2))


class TradeError(Exception):
    pass


def buy(st, pid, amount):
    if pid not in PRODUCTS:
        raise TradeError("Unknown product.")
    prod = PRODUCTS[pid]
    amount = float(amount)
    if not math.isfinite(amount) or amount <= 0:
        raise TradeError("Enter an amount above zero.")
    if amount < prod["min"]:
        raise TradeError(f"Minimum investment for this product is SGD {prod['min']:,.0f}.")
    if amount > st["cash"] + 1e-6:
        raise TradeError(f"Not enough cash. You have SGD {st['cash']:,.2f} available.")
    m, t = st["market"], st["t"]
    k = prod["kind"]
    pos = dict(id=st["next_id"], pid=pid, kind=k, cost=amount, income=0.0, opened_t=t)
    if k == "fund":
        units = amount * (1 - prod.get("buy_cost", 0)) / m["px"][pid]
        existing = next((p for p in st["positions"] if p["pid"] == pid and p["kind"] == "fund"), None)
        if existing:
            existing["units"] += units
            existing["cost"] += amount
            pos = existing
        else:
            pos["units"] = units
            st["positions"].append(pos)
            st["next_id"] += 1
    else:
        if k == "bond":
            months = prod["tenor"] * 12
            cpn = round(bond_yield(m, prod, months) * 20) / 20  # new issue at par, coupon to nearest 0.05%
            pos.update(face=amount, coupon=cpn, issue_t=t, mat_t=t + months)
            # bought at par: adjust face so the price we pay equals the fair value
            pos["face"] = amount * amount / bond_price(amount, cpn, bond_yield(m, prod, months), months, 0)
        elif k == "option":
            S = underlying_price(m, prod["und"])
            pos.update(und=prod["und"], otype=prod["otype"], strike=S * prod["moneyness"], spot0=S,
                       expiry_t=t + prod["tenor"], qty=1.0)
            unit = option_value(pos, m, t) * (1 + prod["sell_cost"])  # pay the 'ask' price
            pos["qty"] = amount / unit
        elif k == "future":
            S = m["sg"]
            pos.update(side=prod["side"], margin=amount, units=amount * prod["leverage"] / S, entry=S, last=S)
        elif k == "cpn":
            pos.update(principal=amount, s0=m["sg"], issue_t=t, mat_t=t + prod["tenor"], part=0.0)
            base = cpn_unit_value(m, 1.0, prod["tenor"], 0.0)
            opt = cpn_unit_value(m, 1.0, prod["tenor"], 1.0) - base
            pos["part"] = round(max(0.1, min(1.2, (0.97 - base) / opt)), 2) if opt > 0 else 0.1
        elif k == "rc":
            pos.update(principal=amount, und=prod["und"], s0=m["px"][prod["und"]], coupon=prod["coupon"],
                       barrier=prod["barrier"], ki=False, issue_t=t, mat_t=t + prod["tenor"])
        st["positions"].append(pos)
        st["next_id"] += 1
    st["cash"] -= amount
    _log(st, "trade", f"Bought {prod['name']} for SGD {amount:,.2f}.")
    return pos


def sell(st, pos_id, amount=None):
    """Sell all of a position (amount=None) or roughly SGD `amount` worth of it."""
    pos = next((p for p in st["positions"] if p["id"] == int(pos_id)), None)
    if not pos:
        raise TradeError("Position not found.")
    prod = PRODUCTS[pos["pid"]]
    value = position_value(pos, st["market"], st["t"])
    if value <= 0:
        frac = 1.0
    elif amount is None:
        frac = 1.0
    else:
        amount = float(amount)
        if not math.isfinite(amount) or amount <= 0:
            raise TradeError("Enter an amount above zero.")
        frac = min(1.0, amount / value)
        if frac > 0.995:
            frac = 1.0
    proceeds = max(value, 0.0) * frac * (1 - prod.get("sell_cost", 0))
    st["cash"] += proceeds
    if frac >= 1.0:
        st["positions"].remove(pos)
    else:
        for key in ("units", "face", "qty", "principal", "margin", "cost", "income"):
            if key in pos:
                pos[key] *= (1 - frac)
    _log(st, "trade", f"Sold {'all of ' if frac >= 1 else f'{frac:.0%} of '}{prod['name']} for SGD {proceeds:,.2f}.")
    return proceeds


# ---------------------------------------------------------------------------
# Monthly step
# ---------------------------------------------------------------------------
def _maybe_start_event(st, rng):
    m = st["market"]
    active = {e["type"] for e in m["events"]}
    u = rng.random()
    acc = 0.0
    for key, e in EVENTS.items():
        acc += e["prob"]
        if u < acc:
            if key in active or EXCLUSIVE.get(key, set()) & active:
                return None
            months = int(rng.integers(e["dur"][0], e["dur"][1] + 1))
            eff = deepcopy(e["eff"])
            if key == "fx":
                headline, fxeff = e["variants"][int(rng.integers(len(e["variants"])))]
                eff["fx"] = fxeff
            else:
                headline = e["headlines"][int(rng.integers(len(e["headlines"])))]
            ev = dict(type=key, label=e["label"], headline=headline, left=months, eff=eff)
            m["events"].append(ev)
            _log(st, "event", e["text"], headline=f"{e['label']}: {headline}")
            return ev
    return None


def step_month(st):
    m, t = st["market"], st["t"]
    rng = np.random.default_rng([st["seed"], t])
    ev_new = _maybe_start_event(st, rng)
    vm = vol_mult(m)
    z = CHOL @ rng.standard_normal(len(FACTORS))
    sh = dict(zip(FACTORS, z))
    eff = {}
    for e in m["events"]:
        for k2, v in e["eff"].items():
            if k2 == "fx":
                for c, fv in v.items():
                    eff["fx_" + c] = eff.get("fx_" + c, 0.0) + fv
            elif k2 not in ("vol", "hazard"):
                eff[k2] = eff.get(k2, 0.0) + v
    hazard_mult = 1.0
    for e in m["events"]:
        hazard_mult *= e["eff"].get("hazard", 1.0)
    dt = 1 / 12
    sq = math.sqrt(dt)

    # --- rates, spreads, inflation (mean-reverting with event pushes)
    old = {k2: m[k2] for k2 in ("r", "L", "ig", "hy")}
    y_old = {"g7": curve(m, 7), "ig5": curve(m, 5) + m["ig"]}
    for k2 in ("r", "L", "ig", "hy", "infl"):
        m[k2] += KAPPA[k2] * (MEAN[k2] - m[k2]) + FACTOR_VOL[k2] * sq * sh[k2] * (vm if k2 in ("ig", "hy") else 1) + eff.get(k2, 0.0)
        m[k2] = max(FLOOR[k2], m[k2])
    m["cpi"] *= 1 + m["infl"] / 1200
    dL = m["L"] - old["L"]

    # --- equity factors (log returns)
    s_sg = FACTOR_VOL["sg"] * vm
    s_gl = FACTOR_VOL["gl"] * vm
    ret_sg = (SG_DRIFT - 0.5 * s_sg ** 2) * dt + s_sg * sq * sh["sg"] + eff.get("sg", 0.0)
    ret_gl = (GL_DRIFT - 0.5 * s_gl ** 2) * dt + s_gl * sq * sh["gl"] + eff.get("gl", 0.0)
    m["sg"] *= math.exp(ret_sg)
    m["gl"] *= math.exp(ret_gl)

    # --- FX (random walk; slight drift against the interest-rate gap)
    fxret = {}
    for c in ("USD", "EUR", "JPY", "AUD"):
        s = FACTOR_VOL[c] * (1 + (vm - 1) * 0.5)
        drift = -0.5 * (foreign_rate(m, c) - m["r"]) / 100
        fxret[c] = (drift - 0.5 * s * s) * dt + s * sq * sh[c] + eff.get("fx_" + c, 0.0)
        m["fx"][c] *= math.exp(fxret[c])

    # --- product prices
    px = m["px"]
    rets = {"sg": ret_sg, "gl": ret_gl}
    for sym, s in STOCKS.items():
        idio = s["idio"] * vm
        r_ = (s["beta"] * rets[s["factor"]] + (s["alpha"] - 0.5 * idio ** 2) * dt
              + idio * sq * float(rng.standard_normal()) + s["rate_sens"] * dL / 100)
        if sym == "ORCHID":
            r_ += eff.get("tech", 0.0) - TECH_EVENT * dt
        px[sym] *= math.exp(r_)
    px["MMF_SGD"] *= 1 + max(m["r"] - 0.3, 0) / 1200
    px["MMF_USD"] *= math.exp(fxret["USD"]) * (1 + max(usd_rate(m) - 0.3, 0) / 1200)
    px["ETF_STI"] *= math.exp(ret_sg) * (1 - 0.3 / 1200)
    px["ETF_WORLD"] *= math.exp(ret_gl + fxret["USD"]) * (1 - 0.2 / 1200)
    alpha = 0.03 * sq * float(rng.standard_normal())
    px["MF_GLOBAL"] *= math.exp(ret_gl + fxret["USD"] + alpha) * (1 - 1.75 / 1200)

    def bond_ret(y0, y1, D):
        d = (y1 - y0) / 100
        return y0 / 1200 - D * d + 0.5 * D * D * d * d

    r_g7 = bond_ret(y_old["g7"], curve(m, 7), 6.3)
    r_ig5 = bond_ret(y_old["ig5"], curve(m, 5) + m["ig"], 4.5)
    px["ETF_SGBOND"] *= (1 + r_g7) * (1 - 0.25 / 1200)
    px["MF_BAL"] *= (1 + 0.5 * (math.exp(ret_sg) - 1 + 0.035 / 12) + 0.5 * r_ig5) * (1 - 1.5 / 1200)
    for c in ("USD", "EUR", "JPY", "AUD"):
        px["FX_" + c] *= math.exp(fxret[c]) * (1 + foreign_rate(m, c) / 1200)

    st["t"] = t = t + 1
    # interest on idle cash (only if positive)
    if st["cash"] > 0:
        st["cash"] *= 1 + CASH_RATE / 1200

    # --- defaults (issuer level)
    for prod in (PRODUCTS["CORP_IG"], PRODUCTS["CORP_HY"]):
        h = prod["hazard"] * hazard_mult * (m[prod["spread"]] / MEAN[prod["spread"]]) ** 1.5
        p_def = h / 12
        if ev_new and ev_new["type"] == "credit" and prod["id"] == "CORP_HY":
            p_def = max(p_def, 0.25)
        if rng.random() < p_def:
            name = prod["issuer"]
            held = [p for p in st["positions"] if p["pid"] == prod["id"]]
            msg = f"{name} has DEFAULTED and cannot repay its bonds in full."
            if held:
                rec = sum(p["face"] for p in held) * prod["recovery"]
                st["cash"] += rec
                for p in held:
                    st["positions"].remove(p)
                msg += f" Bondholders recover {prod['recovery']:.0%} of face value: you received SGD {rec:,.2f}."
            else:
                msg += " Luckily you didn't own any. New bonds of a similar issuer remain available."
            _log(st, "event", msg, headline=f"Default: {name} misses bond payments")

    # --- cash flows, maturities, expiries, margin
    for pos in list(st["positions"]):
        prod = PRODUCTS[pos["pid"]]
        k = pos["kind"]
        if k == "fund" and prod.get("div"):
            freq = 6 if pos["pid"] == "ETF_STI" else 3
            if t % freq == 0:
                d = pos["units"] * px[pos["pid"]] * prod["div"] * freq / 12
                st["cash"] += d
                pos["income"] += d
        elif k == "bond":
            since = t - pos["issue_t"]
            if since % 6 == 0 and since > 0:
                c = pos["face"] * pos["coupon"] / 200
                st["cash"] += c
                pos["income"] += c
            if t >= pos["mat_t"]:
                st["cash"] += pos["face"]
                st["positions"].remove(pos)
                _log(st, "info", f"{prod['name']} matured: SGD {pos['face']:,.2f} returned to your cash "
                                 f"(total gain incl. coupons SGD {pos['face'] + pos['income'] - pos['cost']:,.2f}).")
        elif k == "option":
            if t >= pos["expiry_t"]:
                pay = option_value(pos, m, t)
                st["cash"] += pay
                st["positions"].remove(pos)
                if pay <= 0.005:
                    _log(st, "warning", f"{prod['name']} expired worthless: the SGD {pos['cost']:,.2f} premium is lost.")
                else:
                    _log(st, "info", f"{prod['name']} expired in the money and paid SGD {pay:,.2f} "
                                     f"({pay / pos['cost']:.1f}x your premium of SGD {pos['cost']:,.2f}).")
        elif k == "future":
            S = m["sg"]
            pos["margin"] += pos["side"] * pos["units"] * (S - pos["last"])
            pos["last"] = S
            notional = pos["units"] * S
            if pos["margin"] < prod["maint"] * notional:
                need = notional / prod["leverage"] - pos["margin"]
                if st["cash"] >= need:
                    st["cash"] -= need
                    pos["margin"] += need
                    pos["cost"] += need
                    _log(st, "warning", f"MARGIN CALL on {prod['name']}: losses ate into your deposit, so "
                                        f"SGD {need:,.2f} was moved from your cash to top it up.")
                else:
                    st["cash"] += pos["margin"]
                    st["positions"].remove(pos)
                    owed = " Your losses were bigger than your deposit, so you now OWE the broker money (negative cash)." if st["cash"] < 0 else ""
                    _log(st, "warning", f"MARGIN CALL on {prod['name']}: not enough cash to top up, so the position "
                                        f"was force-closed. Total loss SGD {pos['cost'] - pos['margin']:,.2f}.{owed}")
        elif k == "cpn":
            if t >= pos["mat_t"]:
                pay = pos["principal"] * cpn_unit_value(m, m["sg"] / pos["s0"], 0, pos["part"])
                st["cash"] += pay
                st["positions"].remove(pos)
                _log(st, "info", f"{prod['name']} matured. STI moved {m['sg'] / pos['s0'] - 1:+.1%}; with "
                                 f"{pos['part']:.0%} participation you received SGD {pay:,.2f}.")
        elif k == "rc":
            S = px[pos["und"]]
            if not pos["ki"] and S <= pos["barrier"] * pos["s0"]:
                pos["ki"] = True
                _log(st, "warning", f"{prod['name']}: {pos['und'].title()} has fallen {1 - S / pos['s0']:.0%} and hit the "
                                    f"knock-in barrier. If it ends below its starting price you will get shares instead of cash.")
            since = t - pos["issue_t"]
            if since % 3 == 0 and since > 0:
                c = pos["principal"] * pos["coupon"] / 400
                st["cash"] += c
                pos["income"] += c
            if t >= pos["mat_t"]:
                st["positions"].remove(pos)
                if pos["ki"] and S < pos["s0"]:
                    units = pos["principal"] / pos["s0"]
                    sp = next((p for p in st["positions"] if p["pid"] == pos["und"] and p["kind"] == "fund"), None)
                    if sp:
                        sp["units"] += units
                        sp["cost"] += pos["cost"]
                    else:
                        st["positions"].append(dict(id=st["next_id"], pid=pos["und"], kind="fund", units=units,
                                                    cost=pos["cost"], income=0.0, opened_t=t))
                        st["next_id"] += 1
                    _log(st, "warning", f"{prod['name']} matured AFTER a knock-in: instead of SGD {pos['principal']:,.0f} "
                                        f"you received {units:,.0f} {pos['und'].title()} shares worth SGD {units * S:,.2f}.")
                else:
                    st["cash"] += pos["principal"]
                    _log(st, "info", f"{prod['name']} matured: your SGD {pos['principal']:,.2f} was repaid in full "
                                     f"(plus SGD {pos['income']:,.2f} of coupons received).")

    # --- age events
    for e in list(m["events"]):
        e["left"] -= 1
        if e["left"] <= 0:
            m["events"].remove(e)
    st["history"].append(_snapshot(st))


def advance(st, months: int) -> dict:
    """Advance the game by `months` and return a summary of the jump."""
    if months not in (6, 12, 24, 36):
        raise TradeError("You can move forward 6, 12, 24 or 36 months.")
    start_t, start_total = st["t"], total_value(st)
    # value + income - cost, so cash added later (e.g. margin top-ups) is not counted as a gain
    start_vals = {p["id"]: position_value(p, st["market"], st["t"]) + p["income"] - p["cost"] for p in st["positions"]}
    log_start = len(st["log"])
    for _ in range(months):
        step_month(st)
    m = st["market"]
    end_total = total_value(st)
    pos_changes = []
    for p in st["positions"]:
        if p["id"] in start_vals:
            v = position_value(p, m, st["t"]) + p["income"] - p["cost"]
            pos_changes.append(dict(id=p["id"], name=PRODUCTS[p["pid"]]["name"], change=round(v - start_vals[p["id"]], 2)))
    first = st["history"][start_t]
    return dict(
        from_date=month_label(start_t), to_date=month_label(st["t"]), months=months,
        start_total=round(start_total, 2), end_total=round(end_total, 2),
        change=round(end_total - start_total, 2),
        change_pct=round((end_total / start_total - 1) * 100, 2) if start_total > 0 else 0.0,
        sti_change_pct=round((m["sg"] / first["sti"] - 1) * 100, 2),
        rate_from=first["rate"], rate_to=round(m["r"], 2), infl=round(m["infl"], 2),
        events=[e for e in st["log"][log_start:] if e["kind"] in ("event", "warning", "info")],
        positions=pos_changes,
    )


# ---------------------------------------------------------------------------
# View model for the frontend
# ---------------------------------------------------------------------------
def quote(st, prod):
    m, t = st["market"], st["t"]
    k = prod["kind"]
    if k == "fund":
        p = m["px"][prod["id"]]
        q = f"SGD {p:,.4f} per unit" if p < 10 else f"SGD {p:,.2f} per unit"
        if prod["id"] == "MMF_SGD":
            q += f" | yield ~{max(m['r'] - 0.3, 0):.2f}% p.a."
        elif prod["id"] == "MMF_USD":
            q += f" | USD yield ~{usd_rate(m) - 0.3:.2f}% p.a."
        elif prod["id"].startswith("FX_"):
            c = prod["id"][3:]
            q = f"{'100 JPY' if c == 'JPY' else '1 ' + c} = SGD {m['fx'][c]:.4f} | {c} interest {foreign_rate(m, c):.2f}%"
        elif prod["id"] == "ETF_SGBOND":
            q += f" | yield ~{curve(m, 7):.2f}%"
        return q
    if k == "bond":
        return f"New issue at par, coupon {round(bond_yield(m, prod, prod['tenor'] * 12) * 20) / 20:.2f}% p.a."
    if k == "option":
        S = underlying_price(m, prod["und"])
        fake = dict(und=prod["und"], otype=prod["otype"], strike=S * prod["moneyness"], expiry_t=t + prod["tenor"], qty=1.0)
        prem = option_value(fake, m, t) * (1 + prod["sell_cost"])
        unit = "STI" if prod["und"] == "sg" else "share price"
        return f"{unit} {S:,.2f}, strike {S * prod['moneyness']:,.2f}, premium = {prem / S:.1%} of the {unit}"
    if k == "future":
        return f"STI at {m['sg']:,.0f} | 10% margin = 10x leverage"
    if k == "cpn":
        base = cpn_unit_value(m, 1.0, prod["tenor"], 0.0)
        opt = cpn_unit_value(m, 1.0, prod["tenor"], 1.0) - base
        part = max(0.1, min(1.2, (0.97 - base) / opt)) if opt > 0 else 0.1
        return f"100% capital protection, {part:.0%} participation in STI gains"
    if k == "rc":
        return f"{prod['coupon']:.0f}% p.a. coupon, knock-in at {prod['barrier']:.0%}, {prod['und'].title()} at SGD {m['px'][prod['und']]:.2f}"
    return ""


def position_details(pos, st):
    m, t = st["market"], st["t"]
    k = pos["kind"]
    if k == "fund":
        return f"{pos['units']:,.2f} units"
    if k == "bond":
        return f"Face SGD {pos['face']:,.0f}, coupon {pos['coupon']:.2f}%, matures {month_label(pos['mat_t'])}"
    if k == "option":
        S = underlying_price(m, pos["und"])
        return f"Strike {pos['strike']:,.2f} (now {S:,.2f}), expires {month_label(pos['expiry_t'])}"
    if k == "future":
        notional = pos["units"] * m["sg"]
        return f"Controls SGD {notional:,.0f} of STI (entry {pos['entry']:,.0f}, now {m['sg']:,.0f})"
    if k == "cpn":
        return f"STI start {pos['s0']:,.0f} (now {m['sg']:,.0f}), {pos['part']:.0%} participation, matures {month_label(pos['mat_t'])}"
    if k == "rc":
        S = m["px"][pos["und"]]
        status = "KNOCKED IN" if pos["ki"] else "barrier not hit"
        return (f"{pos['coupon']:.0f}% coupon, start SGD {pos['s0']:.2f} (now {S:.2f}), barrier SGD "
                f"{pos['s0'] * pos['barrier']:.2f}, {status}, matures {month_label(pos['mat_t'])}")
    return ""


def view(st):
    m, t = st["market"], st["t"]
    positions = []
    for p in st["positions"]:
        prod = PRODUCTS[p["pid"]]
        v = position_value(p, m, t)
        positions.append(dict(id=p["id"], pid=p["pid"], name=prod["name"], cls=prod["cls"], risk=prod["risk"],
                              value=round(v, 2), cost=round(p["cost"], 2), income=round(p["income"], 2),
                              pnl=round(v + p["income"] - p["cost"], 2),
                              pnl_pct=round((v + p["income"] - p["cost"]) / p["cost"] * 100, 2) if p["cost"] > 0 else 0.0,
                              details=position_details(p, st), sell_cost=prod.get("sell_cost", 0)))
    products = [dict(id=p["id"], cls=p["cls"], name=p["name"], risk=p["risk"], desc=p["desc"], min=p["min"],
                     buy_cost=p.get("buy_cost", 0), sell_cost=p.get("sell_cost", 0), quote=quote(st, p))
                for p in P]
    total = total_value(st)
    return dict(
        date=month_label(t), t=t, cash=round(st["cash"], 2), total=round(total, 2),
        invested_value=round(total - st["cash"], 2), start=START_CASH,
        gain=round(total - START_CASH, 2),
        real_total=round(total * 100 / m["cpi"], 2),
        market=dict(sti=round(m["sg"], 1), rate=round(m["r"], 2), y10=round(curve(m, 10), 2),
                    infl=round(m["infl"], 2), usd=round(m["fx"]["USD"], 4), eur=round(m["fx"]["EUR"], 4),
                    ig=round(m["ig"], 2), hy=round(m["hy"], 2),
                    events=[dict(label=e["label"], headline=e["headline"], left=e["left"]) for e in m["events"]]),
        positions=positions, products=products, classes=CLASSES,
        history=st["history"], log=st["log"][-200:],
    )
