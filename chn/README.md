# Invest Quest SG

A beginner investing simulation game. The player starts with **SGD 50,000** in cash, buys fictional
products from 10 asset classes, and moves time forward (6 months, 1, 2 or 3 years) to see realistic,
randomly simulated results, including market events like crashes, rate hikes and defaults.
An optional AI guide answers questions and explains each time jump when an OpenAI key is set.

## Run it

```bash
cd /workspace/invest-game
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # first time only
export OPENAI_API_KEY=sk-...        # optional: turns on the AI guide (default model gpt-4o-mini)
export OPENAI_MODEL=gpt-4o-mini     # optional
./start_server.sh                   # runs uvicorn in the background on port 8000 (log: server.log)
./start_tunnel.sh                   # optional: Cloudflare quick tunnel, prints a temporary public URL
```

Open http://localhost:8000. Without a key the guide panel shows a "coming soon" message and the
rest of the game works normally.

Deploy anywhere that runs Docker or a Procfile (Render, Railway, Fly.io, Hugging Face Spaces...):
`docker build -t invest-game . && docker run -p 8000:8000 -e OPENAI_API_KEY=... invest-game`.

## Files

| File | What it does |
|---|---|
| `sim.py` | Simulation engine: market factors, events, products, pricing, trades, monthly steps |
| `app.py` | FastAPI server and JSON API. It's stateless: the browser keeps the game in `localStorage` |
| `guide.py` | AI guide (OpenAI Chat Completions via `requests`) with portfolio and event context |
| `static/` | Mobile-first frontend (plain HTML/CSS/JS, hand-drawn canvas chart, no build step) |
| `tests/test_api.py` | End-to-end API test: 25 games x 8 random jumps across all products (NaN, negative cash, sell-all) |
| `tests/check_products.py` | Outcome distribution of each contract-type product held to maturity |
| `tests/calibrate.py` | Monte Carlo of realised return and volatility per product |

API: `POST /api/new`, `/api/view`, `/api/trade` (`buy` with product_id + amount, or `sell` with position_id + optional amount),
`/api/advance` (`months` = 6/12/24/36), `/api/chat`, `/api/explain`, `GET /api/health`.

## Simulation model (monthly steps)

**Correlated factors** (Cholesky of a correlation matrix): SG equities, global equities, SGD short rate,
long rate, IG and HY credit spreads, USD/EUR/JPY/AUD vs SGD, and inflation. Equities are about 0.75 correlated.
Spreads widen when stocks fall. USD and JPY rise and AUD falls in risk-off. Rates, spreads and inflation mean-revert
(short rate 2.5%, long rate 3.0%, IG 1.0%, HY 4.0%, inflation 2.5%). Yield curve = blend of short and long rate.

| Class | Products | Expected return / volatility (approx., p.a.) |
|---|---|---|
| Money market funds | SGD MMF, USD MMF | SGD rate - 0.3% (~2.2%), vol ~0.3% / USD rate - 0.3% plus USD/SGD moves (vol ~5%) |
| Government bonds | SGS 2Y, 10Y, 30Y (new issue at par, coupon = current yield) | 2.5-3% coupons; 1-yr price vol ~1% / ~5% / ~12%; semi-annual coupons, real bond pricing |
| Corporate bonds | A-rated 5Y (spread ~1%), high-yield 4Y (spread ~4%) | IG ~3.8% coupon, rare default; HY ~6.8% coupon, ~3-4%/yr default risk (higher in recessions/credit events), 40% recovery |
| Stocks | Merlion Bank (blue chip), Harbour Telecom (defensive), Orchid Tech (growth) | ~8.5% / 22%, ~6.5% / 15%, ~12% / 44%; dividends 5%, 6%, 0% paid quarterly; tech is hurt by rising yields |
| Mutual funds | Global equity (active), 50/50 balanced | 3% sales charge plus 1.75% / 1.5% annual fee; ~6% / 17% and ~4% / 8.5% after fees |
| ETFs | STI ETF, Global World ETF (USD), SG Govt Bond ETF | ~7% / 16.5% (incl. 3.5% dividend), ~7.5% / 16.5%, ~2.8% / 4%; fees 0.2-0.3% |
| Structured notes | 3Y capital-protected note on STI; 1Y reverse convertibles on Merlion (8%, 70% knock-in) and Orchid (16%, 65% knock-in) | CPN participation is set at issue from rates/vol (~60-65%); RC coupons quarterly, knock-in checked monthly, physical delivery of shares |
| Options | STI 6M call (+5% strike), STI 6M put (-5%), Orchid 1Y ATM call | Black-Scholes with 1.2x implied vol premium and 3% bid/ask; ~50-75% expire worthless |
| Futures | STI long / short, 10x leverage | Margin = deposit; margin call when it falls below 5% of exposure: auto top-up from cash, or forced close (cash can go negative) |
| FX | USD, EUR, JPY, AUD deposits | Random walk, vol 5% / 7% / 10% / 10%, plus foreign interest; 0.25% spread each way |

Idle cash earns 0.05% p.a. "After inflation" value is shown using the simulated CPI.

**Market events** (chance per month; each one hits assets every month while it lasts):

| Event | Prob/month | Lasts | Main effects |
|---|---|---|---|
| Recession / market crash | 1.0% | 3-6 mo | Stocks about -5.5%/mo, rate cuts, govt bonds rally, spreads blow out, USD/JPY up, AUD down, vol x1.8, more defaults |
| Central bank rate hike | 1.8% | 3 mo | +0.75% short rate, bonds down, MMF yields up, growth stocks dip |
| Central bank rate cut | 1.8% | 3 mo | -0.75% short rate, bonds up, stocks up |
| Inflation spike | 0.8% | 6 mo | Inflation up ~3%, yields up, stocks and bonds fall together, SGD strengthens |
| Bull run | 1.2% | 6-9 mo | Stocks +2-2.5%/mo, spreads tighten, low vol |
| Currency shock | 1.0% | 2 mo | One of: USD surge, SGD jump, euro slide, yen surge, AUD plunge |
| Credit event / default | 0.8% | 2 mo | HY spreads +1.6%, stocks wobble, flight to govt bonds, 25% chance the HY issuer defaults |

Base equity drifts are offset so that, including events, long-run returns match the targets above
(checked with `tests/calibrate.py`).

## Limitations

- Fictional, simplified products. This is educational, not financial advice. Monthly steps mean no intra-month moves,
  so margin calls and barriers are checked once a month.
- Product prices are shared across the market (one issuer per corporate bond; after a default, new bonds of a "similar issuer" stay available).
- Game state lives in the browser's localStorage (per device). Clearing browser data resets the game.
- Quick-tunnel URLs are temporary and change each time the tunnel restarts.
