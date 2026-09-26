# Polymarket edge hunt: findings (2026-09-26)

**Bottom line.** The best edge found that is robust, executable and not speed-dependent is **weather temperature
nowcasting**. A model combines the market's last traded price with a nowcast of the day's max/min built from live airport
METAR observations. It trades only visible liquidity (taker, fill-and-kill at the displayed ask), minutes to hours after
each observation.

Out-of-sample walk-forward results (trained through Jul 31, tested Aug 1 – Sep 25, 56 days), filling only against real
trade prints that occurred ≥5 minutes after each signal:

| metric | value |
|---|---|
| return on turnover (after taker fees) | **+8.5%** |
| P&L per day | **$227** (std $434) |
| positive days | 68% |
| daily Sharpe (annualised) | 10.0 |
| max drawdown | $545 |
| capital in use | mean $1.6k, p95 $2.4k, max $3.2k |
| fills | 70,751 (3,296 events) |

It is profitable in every slice: YES and NO sides, highs (+7.3%) and lows (+28%), NOAA and Wunderground cities, and
every hour bucket. The stress test uses 15-minute delays and only 25% of each print. It still returns +7.7% on turnover
and $140/day.

**Capacity is the limit, not skill.** Raising the per-bucket cap from $50 to $200 only lifts P&L from $228 to $265 a day.
This is a $150–260/day strategy on ~$2–4k of capital. It starts small and grows only by adding market families.

---

## 1. What was tested (and the honest verdicts)

| # | idea | data / method | verdict |
|---|---|---|---|
| 1 | **Weather stacked nowcast (taker)** | 77k weather markets, 4.4M taker prints, 3 yrs METAR, GBM nowcast + stacked model, print-based fills | **EDGE**: +8.5% OOS, Sharpe 10 (see above) |
| 2 | Weather "dead-bucket" NO sniping | reverse-engineered wallet `0x6011655c` (34/34 green days, ~$974/day) | real, but a **1–3 minute race** vs 3–4 bots → excluded per requirement |
| 3 | Weather day-ahead from NOAA NBM forecasts | IEM NBM (txn/xnd) archive, 12 N. American stations | **no edge**: market Brier 0.057 vs NBM 0.076; blend worse than market |
| 4 | Pure METAR nowcast vs market | same | weak (+2–3%); market knows forecast; edge only when combined (→ #1) |
| 5 | Liquidity-rewards farming (weather pool $16k/day) | recorded 1-min books, Polymarket scoring formula | rewards are real (wallets earn $80–250/day), but naive tight quoting loses more to adverse selection than it earns; fills can't be tested exactly → secondary |
| 6 | Elon Musk post-count brackets ($1.4B traded) | full xtracker post history (resolution source; 200/202 outcomes reproduced), NB nowcast | **no edge**: market beats the model at every horizon |
| 7 | "Buy NO on longshots" (favourite–longshot bias) | 134k closed events, 12h histories | illusion on midpoints; **vanishes in liquid markets** (vol > $50k) |
| 8 | UMA "proposed" resolution lag | live scan | only illiquid props; no size |
| 9 | Earthquake weekly counts (USGS) | live check | occasional mispricing; tiny markets ($10–30k) → future add-on |
| 10 | NegRisk / binary / nested-ladder arbitrage | 89-min full-universe scan (~215k markets), 1.5 s hot poll, in-play probe | **no executable arb**: binary impossible (unified book); negRisk baskets fee-killed (~2% drag); leftovers are dust or sub-T-bill yield; in-play windows last <10 s (`research/arb/REPORT.md`) |
| 11 | Crypto thresholds vs options-implied | point-in-time Deribit smiles from 777k option trades, 1.12M Polymarket prints | **no edge**: market Brier 5.19 vs Deribit 5.26; taker rules −0 to −3.5c/share; makers adversely selected (`research/crypto/REPORT.md`) |
| 13 | Weather stacked model as **maker** (resting bids) | same data, fill only when a real seller trades through | **strongly negative** (−18% to −24%): resting orders get picked off by informed sellers → taker-only |
| 12 | Jev structural tags → mispriced segments | 31k resolved markets tagged by Jev (11 questions, $0.79), executable VWAP prices, event-clustered stats, permutation nulls, leakage probe | **no robust edge**: tags find YES ~1–2¢ rich, less than spread+fee; decays over 2026 (`research/jev_tags/REPORT.md`) |

Leaderboard study (`research/wallets/REPORT.md`): the consistent small wallets trade many small tickets in niche
data-driven markets. The two families that small traders can replicate without speed are Asian-hours weather nowcasting
(windy-weather, opopv3: +11–16% on turnover) and niche data trackers. That study pointed to #1.

## 2. The strategy (#1) in detail

**Ground truth.** Since late August most cities resolve on the NOAA weather.gov time series of the airport METAR. The
local-calendar-day max/min computed from METARs reproduces the official outcome **~100%** of the time for NOAA cities.
For Wunderground cities it is about 96%. Hong Kong resolves on HKO data, so it is excluded.

**Nowcast model** (`research/weather/nowcast.py`, `train_nowcast.py`):
- Trained on 3 years of METAR (2023-06 → 2026-06) for 52 stations.
- Predicts D = final extreme − running extreme, using hour, season, gap to the running extreme, 1h/3h trends and
  station.
- It is only used when P(D ≥ 6) < 1%, i.e. when the observations are informative.

**Stacked model** (`research/weather/stack.py`). A GBM combines the nowcast bucket probability with market features:
last print price, print age, prints in the last hour, event overround, normalised price, bucket distance to the running
extreme, and so on. Out of sample it beats the market's last print on Brier score (0.0300 vs 0.0316), by 20–37% late in
the local day.

**Trading rule** (`research/weather/bt_stack.py`, live: `live/weather_bot.py`):
- Buy YES if p − ask ≥ 0.15; buy NO if (1−p) − ask_NO ≥ 0.15. Only prices in [0.02, 0.98].
- Cap: $50 per bucket per side. Taker fee: 0.05·p·(1−p) per share.
- Hold to resolution.
- Execution is fill-and-kill at the displayed ask, so orders never rest.

**Backtest execution realism:**
- Fills are only counted against actual taker prints in a 30-minute window starting 5 minutes after the signal (the
  signal already includes 5 minutes of METAR latency).
- We take at most 50% of each print's size (25% in the stress test).
- Dropping orders under the 5-share minimum changes P&L by under 2%.

### Robustness table (test period Aug 15 – Sep 25 unless noted)

| variant | return on turnover | $/day | positive days | Sharpe | max DD |
|---|---|---|---|---|---|
| edge ≥ 0.05 | 3.5% | 287 | 69% | 10.4 | 1284 |
| edge ≥ 0.10 | 4.7% | 202 | 62% | 8.9 | 1130 |
| **edge ≥ 0.15** | **9.1%** | **228** | **71%** | **11.0** | **563** |
| edge ≥ 0.20 | 12.5% | 194 | 74% | 10.4 | 552 |
| edge ≥ 0.15, 15-min delay, 25% of prints | 7.7% | 140 | 64% | 8.9 | 384 |
| edge ≥ 0.15, $200 cap | 7.9% | 265 | 64% | 8.5 | 1341 |
| **walk-forward (train ≤ Jul 31, test 56 d)** | **8.5%** | **227** | **68%** | **10.0** | **545** |

Both halves of the test period are positive for both kinds:
- highs: +9.5% / +4.6%
- lows: +30% / +27%


### Rolling retrain (production cadence: retrain every ~2 weeks on all data to date, trade the next window)

| train through | trade window | return on turnover | $/day | positive days | Brier market → model |
|---|---|---|---|---|---|
| Aug 14 | Aug 15 – Aug 31 | +8.5% | 234 | 53% | 0.0307 → 0.0297 |
| Aug 31 | Sep 1 – Sep 14 | **+10.6%** | 238 | **93%** | 0.0307 → 0.0269 |
| Sep 14 | Sep 15 – Sep 25 | +6.0% | 122 | 73% | 0.0335 → 0.0287 |
| **all folds** | **42 days** | **+8.6%** on $101k | **~206** | | model's Brier advantage grows with training data |

Weekly P&L in the Aug 1 – Sep 25 walk-forward was positive in 9 of 9 weeks.

The early-morning trades are also profitable (+9.8%), when the nowcast is uninformative (P(D≥6) ≥ 1%). There the model
exploits the market's bucket-pricing structure: stale last prints, overround, and edge buckets. The live bot therefore
trades all hours, exactly like the backtest. It uses a production model retrained on all data through Sep 25
(`research/weather/train_prod.py`).

**Day-ahead add-on** (`research/weather/stack_prev.py`). Trades buckets in the 30h before the observation day, using
market structure plus yesterday's observed extreme. Out of sample it returned +8.0% on turnover, $65/day, 66% positive days (positive in
both halves). Its overall Brier does not beat the market (0.0620 vs 0.0613); only large disagreements pay. It is a candidate
for about 30% more capacity and is not yet in the live bot.

**Maker variant is negative.** Resting bids at fair − edge get filled only by better-informed sellers, for −18% to −24%. Taker-only is correct.


### Live paper-trading log
- **2026-09-26 05:20 UTC, bug found and fixed.** The aviationweather.gov METAR API returns at most 400 observations per
  request. The first live bot fetched 40 stations × 40h in one call, so it only saw the last ~5 hours per station. That
  made its running daily extremes wrong: for example, the NYC low read 68°F when the true low was 58°F. Its first 39
  paper fills (00:57–05:15 UTC) are therefore **invalid** and archived in `data/live/weather/v1_buggy/`. Several were
  losing NO bets against the correct bucket.
- The fix:
  - Each station is backfilled individually, then polled in small batches.
  - A coverage guard skips any event whose observations don't cover the local day so far.
  - Independent verification on 12 random events: 0 mismatches.
- The backtests are **not** affected; they use the complete IEM archive.
- **Other parity changes made so live matches the validated regime:**
  - Trade all hours, matching the backtest (no nowcast-informativeness gate).
  - Only trade markets with a print in the last 60 minutes; 96% of backtest fills were in that regime.
  - Use a production model retrained through Sep 25.
- **Book vs print fills.** For live fills where later prints existed, the prints were on average about 5¢ cheaper than
  the displayed ask the bot paid. The print-based backtest does not flatter fill prices; if anything it's the other way.
- **Infrastructure caveat.** This cloud container is reclaimed when the session is idle, which stops the bot.
  `live/ensure_running.sh` restarts it; for continuous operation run the bot on your own always-on machine or VPS.


### Minute-level add-on: dead-bucket sniping (re-tested after clarifying that minute-level reaction is allowed)

**Mechanism.** When a new METAR pushes the day's running max above a bucket (or the running min below it), that bucket
can no longer win. We buy NO at the displayed price. Backtest (`research/weather/bt_dead.py`):
- Fills only against real NO-buy prints that happened at least L minutes after the observation time.
- 50% share of each print, $200 cap per bucket.
- Actual outcomes, so any METAR/resolution mismatch is paid for.

| reaction L after obs time | $/day (NOAA cities, Aug 24 – Sep 25) | positive days | return on turnover |
|---|---|---|---|
| 3 min | ~$68 | 97% | +3.0% |
| 5 min | ~$26 | 97% | +2.4% |
| 7 min | ~$20 | 97% | +2.7% |
| 10 min | ~$7 | 97% | +1.3% |

**Feed latency.** Median delay from observation time to availability, measured live:

| source | median delay |
|---|---|
| NOAA tgftp | 4.0 min |
| aviationweather API | 5.1 min |
| aviationweather cache | 5.1 min |
| IEM | 5.5 min |

A tgftp poller realistically reacts in about 4–5 minutes, so expect **~$20–30/day**. The edge is nearly riskless and
separate from the main strategy, whose 15-point edge threshold never fires at 95–99¢. It runs as a paper module in
`live/sniper.py`.

**Speed does not help the main strategy.** Re-tested on a 10-minute grid with 1–5 minute delays (walk-forward Aug 1 –
Sep 25) it made $222–225/day, the same as the 30-minute grid ($227). The binding constraint is liquidity, not latency.

**Combined backtest expectation:** main strategy ~$200–230/day plus sniping ~$20–30/day, i.e. **~$220–260/day** on
~$3–5k capital. Conservative (stress-test) figure: ~$160/day.

## 3. Jev + LLM

- **Jev as a rules guard** (`live/rules_guard.py`). Before trading an event, Jev answers typed questions about its rules
  text: source, statistic, whole-day window, unit, and whether there is a non-standard clause. Answers must agree with
  the regex parser.
  - Today it approved 188/192 events and correctly rejected Hong Kong (HKO source).
  - The standard boilerplate scores a stable ~0.83 on "unusual clause", so a jump above that baseline flags a rules change.
  - Disagreements escalate once to an LLM (Claude via OpenRouter); this is the verified-cascade pattern.
  - Cost is about $0.00005 per event per day.
- **Jev as a forecaster.** Per TypeSafe's docs and independent tests, Jev is a calibrated classifier, not a world-knowledge
  forecaster. The structural-tag study (subagent) measures whether Jev tags isolate mispriced segments using executable
  prices.
- **The LLM (Opus)** did the research, strategy design and model building. In production it is only an escalation path,
  because the OpenRouter key has about $3 left.

## 4. How to run

```
python live/book_recorder.py weather 60     # 1-minute order-book recorder
python live/weather_bot.py                  # paper trading: fills only at displayed ask/depth, settles at resolution
# logs: data/live/weather/{signals,fills,settled}.jsonl, bot.log
```

Re-create the data and models:
- `research/weather/fetch_*.py`
- `nowcast.py high|low`
- `train_nowcast.py high|low`
- `stack.py build`
- `bt_stack.py`

**Live money.** Order placement is not implemented in this repo. The bot is paper-only. Going live means adding
fill-and-kill BUY orders through Polymarket's official `py-clob-client-v2`, funded by a Polymarket wallet. That is a
decision for you.

## 5. Risks

- **Resolution-source changes.** A city can switch stations or sources (for example Wunderground to NOAA). The Jev guard
  plus per-city validation catch this. Keep HKO excluded.
- **Competition.** Other nowcasters may shrink the edge. Monitor live fills against the backtest.
- **Model drift and seasonality.** Retrain the stacked model monthly (walk-forward).
- **Capacity.** About $200–260/day with current liquidity. Scaling means more market families: lows, precipitation,
  earthquakes, other data-driven niches.
