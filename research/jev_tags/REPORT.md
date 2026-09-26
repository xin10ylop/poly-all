# Jev structural tags vs executable Polymarket prices

**Verdict: I found no robust, executable, out-of-sample edge.**

- **What the tags do find:** Jev tags identify segments where YES is persistently slightly overpriced. The gap is about 1–2¢/share at the trade-implied mid, and up to ~4.7¢ in one segment.
- **Why it doesn't pay:** that is less than a taker's cost of a 1–2¢ half-spread plus ~0.6¢ fee.
- **Validation:** no tag, tag pair or tag-based model beats costs in validation.
- **Decay:** the effect weakens over time (May–Jun 2026 > Jul–Sep 2026).
- **Cost:** total Jev spend was **$0.787** for 15,107 calls.

## Setup
- **Universe:** 31,071 resolved 0/1 binary markets in 7,498 events, with vol ≥ $5k.
  - Excluded: sports, games, crypto-prices, recurring, weather, and asset-price markets.
- **Executable prices:** for horizons hd ∈ {1, 3, 7} days, the VWAP of taker prints in the 12 hours after the horizon, taken per side.
  - Fee: rate·p(1−p).
  - 13,663 markets (43,766 side-entries) have an executable price in [0.03, 0.97].
- **Jev tagging:** each market was tagged once, using only the question, title, rules excerpt, creation/end dates and day count.
  - 11 structural questions: YES needs a new event, involves a specific person, scheduled, numeric, mention, entertainment, ambiguous wording, conflict, government, plus drama and hype scores.
  - No outcome information was given to Jev.
- **Time split:** discovery = markets ending before 2026-05-01; validation = markets ending after.
  - Statistics are event-clustered.

## Leakage check
Jev was asked directly for outcomes on 1,430 markets. This was a diagnostic only, never used as a signal.

| Predictor | AUC |
|---|---|
| Jev | ≈ 0.50–0.67 |
| Market price | 0.73–0.87 |

Jev does not appear to know the outcomes.

## Results
- **Baseline (no tags):**
  - Buying YES loses in every price bucket.
  - Buying NO is flat to negative, e.g. −1.8% (discovery) and −3.5% (validation) at 85–97¢.
  - The longshot-NO pattern seen on mid prices does **not** survive executable prices.
- **Cell scan:** 4,434 cells (single tags and pairs × side × price bucket) were tested.
  - 11 tag cells passed discovery (t ≥ 3), against 5.7 ± 3 expected by chance (permutation null).
  - Only 1 cell validated, and it is fragile.
- **Logistic model:** logit(price) + tags + interactions.
  - With VWAP prices, the tags add nothing (log-loss 0.4560 vs 0.4559).
  - With first-print prices, the NO-side model returned +19%/$ (t = 2.5). But it decays month by month (May +40% → Sep −6%) and is concentrated in Elon-tweet and Trump-"say" series.
- **Segment "sq_no & drama_hi"** (YES needs a new event, and the event would be dramatic):
  - The edge at the mid is +4.7¢ in discovery, +2.4¢ in validation, and 0.0¢ in the late period.
  - It is not significant after execution costs.
- **False positive caught:** "illiquid" originally used lifetime volume, which is ex-post information. After switching to entry-window volume, the effect disappears.

## Takeaway
Jev is a calibrated **classifier**, not a forecaster. In this project it earns its place as a **rules guard** (`live/rules_guard.py`): it verifies resolution source, statistic, window and unit before each weather trade. It is not useful as a source of price signals.

Code: `01_universe` … `08_deepdive`, `jevq.py`, `stats.py`. Data: `data/jev_tags/`.
