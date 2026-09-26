# Polymarket crypto price-threshold markets vs options-implied fair value

*Data: resolved markets 2026-07-20 → 2026-09-25. Live snapshots 2026-09-25 23:52 → 2026-09-26 01:01 UTC (15 snapshots, ~1,000 markets each).*

## Verdict: no exploitable small-capital edge
- **Liquid daily ladders track Deribit closely.** Liquid daily ladders ("above K at 12:00 ET" and range buckets) sit within 1–1.5 pp of the Deribit-smile fair value. That is inside the spread plus the taker fee (1.5–3.5c).
- **The market is better calibrated than every model.** Out of sample, Polymarket beats a point-in-time Deribit-smile model: Brier×100 5.19 vs 5.26. In a logistic stack, Deribit gets a weight of 0.02.
- **Model-driven taker rules lose.** They lose 0–3.5c per share after fees. The only positive setting (12c threshold, +1.6c) has t = 0.8 and comes from trend exposure.
- **Makers are adversely selected.** Across all 1.12M prints ($129.7M), makers made −0.58% gross. A model-anchored maker loses 1.2–10c per share, even when using Binance spot from 1 second before each fill. The flow is better informed, not merely faster.
- **Large live gaps exist only in long-dated barrier markets.** These are weekly, monthly and yearly "hit" markets, plus the 2027 "when will X hit" markets. The gaps are structural and could not be validated historically: Deribit overstates touch probabilities.

## Method summary
- **Universe:** ~1,000 BTC/ETH/SOL/XRP threshold markets:
  - above_daily and range_daily resolve on the Binance 1m close at 12:00 ET.
  - above_hourly resolves on the Binance 1h candle.
  - The hit markets resolve on 1m High/Low touches.
  - Resolution data comes from Binance's public mirror `data-api.binance.vision`.
- **Point-in-time Deribit smiles:** rebuilt from 777k historical option trades pulled from `history.deribit.com`. They match the live surface within 1–2 vol points.
- **Liquidity filter:** only liquid rows are used (a print within 2h, and the mid within 4c of that print). Illiquid history mids are unreliable.

| forecast (Brier ×100, n = 14,461) | score |
|---|---|
| Polymarket | **5.19** |
| Deribit ATM lognormal | 5.24 |
| Deribit smile | 5.26 |
| RV-empirical | 5.34 |

| taker rule (Deribit smile) | n | c/share | t |
|---|---|---|---|
| gap > 2c | 3175 | −1.42 | −1.4 |
| gap > 5c | 1369 | −1.15 | −0.7 |
| gap > 12c | 505 | +1.62 | +0.8 |

**No-arbitrage checks:**
- Range vs synthetic ladder: 45 crossings in 3,300 checks, all ≤ 1.9c gross, below the ~3.5c of fees.
- Touch vs digital violations: 0 in 2,535 checks.

**Longshots:** buying NO on daily-hit YES at 3c or less earns +0.19c per share. Real but tiny.

## Caveats
- **Short window:** 68 days in a strong uptrend (BTC +30%).
- **Few long-dated events:** only 40 weekly and 8 monthly hit events.
- **Fill assumption:** trade-through fills with no queue modelling.

Code is in this folder: `clib.py`, `02_snapshot.py`…`16_latency_test.py`. Data and logs are in `data/crypto/`.
