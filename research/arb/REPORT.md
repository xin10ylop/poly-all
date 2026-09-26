# Polymarket executable-arbitrage scan (2026-09-25 23:57 → 2026-09-26 02:09 UTC)

## Bottom line
No riskless, executable (≥5-share minimum), instantly realisable arbitrage turned up in ~89 minutes of full-universe
scanning. The scan covered ~7,400 negRisk events and ~215k live markets, plus a 35-minute poll every 1.5 s of the
events closest to breakeven.

- **Binary YES/NO arbitrage: 0 found. It is structurally impossible.** The CLOB keeps one unified book per market
  (YES ask = 1 − NO bid, level for level), and it mints and merges YES/NO shares automatically.
- **negRisk "NO basket" (Σ YES bids > 1): riskless and instantly realisable (convert + merge), but taker fees kill it.**
  - Before fees it is common: 5,565 event-sweeps across 82 events, typically +0.5%.
  - The median fee drag is 2.2%, which is larger than the edge.
  - Only 4 events were positive after fees, and all were below the 5-share minimum ($0.002–0.06).
  - The 1.5 s poll found **0** executable windows.
- **negRisk "buy all YES" (Σ YES asks < 1):**
  - ~300 flagged events are false positives: their outcome lists aren't exhaustive (placeholder/"Other" outcomes, or a
    missing outcome).
  - ~10 events are genuinely exhaustive, with 0.4–4% edges held to resolution in 2027. Together they total ~$17 on
    ~$1k of capital, i.e. below T-bill yields.
- **Nested ladders (over/under lines, spreads, thresholds, "by date" markets):**
  - Static violations are long-dated and tiny: ~$17 of standing stock in total. The largest is the Clarity Act at
    +0.45%, $11.
  - Live sports produce real but fleeting violations. One MLB over/under pair worth $61 was gone within 36 s. A
    25-minute probe of 62 live games found 5 windows totalling $2.15, each lasting under 10 s.

**Estimated value to a $1–5k bot:**
- ~$0/day from instant arbitrage.
- A one-off ~$35 from hold-to-resolution trades.
- In-play ladder pick-offs might add $10–50/day at best, but need sub-second, two-leg execution. **Not worth running.**

## Coverage
| component | coverage | cadence |
|---|---|---|
| scanner.py (negRisk) | 151 sweeps × ~7,390 groups / 57.5k markets / 115k tokens | ~30 s |
| scanner.py (binary) | 38 sweeps × ~158k markets / 317k tokens | ~2.3 min |
| scanner.py (rechecks) | each new opportunity re-fetched at +5/15/30/60/180 s | — |
| ladders.py | 35 sweeps × 21.7k ladders / 102.6k markets | 2 min |
| hot.py | 126 near-breakeven negRisk events, 35 min | 1.5 s |
| inplay.py | 1,617 ladders in 62 live games, 25 min | 2.4 s |

**Method:**
- Walk the real order-book ladders level by level.
- Charge the per-share taker fee as `rate·p(1−p)`, using each market's own fee rate.
- Require ≥5 shares on every leg.

## Files
- Code: `universe.py`, `scanner.py`, `ladders.py`, `hot.py`, `inplay.py`, `analyze.py`.
- Data: `data/arb/` (gitignored).

**Caveats:**
- An ~11-minute gap after an out-of-memory restart.
- The negRisk adapter conversion fee is assumed to be 0.
- Detection lag is at most one sweep (≤13 s for negRisk, ≤47 s for ladders).
