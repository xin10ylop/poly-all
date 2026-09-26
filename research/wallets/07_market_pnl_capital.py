"""Step 4d: per-market PnL reconstruction + capital-in-use estimate from fills.

Per (wallet, market) within the fetched window:
  cash = sells - buys (+ MERGE usdc - SPLIT usdc from activity)
  + net shares held x resolution payout (if resolved) ; unresolved -> excluded from PnL stats
Only markets whose first fill is >= 1h after the window start are used (avoid positions
opened before the window). Capital in use = sum over open markets of net cost, where a
market is 'open' from first fill until max(last fill, closedTime) (unknown close -> last
fill + 24h); we report median / p90 of that time series.

Output: data/wallets/market_pnl.csv, data/wallets/capital.csv
"""
import json
import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wallets")

A = pd.read_csv(os.path.join(D, "trades_all.csv"))
ACT = pd.read_csv(os.path.join(D, "activity.csv"))
M = pd.read_csv(os.path.join(D, "markets_meta.csv"))
P = pd.read_csv(os.path.join(D, "wallet_profiles.csv"))
names = P.set_index("wallet")["name"].to_dict()

A = A.drop_duplicates(["wallet", "transactionHash", "asset", "side", "price", "size", "timestamp"])
A["usdc"] = A.price * A["size"]
A["sgn"] = np.where(A.side == "BUY", 1, -1)


def payouts(r):
    try:
        o = json.loads(r.outcomes)
        pr = [float(x) for x in json.loads(r.outcomePrices)]
    except Exception:  # noqa: BLE001
        return None
    if str(r.closed).lower() != "true" or max(pr) < 0.99:
        return None
    return dict(zip(o, pr))


M["pay"] = M.apply(payouts, axis=1)
pay = M.set_index("conditionId").pay.to_dict()
close = pd.to_datetime(M.set_index("conditionId").closedTime, utc=True, errors="coerce",
                       format="mixed").dt.tz_localize(None)
close_ts = (close.astype("int64") // 10**9).where(close.notna()).to_dict()

mg = ACT[ACT.type.isin(["MERGE", "SPLIT"])].copy()
mg["cash"] = np.where(mg.type == "MERGE", 1, -1) * mg.usdcSize.fillna(0)
mg_cash = mg.groupby(["wallet", "conditionId"]).cash.sum().to_dict()

rows, cap_rows = [], []
for w, g in A.groupby("wallet"):
    t0 = g.timestamp.min()
    for cid, gm in g.groupby("conditionId"):
        first, last = gm.timestamp.min(), gm.timestamp.max()
        buys = gm[gm.side == "BUY"].usdc.sum()
        sells = gm[gm.side == "SELL"].usdc.sum()
        net_sh = (gm.sgn * gm["size"]).groupby(gm.outcome).sum()
        cash = sells - buys + mg_cash.get((w, cid), 0.0)
        pz = pay.get(cid)
        clean = first >= t0 + 3600
        resolved = pz is not None
        # merges remove equal YES/NO shares: approximate by removing min of holdings
        if (w, cid) in mg_cash and len(net_sh) == 2:
            mrg = mg[(mg.wallet == w) & (mg.conditionId == cid)]
            mshares = (np.where(mrg.type == "MERGE", 1, -1) * mrg["size"].fillna(0)).sum()
            net_sh = net_sh - mshares
        pnl = cash + sum(max(s, 0) * (pz or {}).get(o, 0) for o, s in net_sh.items()) if resolved else np.nan
        rows.append({"wallet": w, "name": names.get(w), "conditionId": cid, "slug": gm.slug.iloc[0],
                     "first": first, "last": last, "buys": buys, "sells": sells, "cash": cash,
                     "min_net_sh": net_sh.min(), "resolved": resolved, "clean": clean, "pnl": pnl})
        end = max(last, close_ts.get(cid) or (last + 86400))
        cap_rows.append((w, first, end, max(buys - sells, 0)))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(D, "market_pnl.csv"), index=False)

# capital in use
C = pd.DataFrame(cap_rows, columns=["wallet", "start", "end", "cost"])
caps = []
for w, g in C.groupby("wallet"):
    ev = pd.concat([pd.Series(g.cost.values, index=g.start.values),
                    pd.Series(-g.cost.values, index=g.end.values)]).groupby(level=0).sum().sort_index()
    lvl = ev.cumsum()
    t_lo, t_hi = A[A.wallet == w].timestamp.min(), A[A.wallet == w].timestamp.max()
    # sample hourly across the fill window, ignoring first 10% (positions opened pre-window)
    grid = np.linspace(t_lo + 0.1 * (t_hi - t_lo), t_hi, 200)
    idx = np.searchsorted(lvl.index.values, grid, side="right") - 1
    vals = np.where(idx >= 0, lvl.values[np.clip(idx, 0, None)], 0)
    caps.append({"wallet": w, "name": names.get(w), "cap_med": np.median(vals), "cap_p90": np.percentile(vals, 90)})
CAP = pd.DataFrame(caps)

S = R[R.clean & R.resolved & (R.min_net_sh > -1)].groupby("wallet").agg(
    n_mkts=("pnl", "size"), win_rate=("pnl", lambda x: (x > 0).mean()), pnl_sum=("pnl", "sum"),
    pnl_med=("pnl", "median"), turnover=("buys", "sum"),
    worst=("pnl", "min"), best=("pnl", "max"))
S["ret_on_turnover"] = S.pnl_sum / S.turnover
S["best_share"] = S.best / S.pnl_sum.where(S.pnl_sum > 0)
out = CAP.merge(S, left_on="wallet", right_index=True, how="left")
out.to_csv(os.path.join(D, "capital.csv"), index=False)
pd.set_option("display.width", 250)
print(out.round(3).to_string())
