"""Executable-entry panel: for each (market, hd) the VWAP of taker prints in the 12h after the horizon,
separately for buying YES and buying NO, plus fees and realized PnL at resolution.
Output: data/jev_tags/panel.pkl (one row per market x hd x side)."""
import os, pickle
import numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
U = pd.read_pickle(os.path.join(ROOT, 'data', 'jev_tags', 'universe.pkl')).set_index('mid')
W = pickle.load(open(os.path.join(ROOT, 'data', 'jev_tags', 'win_trades.pkl'), 'rb'))

rows = []
for (mid, hd), pr in W.items():
    if not pr or mid not in U.index:
        continue
    r = U.loc[mid]
    wvol = sum(p * sz for _, _, _, p, sz in pr)     # all taker $ in the 12h window (no ex-post info)
    buy = {'YES': [], 'NO': []}
    for ts, side, oi, p, sz in pr:
        if sz <= 0 or not (0 < p < 1):
            continue
        if (side == 'BUY' and oi == 0) or (side == 'SELL' and oi == 1):
            buy['YES'].append((ts, p if oi == 0 else 1 - p, sz))
        elif (side == 'BUY' and oi == 1) or (side == 'SELL' and oi == 0):
            buy['NO'].append((ts, p if oi == 1 else 1 - p, sz))
    for s, L in buy.items():
        if not L:
            continue
        L.sort()
        a = np.array([(x[1], x[2]) for x in L])
        vwap = float((a[:, 0] * a[:, 1]).sum() / a[:, 1].sum())
        rate = r.fee_rate if r.fee_rate == r.fee_rate and r.fee_rate is not None else None
        exp = r.fee_exp if r.fee_exp == r.fee_exp and r.fee_exp else 1
        win = int(r.y == 1) if s == 'YES' else int(r.y == 0)
        big = [x for x in L if x[1] * x[2] >= 10]          # first print with >= $10 notional (skip dust)
        rows.append(dict(mid=mid, eid=r.eid, hd=hd, side=s, px=vwap, px_first=big[0][1] if big else np.nan,
                         t_first=(big[0][0] - (r.end - hd * 86400)) / 3600 if big else np.nan,
                         n_prints=len(L), usd=float((a[:, 0] * a[:, 1]).sum()), wvol=wvol,
                         fee_c=(0.04 if rate is None else rate) * (vwap * (1 - vwap)) ** exp,   # conservative default
                         fee_a=(0.0 if rate is None else rate) * (vwap * (1 - vwap)) ** exp,    # listed fee only
                         win=win, end=r.end, vol=r.vol, negRisk=r.negRisk, n_ev_mkts=r.n_ev_mkts))
P = pd.DataFrame(rows)
P['cost'] = P.px + P.fee_c
P['pnl'] = P.win - P.cost            # per share ($1 face)
P['ret'] = P.pnl / P.cost            # per $ staked
P['cost_a'] = P.px + P.fee_a
P['ret_a'] = (P.win - P.cost_a) / P.cost_a
P.to_pickle(os.path.join(ROOT, 'data', 'jev_tags', 'panel.pkl'))
print(len(P), P.mid.nunique(), P.eid.nunique())
print(P.groupby(['hd', 'side']).size())
