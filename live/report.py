"""Summarize the live paper-trading run vs backtest expectations."""
import json, os, sys
import pandas as pd, numpy as np
D = os.path.join(os.path.dirname(__file__), '..', 'data', 'live', 'weather')


def load(name):
    p = os.path.join(D, name)
    return pd.DataFrame([json.loads(l) for l in open(p)]) if os.path.exists(p) else pd.DataFrame()


F, S = load('fills.jsonl'), load('settled.jsonl')
sig = load('signals.jsonl')
print('signals logged', len(sig), '| events seen', sig.slug.nunique() if len(sig) else 0)
if len(F):
    F['cost'] = F.shares * F.px
    print(f"paper fills {len(F)}  cost ${F.cost.sum():.0f}  avg edge {(F.fair - F.px).mean():.3f}  "
          f"sides {F.side.value_counts().to_dict()}  kinds {F.kind.value_counts().to_dict()}")
if len(S):
    S['cost'] = S.shares * S.px
    print(f"settled {len(S)}  cost ${S.cost.sum():.0f}  pnl ${S.pnl.sum():.1f}  return {S.pnl.sum() / S.cost.sum():.3f}  "
          f"win rate {S.win.mean():.2f}  avg px {S.px.mean():.3f}")
    print(S.groupby('kind').agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum')).round(2))
    print('backtest expectation (walk-forward, edge>=0.15): return +8.5% on turnover, ~68% positive days')
open_ = F[~F.set_index(['cid', 'side', 'ts']).index.isin(S.set_index(['cid', 'side', 'ts']).index)] if len(S) and len(F) else F
if len(open_):
    print(f"open positions {len(open_)}  exposure ${(open_.shares * open_.px).sum():.0f}")

# mark-to-market of open positions at current book mid (early read before settlement)
if len(open_):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from polylib import get, GAMMA, jl
    cids = list(open_.cid.unique()); mids = {}
    for i in range(0, len(cids), 50):
        for m in get(f'{GAMMA}/markets', {'condition_ids': cids[i:i + 50], 'limit': 50}):
            try:
                bb, ba = float(m.get('bestBid') or 0), float(m.get('bestAsk') or 1)
                mids[m['conditionId']] = (bb + ba) / 2
            except Exception:
                pass
    o = open_.copy()
    o['mid_yes'] = o.cid.map(mids)
    o['mark'] = np.where(o.side == 'YES', o.mid_yes, 1 - o.mid_yes)
    o['mtm'] = o.shares * (o['mark'] - o.px) - o.fee
    print(f"open MTM ${o.mtm.sum():.2f} on ${(o.shares * o.px).sum():.0f}  ({o.mtm.sum() / (o.shares * o.px).sum():+.1%})")
    g = o.groupby(['slug', 'bucket', 'side']).agg(cost=('px', lambda s: 0), mtm=('mtm', 'sum')).mtm.sort_values()
    print('worst:', g.head(3).round(2).to_dict()); print('best:', g.tail(3).round(2).to_dict())
