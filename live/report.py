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
