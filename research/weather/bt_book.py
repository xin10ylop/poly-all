"""Book-based fill check: trade the same model signals against RECORDED order books (data/books/weather, 60s snapshots)
instead of later taker prints. The live bot's policy is 'take the displayed ask when ask <= fair - EDGE'; the print-based
backtest only fills when some other taker traded at such a price, which may select the good moments. This measures the gap.
usage: TEST=data/stack_fwd_test.parquet PRINTS=data/bt_stack_fwd_cap50.pkl python3 research/weather/bt_book.py"""
import os, sys, json, gzip, glob, bisect
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
EDGE = float(os.environ.get('EDGE', 0.15)); DELAY = float(os.environ.get('DELAY', 300)); MAXUSD = float(os.environ.get('MAXUSD', 50))
TOL = 90; FEE = 0.05
te = pd.read_parquet(os.environ.get('TEST', 'data/stack_fwd_test.parquet'))
evs = {}
for l in open('data/weather_events.jsonl'):
    e = json.loads(l)
    for m in e['markets']:
        if m.get('toks'):
            evs[m['cid']] = m['toks'][0]
te = te.assign(tok=te.cid.map(evs)).dropna(subset=['tok'])
need = set(te.tok)
snaps = []   # (ts, {tok: (bids, asks)})
for f in sorted(glob.glob('data/books/weather/*.jsonl.gz')):
    with gzip.open(f, 'rt') as h:
        for l in h:
            try:
                d = json.loads(l)
            except Exception:
                continue
            snaps.append((d['ts'], {k: v for k, v in d['b'].items() if k in need}))
snaps.sort(key=lambda x: x[0]); sts = [s[0] for s in snaps]
print('snapshots', len(snaps), 'span', pd.to_datetime(sts[0], unit='s'), '->', pd.to_datetime(sts[-1], unit='s'))
out = []; covered = []
for (slug, j), g in te.groupby(['slug', 'j'], sort=False):
    spent = {'YES': 0.0, 'NO': 0.0}; tok = g.tok.iloc[0]; Y = g.y.iloc[0]
    for r in g.itertuples():
        k = bisect.bisect_left(sts, r.t + DELAY)
        if k >= len(sts) or sts[k] - (r.t + DELAY) > TOL or tok not in snaps[k][1]:
            continue
        covered.append((slug, j, r.t))
        bids, asks = snaps[k][1][tok]
        for side in ('YES', 'NO'):
            fair = r.p if side == 'YES' else 1 - r.p
            lv = [(px, sz) for px, sz in asks] if side == 'YES' else [(1 - px, sz) for px, sz in sorted(bids, key=lambda x: -x[0])]
            for px, sz in sorted(lv):
                if px > fair - EDGE or spent[side] >= MAXUSD or px < 0.02 or px > 0.98:
                    break
                take = min(sz, (MAXUSD - spent[side]) / px)
                if take < 5:
                    continue
                spent[side] += take * px; win = Y if side == 'YES' else 1 - Y
                out.append((slug, r.city, r.kind, r.date, j, side, r.t, r.hr, fair, px, take,
                            take * (win - px) - FEE * px * (1 - px) * take, win))
B = pd.DataFrame(out, columns=['slug', 'city', 'kind', 'date', 'j', 'side', 'tsig', 'hr', 'p', 'px', 'shares', 'pnl', 'win'])
B['cost'] = B.shares * B.px
cov = pd.DataFrame(covered, columns=['slug', 'j', 'tsig'])
P = pd.read_pickle(os.environ.get('PRINTS', 'data/bt_stack_fwd_cap50.pkl'))
Pc = P.merge(cov.drop_duplicates(), on=['slug', 'j', 'tsig'])
def s(d, name):
    if len(d) == 0:
        print(name, 'no fills'); return
    e = d.groupby('slug').pnl.sum().sort_values()
    print(f"{name:28s} fills {len(d):5d} events {d.slug.nunique():3d} cost ${d.cost.sum():7.0f} pnl ${d.pnl.sum():7.1f} "
          f"ret {d.pnl.sum() / d.cost.sum():+.1%}  ex-top-event {(d.pnl.sum() - e.iloc[-1]) / (d.cost.sum() - d[d.slug == e.index[-1]].cost.sum()):+.1%}  avg px {d.px.mean():.3f}")
print('signal rows with a book snapshot:', len(cov), 'of', len(te))
s(B, 'BOOK (take displayed ask)'); s(Pc, 'PRINTS (same signal rows)')
for side in ('YES', 'NO'):
    s(B[B.side == side], f'  book {side}'); s(Pc[Pc.side == side], f'  prints {side}')
B.to_pickle('data/bt_book.pkl')
