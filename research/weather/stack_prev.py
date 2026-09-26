"""Day-ahead stacked model: trade temperature buckets during the 30h BEFORE the observation day starts.
Features: market structure (last print, age, activity, overround, normalized price, bucket position) + persistence
anchor (most recent completed local day's extreme vs bucket bounds) + calendar. Print-based fills as bt_stack."""
import sys, os, json, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(__file__))
from wlib import load_events, load_wtrades, load_metar

SPLIT = os.environ.get('SPLIT', '2026-08-01')
EDGE = float(os.environ.get('EDGE', 0.15)); DELAY = 300; WIN = 1800; SHARE = 0.5; MAXUSD = 50; FEE = 0.05
SF = ['hb', 'ref', 'age', 'n1h', 'over', 'refn', 'pos', 'edgeb', 'nb', 'dlo_p', 'dhi_p', 'kindH', 'unitF', 'rank']


def build():
    evs = [e for e in load_events() if e['date'].isoformat() >= '2026-06-20']
    src = json.load(open('data/weather_src.json'))
    W = load_wtrades()
    byc = {c: (g.ts.values, g.pyes.values, g.yes_buy.values, g.sz.values) for c, g in W.groupby('cid', sort=False)}
    del W
    metars = {}
    rows = []
    for e in evs:
        if src.get(e['slug']) == 'HKO':
            continue
        if e['icao'] not in metars:
            metars[e['icao']] = load_metar(e['icao'], e['unit'])
        mt = metars[e['icao']]
        if mt is None:
            continue
        grid = np.arange(e['day_start'] - 30 * 3600, e['day_start'], 1800)
        nb = len(e['buckets'])
        refs = np.full((len(grid), nb), np.nan); ages = np.full_like(refs, np.nan); n1h = np.zeros_like(refs)
        for j, b in enumerate(e['buckets']):
            if b['cid'] not in byc:
                continue
            ts, py = byc[b['cid']][0], byc[b['cid']][1]
            idx = np.searchsorted(ts, grid, side='right') - 1; ok = idx >= 0
            refs[ok, j] = py[idx[ok]]; ages[ok, j] = (grid[ok] - ts[idx[ok]]) / 60
            n1h[:, j] = idx + 1 - np.searchsorted(ts, grid - 3600, side='right')
        over = np.nansum(refs, axis=1)
        rank = np.argsort(np.argsort(-np.nan_to_num(refs, nan=-1), axis=1), axis=1)
        for k, t in enumerate(grid):
            # most recent completed local day before t
            pd_end = e['day_start'] - 86400 * int(np.ceil((e['day_start'] - t) / 86400))
            w = mt[(mt.ts >= pd_end - 86400) & (mt.ts < pd_end) & (mt.ts + 300 <= t)]
            prev = (w.t.max() if e['kind'] == 'high' else w.t.min()) if len(w) > 8 else np.nan
            for j, b in enumerate(e['buckets']):
                if np.isnan(refs[k, j]):
                    continue
                lo, hi = max(b['rng'][0], -200), min(b['rng'][1], 200)
                rows.append((e['slug'], e['date'].isoformat(), j, b['cid'], t, (e['day_start'] - t) / 3600, refs[k, j],
                             ages[k, j], n1h[k, j], over[k], refs[k, j] / over[k] if over[k] > 0 else np.nan,
                             j / (nb - 1), int(j in (0, nb - 1)), nb, lo - prev, hi - prev,
                             int(e['kind'] == 'high'), int(e['unit'] == 'F'), rank[k, j], float(b['won'])))
    df = pd.DataFrame(rows, columns=['slug', 'date', 'j', 'cid', 't', 'hb', 'ref', 'age', 'n1h', 'over', 'refn', 'pos',
                                     'edgeb', 'nb', 'dlo_p', 'dhi_p', 'kindH', 'unitF', 'rank', 'y'])
    df.to_parquet('data/stack_prev_ds.parquet')
    return df, byc


if __name__ == '__main__':
    df, byc = build()
    tr, te = df[df.date < SPLIT], df[df.date >= SPLIT].copy()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0).fit(tr[SF], tr.y)
    te['p'] = clf.predict_proba(te[SF])[:, 1]
    print('train', len(tr), 'test', len(te), 'Brier market %.5f model %.5f' % (np.mean((te.ref - te.y) ** 2), np.mean((te.p - te.y) ** 2)))
    pickle.dump(clf, open('data/stack_prev_model.pkl', 'wb'))
    out = []
    for (slug, j), g in te.groupby(['slug', 'j'], sort=False):
        ts, py, yb, sz = byc[g.cid.iloc[0]]; used = np.zeros(len(ts)); spent = {'YES': 0.0, 'NO': 0.0}; Y = g.y.iloc[0]
        for r in g.itertuples():
            lo_i = np.searchsorted(ts, r.t + DELAY, side='right'); hi_i = np.searchsorted(ts, r.t + DELAY + WIN, side='right')
            for i in range(lo_i, hi_i):
                for side in ('YES', 'NO'):
                    if spent[side] >= MAXUSD:
                        continue
                    if side == 'YES' and yb[i] and py[i] <= r.p - EDGE:
                        px = py[i]
                    elif side == 'NO' and (not yb[i]) and (1 - py[i]) <= (1 - r.p) - EDGE:
                        px = 1 - py[i]
                    else:
                        continue
                    if px < 0.02 or px > 0.98:
                        continue
                    avail = sz[i] * SHARE - used[i]
                    if avail <= 0:
                        continue
                    take = min(avail, (MAXUSD - spent[side]) / px); used[i] += take; spent[side] += take * px
                    win = Y if side == 'YES' else 1 - Y
                    out.append((slug, r.date, side, r.hb, px, take, take * (win - px) - FEE * px * (1 - px) * take, win))
    F = pd.DataFrame(out, columns=['slug', 'date', 'side', 'hb', 'px', 'shares', 'pnl', 'win']); F['cost'] = F.shares * F.px
    d = F.groupby('date').pnl.sum()
    print(f'DAY-AHEAD EDGE={EDGE}: fills {len(F)} cost ${F.cost.sum():.0f} pnl ${F.pnl.sum():.0f} ret {F.pnl.sum()/F.cost.sum():.3f} '
          f'$/day {d.mean():.0f} posdays {(d>0).mean():.2f} sharpe {d.mean()/d.std()*np.sqrt(365):.1f}')
    F['half'] = np.where(F.date < '2026-08-29', 'A', 'B')
    print(F.groupby(['half', 'side']).agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum')).assign(ret=lambda x: x.pnl / x.cost).round(3))
    F.to_pickle('data/bt_prev.pkl')
