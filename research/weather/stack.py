"""Stacked model: combine market reference price (last print) with the METAR nowcast.
Build dataset on a 30-min grid, train on early period, evaluate & trade on later period with print-based fills."""
import sys, os, pickle, json, glob
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(__file__))
from wlib import load_events, load_wtrades
from train_nowcast import prep, FEATS, KMAX
from bt_nowcast import bucket_probs

SPLIT = os.environ.get('SPLIT', '2026-08-15')


def build():
    evs = [e for e in load_events() if e['date'].isoformat() >= '2026-06-20']
    src = json.load(open('data/weather_src.json'))
    W = load_wtrades()
    byc = {c: g[['ts', 'pyes']] for c, g in W.groupby('cid', sort=False)}
    del W
    models = {k: pickle.load(open(f'data/nowcast_{k}.pkl', 'rb')) for k in ('high', 'low')}
    rows = []
    for kind in ('high', 'low'):
        parts = [pd.read_parquet(f) for f in glob.glob(f'data/nowcast/{kind}_*.parquet')]
        F = prep(pd.concat([p[p.date >= '2026-06-20'] for p in parts], ignore_index=True))
        F = F[(np.round(F.hr * 6) % 3 == 0)]
        P = models[kind].predict_proba(F[FEATS])
        F = F.assign(key=F.city + '|' + F.date, row=np.arange(len(F)))
        groups = dict(tuple(F.groupby('key')))
        for e in evs:
            if e['kind'] != kind or src.get(e['slug']) == 'HKO':
                continue
            g = groups.get(e['city'] + '|' + e['date'].isoformat())
            if g is None:
                continue
            Pg = P[g.row.values]
            run = g.run.values if kind == 'high' else -g.run.values
            rngs = [b['rng'] for b in e['buckets']]
            Q = bucket_probs(Pg, run, rngs, kind)
            tail = Pg[:, KMAX]
            refs = np.full((len(g), len(rngs)), np.nan); ages = np.full((len(g), len(rngs)), np.nan)
            n1h = np.zeros((len(g), len(rngs)))
            for j, b in enumerate(e['buckets']):
                tr = byc.get(b['cid'])
                if tr is None:
                    continue
                ts = tr.ts.values; py = tr.pyes.values
                idx = np.searchsorted(ts, g.t.values, side='right') - 1
                ok = idx >= 0
                refs[ok, j] = py[idx[ok]]; ages[ok, j] = (g.t.values[ok] - ts[idx[ok]]) / 60
                n1h[:, j] = idx + 1 - np.searchsorted(ts, g.t.values - 3600, side='right')
            over = np.nansum(refs, axis=1)
            for j, b in enumerate(e['buckets']):
                lo, hi = b['rng']
                lo = max(lo, -200); hi = min(hi, 200)
                rows.append(pd.DataFrame(dict(
                    slug=e['slug'], city=e['city'], kind=kind, date=e['date'].isoformat(), j=j, cid=b['cid'],
                    t=g.t.values, hr=g.hr.values, q=Q[:, j].astype('float32'), tail=tail.astype('float32'),
                    ref=refs[:, j].astype('float32'), age=ages[:, j].astype('float32'), n1h=n1h[:, j].astype('float32'),
                    over=over.astype('float32'),
                    refn=np.where(over > 0, refs[:, j] / np.where(over > 0, over, 1), np.nan).astype('float32'),
                    dlo=((lo - run) if kind == 'high' else (run - hi)).astype('float32'),
                    dhi=((hi - run) if kind == 'high' else (run - lo)).astype('float32'),
                    gap=g.gap.values.astype('float32'), tr1=g.tr1.values.astype('float32'),
                    tr3=g.tr3.values.astype('float32'), unitF=g.unitF.values, doy=g.doy.values, nb=len(rngs),
                    y=float(b['won']), src=src.get(e['slug']))))
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet('data/stack_ds.parquet')
    return df


SF = ['hr', 'q', 'tail', 'ref', 'age', 'n1h', 'over', 'refn', 'dlo', 'dhi', 'gap', 'tr1', 'tr3', 'unitF', 'kindH', 'nb']


def fit(df):
    df = df.assign(kindH=(df.kind == 'high').astype(int))
    tr = df[(df.date < SPLIT) & df.ref.notna()]
    te = df[(df.date >= SPLIT) & (df.date < os.environ.get('TEST_END', '2099-01-01')) & df.ref.notna()]
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(tr[SF], tr.y)
    te = te.assign(p=clf.predict_proba(te[SF])[:, 1])
    br_m = np.mean((te.ref - te.y) ** 2); br_s = np.mean((te.p - te.y) ** 2); br_q = np.mean((te.q - te.y) ** 2)
    print('train', len(tr), 'test', len(te), 'Brier market(ref) %.5f  stacked %.5f  nowcast %.5f' % (br_m, br_s, br_q))
    for h0, h1 in [(0, 10), (10, 13), (13, 16), (16, 19), (19, 24)]:
        x = te[(te.hr >= h0) & (te.hr < h1)]
        print('  hr %2d-%2d n %7d  market %.5f stacked %.5f' % (h0, h1, len(x), np.mean((x.ref - x.y) ** 2), np.mean((x.p - x.y) ** 2)))
    pickle.dump(clf, open('data/stack_model.pkl' if SPLIT == '2026-08-15' else f'data/stack_model_{SPLIT}.pkl', 'wb'))
    return clf, te


if __name__ == '__main__':
    df = build() if (len(sys.argv) > 1 and sys.argv[1] == 'build') or not os.path.exists('data/stack_ds.parquet') else pd.read_parquet('data/stack_ds.parquet')
    print(len(df))
    clf, te = fit(df)
    te.to_parquet('data/stack_test.parquet' if SPLIT == '2026-08-15' else f'data/stack_test_{SPLIT}.parquet')
