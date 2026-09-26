"""Train nowcast classifiers P(D = k | obs-so-far features) for daily high and low.
Train: data before 2026-06-15. Test/market period: 2026-06-20 onwards."""
import sys, os, glob, pickle
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, os.path.dirname(__file__))
from stations import STATIONS

KMAX = 6
FEATS = ['hr', 'doy_s', 'doy_c', 'gap', 'tr1', 'tr3', 'unitF', 'prevgap', 'city_code', 'hr_x_gap', 'age']
CITY_CODE = {c: i for i, c in enumerate(sorted(STATIONS))}


def prep(df):
    df = df.copy()
    df['doy_s'] = np.sin(2 * np.pi * df.doy / 365.25); df['doy_c'] = np.cos(2 * np.pi * df.doy / 365.25)
    df['city_code'] = df.city.map(CITY_CODE)
    df['hr_x_gap'] = df.hr * df.gap
    df['y'] = np.clip(df.D, 0, KMAX).astype(int)
    return df


if __name__ == '__main__':
    kind = sys.argv[1]
    parts = []
    for f in sorted(glob.glob(f'data/nowcast/{kind}_*.parquet')):
        d = pd.read_parquet(f)
        d = d[(np.round(d.hr * 6) % 3 == 0)]          # every 30 minutes
        parts.append(d)
    df = prep(pd.concat(parts, ignore_index=True))
    tr = df[df.date < '2026-06-15']; te = df[df.date >= '2026-06-20']
    print(kind, 'train', len(tr), 'test', len(te), flush=True)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                         categorical_features=[FEATS.index('city_code')], random_state=0)
    clf.fit(tr[FEATS], tr.y)
    pickle.dump(clf, open(f'data/nowcast_{kind}.pkl', 'wb'))
    P = clf.predict_proba(te[FEATS])
    ll = -np.mean(np.log(np.clip(P[np.arange(len(te)), te.y.values], 1e-6, 1)))
    # climatological baseline by (city, hour)
    base = tr.groupby([tr.city, (tr.hr).round()]).y.value_counts(normalize=True).unstack(fill_value=0)
    print('test logloss', round(ll, 4))
    te = te.assign(p0=P[:, 0])
    for h in [10, 12, 14, 15, 16, 17, 18, 20, 22]:
        x = te[(te.hr >= h) & (te.hr < h + 1)]
        # calibration of P(D=0)
        bins = pd.cut(x.p0, [0, .5, .8, .9, .95, .98, .99, 1.0])
        g = x.groupby(bins, observed=True).agg(n=('y', 'size'), p=('p0', 'mean'), obs=('y', lambda s: (s == 0).mean()))
        print('hour', h, g.round(3).to_dict('index'))
