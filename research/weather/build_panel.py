"""Build a 10-minute panel: for each event x time: bucket prices, obs running extreme, outcome."""
import sys, pickle, os
sys.path.insert(0, os.path.dirname(__file__))
from wlib import *

GRID = 600
LAT = 300   # assume a METAR becomes usable 5 min after its timestamp


def build(evs, P, metars):
    rows = []
    for e in evs:
        mt = metars.get(e['icao'])
        if mt is None:
            continue
        nb = len(e['buckets'])
        wi = [i for i, b in enumerate(e['buckets']) if b['won']]
        if len(wi) != 1:
            continue
        wi = wi[0]
        t0 = int(e['day_start'] - 36 * 3600) // GRID * GRID
        t1 = int(e['day_end'] + 6 * 3600) // GRID * GRID
        grid = np.arange(t0, t1 + 1, GRID)
        mat = np.full((len(grid), nb), np.nan)
        for j, b in enumerate(e['buckets']):
            h = P.get(b['toks'][0]) if b['toks'] else None
            if not h:
                continue
            ts = np.array([x['t'] for x in h]); ps = np.array([x['p'] for x in h])
            idx = np.searchsorted(ts, grid, side='right') - 1
            ok = idx >= 0
            mat[ok, j] = ps[idx[ok]]
        w = mt[(mt.ts >= e['day_start']) & (mt.ts < e['day_end'])]
        if len(w) < 8:
            continue
        ots = w.ts.values + LAT
        ext = np.maximum.accumulate(w.t.values) if e['kind'] == 'high' else np.minimum.accumulate(w.t.values)
        oi = np.searchsorted(ots, grid, side='right') - 1
        final = w.t.max() if e['kind'] == 'high' else w.t.min()
        for k, t in enumerate(grid):
            if np.all(np.isnan(mat[k])):
                continue
            rows.append((e['slug'], e['city'], e['kind'], e['unit'], t, (t - e['day_start']) / 3600.0,
                         ext[oi[k]] if oi[k] >= 0 else np.nan, final, wi, nb, mat[k].copy()))
    df = pd.DataFrame(rows, columns=['slug', 'city', 'kind', 'unit', 't', 'hr', 'ext', 'final', 'wi', 'nb', 'px'])
    return df


if __name__ == '__main__':
    evs = load_events()
    P = pickle.load(open('data/weather_prices.pkl', 'rb'))
    evs = [e for e in evs if all(b['toks'] and b['toks'][0] in P for b in e['buckets'])]
    metars = {}
    for e in evs:
        if e['icao'] not in metars:
            metars[e['icao']] = load_metar(e['icao'], e['unit'])
    df = build(evs, P, metars)
    # attach bucket ranges
    rng = {e['slug']: [b['rng'] for b in e['buckets']] for e in evs}
    df['rngs'] = df.slug.map(rng)
    df.to_pickle('data/weather_panel.pkl')
    print(len(df), df.slug.nunique())
