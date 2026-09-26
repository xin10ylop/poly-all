"""Join a wallet's weather trades with METAR state at trade time to reverse-engineer its rule."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
from wlib import *

wallet = sys.argv[1]
evs = load_events()
T = pickle.load(open('data/weather_trades.pkl', 'rb'))
metars = {}
rows = []
for e in evs:
    for bi, b in enumerate(e['buckets']):
        tr = [x for x in (T.get(b['cid']) or []) if x['proxyWallet'] == wallet]
        if not tr:
            continue
        if e['icao'] not in metars:
            metars[e['icao']] = load_metar(e['icao'], e['unit'])
        mt = metars[e['icao']]
        w = mt[(mt.ts >= e['day_start']) & (mt.ts < e['day_end'])]
        for x in tr:
            ww = w[w.ts <= x['timestamp']]
            run = (ww.t.max() if e['kind'] == 'high' else ww.t.min()) if len(ww) else np.nan
            cur = ww.t.iloc[-1] if len(ww) else np.nan
            lastobs_age = (x['timestamp'] - ww.ts.iloc[-1]) / 60 if len(ww) else np.nan
            lo, hi = b['rng']
            if np.isnan(run):
                rel = 'noobs'
            elif e['kind'] == 'high':
                rel = 'dead' if hi < run else ('current' if lo <= run <= hi else 'above')
            else:
                rel = 'dead' if lo > run else ('current' if lo <= run <= hi else 'below')
            yes = x['outcomeIndex'] == 0
            buy_yes = (x['side'] == 'BUY') == yes
            p = float(x['price']); pyes = p if yes else 1 - p
            Y = 1.0 if b['won'] else 0.0
            pnl = float(x['size']) * ((Y - pyes) if buy_yes else (pyes - Y)) - 0.05 * p * (1 - p) * float(x['size'])
            lh = dt.datetime.fromtimestamp(x['timestamp'], ZoneInfo(e['tz'])).hour + \
                dt.datetime.fromtimestamp(x['timestamp'], ZoneInfo(e['tz'])).minute / 60
            rows.append((e['city'], e['kind'], str(e['date']), x['timestamp'], lh, b['title'], rel, run, cur, lastobs_age,
                         'BUY_YES' if buy_yes else 'BUY_NO', pyes, float(x['size']), pnl, b['won'],
                         (x['timestamp'] - e['day_start']) / 3600))
df = pd.DataFrame(rows, columns=['city', 'kind', 'date', 'ts', 'lhour', 'bucket', 'rel', 'run', 'cur', 'obs_age_min',
                                 'action', 'pyes', 'size', 'pnl', 'won', 'hrs_into_day'])
df.to_pickle(f'data/wallet_{wallet[:10]}.pkl')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 20)
print(df.groupby(['rel', 'action']).agg(n=('pnl', 'size'), pnl=('pnl', 'sum'), pyes=('pyes', 'median'),
                                         won=('won', 'mean'), lh=('lhour', 'median'), age=('obs_age_min', 'median')).round(3))
print(df.groupby(pd.cut(df.hrs_into_day, [-48, -12, 0, 6, 9, 12, 15, 18, 24, 48]), observed=True).agg(
    n=('pnl', 'size'), pnl=('pnl', 'sum')).round(1))
print(df.sample(15, random_state=1).round(3).to_string())
