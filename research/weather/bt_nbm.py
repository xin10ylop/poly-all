"""Day-ahead / same-day forecast edge for US cities using archived NBM (NBS) txn + xnd.
Probability of bucket = Normal(txn + bias, (k*xnd)^2) integrated over [lo-0.5, hi+0.5].
Signal at NBM availability time (runtime + AVAIL); fills only on real prints in (t+DELAY, t+DELAY+WIN]."""
import sys, os, json
import numpy as np, pandas as pd, datetime as dt
from zoneinfo import ZoneInfo
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(__file__))
from wlib import load_events, load_wtrades
from stations import STATIONS

AVAIL = 5400; DELAY = float(os.environ.get('DELAY', 300)); WIN = float(os.environ.get('WIN', 3600))
EDGE = float(os.environ.get('EDGE', 0.08)); SHARE = 0.5; MAXUSD = float(os.environ.get('MAXUSD', 50)); FEE = 0.05
K = float(os.environ.get('K', 1.0)); SPLIT = '2026-08-15'


def load_nbm(icao):
    df = pd.read_csv(f'data/nbm/{icao}.csv', usecols=['runtime', 'ftime', 'txn', 'xnd'])
    df = df.dropna(subset=['txn'])
    df['rt'] = pd.to_datetime(df.runtime, utc=True).astype('int64') // 10**9 if False else \
        (pd.to_datetime(df.runtime, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)
    df['ft'] = (pd.to_datetime(df.ftime, utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)
    return df


def main():
    evs = [e for e in load_events() if e['date'].isoformat() >= '2026-06-20' and e['icao'] in
           ['KLGA', 'KATL', 'KMIA', 'KHOU', 'KDAL', 'KORD', 'KAUS', 'KBKF', 'KSEA', 'KLAX', 'KSFO', 'CYYZ']]
    W = load_wtrades()
    byc = {c: g for c, g in W.groupby('cid', sort=False)}
    del W
    nbm = {}
    sig, fills = [], []
    for e in evs:
        if e['icao'] not in nbm:
            nbm[e['icao']] = load_nbm(e['icao'])
        N = nbm[e['icao']]
        z = ZoneInfo(e['tz']); d = e['date']
        if e['kind'] == 'high':
            target = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() + 86400   # 00Z next day
        else:
            target = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() + 12 * 3600  # 12Z same day
        rows = N[(N.ft == target)].sort_values('rt')
        for r in rows.itertuples():
            t = r.rt + AVAIL
            if t > e['day_end']:
                continue
            mu, sd = r.txn, max(K * r.xnd, 0.8)
            probs = []
            for b in e['buckets']:
                lo, hi = b['rng']
                probs.append(norm.cdf(hi + 0.5, mu, sd) - norm.cdf(lo - 0.5, mu, sd))
            for j, b in enumerate(e['buckets']):
                tr = byc.get(b['cid'])
                if tr is None:
                    continue
                q = probs[j]; Y = 1.0 if b['won'] else 0.0
                ts = tr.ts.values
                i0 = np.searchsorted(ts, t, side='right') - 1
                ref = tr.pyes.values[i0] if i0 >= 0 else np.nan
                sig.append((e['slug'], e['kind'], e['date'].isoformat(), j, r.rt, (e['day_start'] - t) / 3600, q, ref, Y))
                lo_i = np.searchsorted(ts, t + DELAY, side='right'); hi_i = np.searchsorted(ts, t + DELAY + WIN, side='right')
                spent = {'YES': 0.0, 'NO': 0.0}
                for i in range(lo_i, hi_i):
                    py = tr.pyes.values[i]; yb = tr.yes_buy.values[i]; sz = tr.sz.values[i] * SHARE
                    for side in ('YES', 'NO'):
                        if spent[side] >= MAXUSD:
                            continue
                        if side == 'YES' and yb and py <= q - EDGE:
                            px = py
                        elif side == 'NO' and (not yb) and (1 - py) <= (1 - q) - EDGE:
                            px = 1 - py
                        else:
                            continue
                        if px <= 0.002 or px >= 0.998:
                            continue
                        take = min(sz, (MAXUSD - spent[side]) / px); spent[side] += take * px
                        win = Y if side == 'YES' else 1 - Y
                        fills.append((e['slug'], e['kind'], e['date'].isoformat(), j, side, r.rt, (e['day_start'] - t) / 3600,
                                      q if side == 'YES' else 1 - q, px, take, take * (win - px) - FEE * px * (1 - px) * take, win))
    S = pd.DataFrame(sig, columns=['slug', 'kind', 'date', 'j', 'rt', 'lead_h', 'q', 'ref', 'y'])
    F = pd.DataFrame(fills, columns=['slug', 'kind', 'date', 'j', 'side', 'rt', 'lead_h', 'q', 'px', 'shares', 'pnl', 'win'])
    S.to_pickle('data/nbm_sig.pkl'); F.to_pickle('data/nbm_fills.pkl')
    s = S.dropna(subset=['ref'])
    s['lb'] = pd.cut(s.lead_h, [-24, 0, 12, 24, 48, 100])
    print('Brier by lead (hours before local day start): market vs NBM(K=%.2f)' % K)
    print(s.groupby(['kind', 'lb'], observed=True).apply(lambda x: pd.Series(dict(n=len(x), market=np.mean((x.ref - x.y) ** 2), nbm=np.mean((x.q - x.y) ** 2), blend=np.mean(((x.q + x.ref) / 2 - x.y) ** 2)))).round(5))
    F['cost'] = F.shares * F.px
    for per, f in [('train', F[F.date < SPLIT]), ('test', F[F.date >= SPLIT])]:
        print(per, 'fills', len(f), 'cost %.0f pnl %.0f ret %.3f' % (f.cost.sum(), f.pnl.sum(), f.pnl.sum() / max(1, f.cost.sum())))
        print(f.groupby(['kind', 'side']).agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum'), win=('win', 'mean'), q=('q', 'mean'), px=('px', 'mean')).round(3))


if __name__ == '__main__':
    main()
