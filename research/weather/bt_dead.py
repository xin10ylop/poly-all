"""Dead-bucket sniping backtest at realistic (minute-level) latency.
A bucket is 'dead' once a METAR shows the running extreme beyond it (high: run > hi; low: run < lo).
We can act at obs_time + LAT minutes (public feed publication ~4-6 min + polling). We buy NO at the price of real
taker NO-buy prints (i.e. resting YES bids / NO asks that were really there) in (obs+LAT, obs+LAT+WIN], taking at most
SHARE of each print, capped per bucket. Outcome = actual resolution (so METAR/resolution mismatches are paid for)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from wlib import load_events, load_wtrades, load_metar

LAT = float(os.environ.get('LAT', 6)) * 60; WIN = float(os.environ.get('WIN', 30)) * 60
SHARE = float(os.environ.get('SHARE', 0.5)); MAXUSD = float(os.environ.get('MAXUSD', 200)); FEE = 0.05
PMIN = float(os.environ.get('PMIN', 0.005))          # min YES price worth selling (NO price <= 1-PMIN)
EXCL = {'moscow', 'manila', 'panama-city', 'hong-kong'}
src = json.load(open('data/weather_src.json'))
evs = [e for e in load_events() if e['date'].isoformat() >= '2026-06-20' and e['city'] not in EXCL and src.get(e['slug']) != 'HKO']
W = load_wtrades()
byc = {c: (g.ts.values, g.pyes.values, g.yes_buy.values, g.sz.values) for c, g in W.groupby('cid', sort=False)}
del W
metars = {}
out = []
for e in evs:
    if e['icao'] not in metars:
        metars[e['icao']] = load_metar(e['icao'], e['unit'])
    mt = metars[e['icao']]
    if mt is None:
        continue
    w = mt[(mt.ts >= e['day_start']) & (mt.ts < e['day_end'])]
    if len(w) < 8:
        continue
    vals = w.t.values if e['kind'] == 'high' else -w.t.values
    run = np.maximum.accumulate(vals); ots = w.ts.values
    for b in e['buckets']:
        lo, hi = b['rng']
        thr = hi if e['kind'] == 'high' else -lo
        idx = np.nonzero(run > thr)[0]
        if len(idx) == 0 or b['cid'] not in byc:
            continue
        tdead = ots[idx[0]]
        ts, py, yb, sz = byc[b['cid']]
        i0 = np.searchsorted(ts, tdead + LAT, side='right'); i1 = np.searchsorted(ts, tdead + LAT + WIN, side='right')
        spent = 0.0; Y = 1.0 if b['won'] else 0.0
        for i in range(i0, i1):
            if yb[i] or py[i] < PMIN:          # need someone BUYING NO (hitting YES bids / NO asks) at pyes >= PMIN
                continue
            px = 1 - py[i]
            take = min(sz[i] * SHARE, (MAXUSD - spent) / px)
            if take <= 0:
                break
            spent += take * px
            out.append((e['slug'], e['city'], e['kind'], e['date'].isoformat(), src.get(e['slug']), b['title'],
                        (ts[i] - tdead) / 60, py[i], px, take, take * ((1 - Y) - px) - FEE * px * (1 - px) * take, Y))
F = pd.DataFrame(out, columns=['slug', 'city', 'kind', 'date', 'src', 'bucket', 'min_after_obs', 'pyes', 'px', 'shares', 'pnl', 'lost'])
F['cost'] = F.shares * F.px
tag = f"L{int(LAT/60)}"; F.to_pickle(f'data/bt_dead_{tag}.pkl')
print(f'LAT={LAT/60:.0f}min WIN={WIN/60:.0f}min SHARE={SHARE} MAXUSD={MAXUSD}')
for name, f in [('ALL', F), ('NOAA', F[F.src == 'NOAA']), ('WU', F[F.src == 'WU']), ('NOAA since Aug24', F[(F.src == 'NOAA') & (F.date >= '2026-08-24')])]:
    if not len(f):
        continue
    d = f.groupby('date').pnl.sum()
    print(f"{name:18s} fills {len(f):6d} buckets {f.groupby(['slug','bucket']).ngroups:5d} cost ${f.cost.sum():9.0f} pnl ${f.pnl.sum():8.0f} "
          f"ret {f.pnl.sum()/f.cost.sum():+.3%} lost-bucket rate {f.groupby(['slug','bucket']).lost.max().mean():.4f} days {len(d)} $/day {d.mean():6.1f} posdays {(d>0).mean():.2f}")
