"""Backtest: nowcast-model taker strategy on weather markets, filled ONLY against real taker prints
that occurred after the signal (no speed assumption), with size capped at a fraction of printed volume.

Signal at 10-min grid time t (obs latency 5 min already applied in features):
  buy YES on bucket b if q_b - edge >= price of a print where someone bought YES (ask proof) in (t+delay, t+delay+win]
  buy NO  on bucket b if (1-q_b) - edge >= price of a print where someone bought NO in that window
"""
import sys, os, pickle, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from wlib import load_events
from train_nowcast import prep, FEATS, KMAX

EDGE = float(os.environ.get('EDGE', 0.08))
DELAY = float(os.environ.get('DELAY', 300))       # extra seconds after grid time
WIN = float(os.environ.get('WIN', 1800))
SHARE = float(os.environ.get('SHARE', 0.5))       # fraction of printed volume we could have taken
MAXUSD = float(os.environ.get('MAXUSD', 50))      # max $ per bucket-event
HRMIN = float(os.environ.get('HRMIN', 0))
FEE = 0.05


def bucket_probs(P, run, rngs, kind):
    """P: (n, KMAX+1) over D; returns (n, nbuckets) probabilities for final value in each bucket."""
    n = len(run); out = np.zeros((n, len(rngs)))
    for k in range(KMAX + 1):
        final = run + k if kind == 'high' else run - k
        for j, (lo, hi) in enumerate(rngs):
            out[:, j] += P[:, k] * ((final >= lo) & (final <= hi))
    # mass for D>=KMAX spread beyond: treat KMAX as ">=KMAX" -> assign to final=run+KMAX (approx)
    return out


def main():
    evs = [e for e in load_events() if e['date'].isoformat() >= '2026-06-20']
    src = json.load(open('data/weather_src.json'))
    T = pickle.load(open('data/weather_trades.pkl', 'rb'))
    models = {k: pickle.load(open(f'data/nowcast_{k}.pkl', 'rb')) for k in ('high', 'low')}
    feats = {}
    trades_out = []
    for kind in ('high', 'low'):
        parts = []
        for f in glob.glob(f'data/nowcast/{kind}_*.parquet'):
            d = pd.read_parquet(f)
            parts.append(d[d.date >= '2026-06-20'])
        feats[kind] = prep(pd.concat(parts, ignore_index=True))
    for kind in ('high', 'low'):
        F = feats[kind]
        F = F.assign(key=F.city + '|' + F.date)
        P = models[kind].predict_proba(F[FEATS])
        F['row'] = np.arange(len(F))
        groups = dict(tuple(F.groupby('key')))
        for e in evs:
            if e['kind'] != kind:
                continue
            key = e['city'] + '|' + e['date'].isoformat()
            if key not in groups or src.get(e['slug']) == 'HKO':
                continue
            g = groups[key]
            g = g[g.hr >= HRMIN]
            if len(g) == 0:
                continue
            run = g.run.values if kind == 'high' else -g.run.values
            rngs = [b['rng'] for b in e['buckets']]
            Pg = P[g.row.values]
            ok = Pg[:, KMAX] < float(os.environ.get('TAILMAX', 0.01))
            g = g[ok]; Pg = Pg[ok]; run = run[ok]
            if len(g) == 0:
                continue
            Q = bucket_probs(Pg, run, rngs, kind)
            tgrid = g.t.values
            for j, b in enumerate(e['buckets']):
                tr = T.get(b['cid'])
                if not tr:
                    continue
                Y = 1.0 if b['won'] else 0.0
                ts = np.array([x['timestamp'] for x in tr]); order = np.argsort(ts); ts = ts[order]
                tr = [tr[i] for i in order]
                yes_buy = np.array([((x['side'] == 'BUY') == (x['outcomeIndex'] == 0)) for x in tr])
                pyes = np.array([float(x['price']) if x['outcomeIndex'] == 0 else 1 - float(x['price']) for x in tr])
                sz = np.array([float(x['size']) for x in tr])
                used = np.zeros(len(tr))
                spent = {'YES': 0.0, 'NO': 0.0}
                for k in range(len(tgrid)):
                    q = Q[k, j]
                    lo_i = np.searchsorted(ts, tgrid[k] + DELAY, side='right')
                    hi_i = np.searchsorted(ts, tgrid[k] + DELAY + WIN, side='right')
                    if hi_i <= lo_i:
                        continue
                    for side in ('YES', 'NO'):
                        if spent[side] >= MAXUSD:
                            continue
                        for i in range(lo_i, hi_i):
                            if side == 'YES' and yes_buy[i] and pyes[i] <= q - EDGE:
                                px = pyes[i]
                            elif side == 'NO' and (not yes_buy[i]) and (1 - pyes[i]) <= (1 - q) - EDGE:
                                px = 1 - pyes[i]
                            else:
                                continue
                            avail = sz[i] * SHARE - used[i]
                            if avail <= 0 or px <= 0.001 or px >= 0.999:
                                continue
                            take = min(avail, (MAXUSD - spent[side]) / px)
                            used[i] += take; spent[side] += take * px
                            win = Y if side == 'YES' else 1 - Y
                            pnl = take * (win - px) - FEE * px * (1 - px) * take
                            trades_out.append((e['slug'], e['city'], kind, b['title'], side, tgrid[k], ts[i], g.hr.values[k],
                                               q if side == 'YES' else 1 - q, px, take, pnl, win, src.get(e['slug'])))
                            if spent[side] >= MAXUSD:
                                break
    df = pd.DataFrame(trades_out, columns=['slug', 'city', 'kind', 'bucket', 'side', 'tsig', 'tfill', 'hr', 'q', 'px',
                                           'shares', 'pnl', 'win', 'src'])
    df['cost'] = df.shares * df.px
    df['date'] = df.slug.str.extract(r'-on-(.*)')[0]
    tag = os.environ.get('TAG', 'base')
    df.to_pickle(f'data/bt_nowcast_{tag}.pkl')
    print(f'EDGE={EDGE} DELAY={DELAY} WIN={WIN} SHARE={SHARE} MAXUSD={MAXUSD} HRMIN={HRMIN}')
    print('fills', len(df), 'events', df.slug.nunique(), 'cost $%.0f pnl $%.0f ret %.3f' % (df.cost.sum(), df.pnl.sum(), df.pnl.sum() / max(df.cost.sum(), 1)))
    for col in ['side', 'kind', 'src']:
        print(df.groupby(col).agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum'), winrate=('win', 'mean'), q=('q', 'mean'), px=('px', 'mean')).round(3))
    df['pxb'] = pd.cut(df.px, [0, .1, .3, .5, .7, .9, .95, 1])
    print(df.groupby('pxb', observed=True).agg(n=('pnl', 'size'), cost=('cost', 'sum'), pnl=('pnl', 'sum'), winrate=('win', 'mean'), q=('q', 'mean')).round(3))
    daily = df.groupby('date').pnl.sum()
    print('days', len(daily), 'mean/day %.1f std %.1f pos-days %.2f' % (daily.mean(), daily.std(), (daily > 0).mean()))


if __name__ == '__main__':
    main()
