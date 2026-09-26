"""Live snapshots: Polymarket crypto threshold books vs Deribit-smile / realized-vol fair values.
Usage: python 02_snapshot.py [n_iter=13] [interval_s=300]
Writes data/crypto/snapshots/snap_<ts>.pkl (one DataFrame per snapshot)."""
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa

n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 13
interval = int(sys.argv[2]) if len(sys.argv) > 2 else 300
out_dir = os.path.join(DATA, 'snapshots')
os.makedirs(out_dir, exist_ok=True)
UNDS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']

ext_cache = {}   # (und, win_start) -> [hi, lo, last_ts]
rv_cache = {}


def load_markets():
    evs = gamma_keyset('events/keyset', {'closed': 'false', 'tag_slug': 'crypto'}, key='events')
    rows = []
    for e in evs:
        s = e.get('slug') or ''
        if 'updown' in s or 'up-or-down' in s:
            continue
        for m in e.get('markets') or []:
            r = parse_market(e, m)
            if r and not r['closed'] and r['accepting'] and r['yes']:
                rows.append(r)
    return pd.DataFrame(rows)


def update_extremes(df, now):
    last1m = {}
    for u in UNDS:
        k = klines(u, '1m', (now - pd.Timedelta(minutes=20)).value // 10**6, now.value // 10**6)
        last1m[u] = k
    for (u, ws) in df.loc[df.payoff.str.startswith('touch'), ['und', 'win_start']].drop_duplicates().itertuples(index=False):
        key = (u, ws)
        if key not in ext_cache:
            try:
                hi, lo = running_extremes(u, ws, now)
            except Exception as ex:
                print('ext err', key, ex, flush=True)
                continue
            ext_cache[key] = [hi, lo, now]
        else:
            k = last1m[u]
            k = k[k.index >= ext_cache[key][2] - pd.Timedelta(minutes=2)]
            if len(k):
                ext_cache[key][0] = max(ext_cache[key][0], k['h'].max())
                ext_cache[key][1] = min(ext_cache[key][1], k['l'].min())
            ext_cache[key][2] = now


def snapshot(i):
    now = pd.Timestamp.now(tz='UTC')
    df = load_markets()
    df = df[df['T'] > now].reset_index(drop=True)
    surfs = deribit_surfaces(UNDS, now)
    spot = spot_now(UNDS)
    for u in UNDS:
        if u not in rv_cache or (now - rv_cache[u][1]) > pd.Timedelta(minutes=30):
            rv_cache[u] = (realized_vol(u, days=7, interval='5m', end=now), now,
                           realized_vol(u, days=1, interval='1m', end=now))
    update_extremes(df, now)
    # books
    toks = list(df['yes'])
    bk = {}
    for j in range(0, len(toks), 400):
        for b in books(toks[j:j + 400]):
            bk[b['asset_id']] = b
    t_book = pd.Timestamp.now(tz='UTC')
    spot2 = spot_now(UNDS)
    out = []
    for r in df.to_dict('records'):
        b = bk.get(r['yes'])
        if b is None:
            continue
        top = book_top(b)
        hi = lo = None
        if r['payoff'].startswith('touch'):
            e = ext_cache.get((r['und'], r['win_start']))
            if e:
                hi, lo = e[0], e[1]
        S = spot[r['und']]
        mp = model_prob(r, S, now, surfs[r['und']], rv_cache[r['und']][0], hi, lo)
        mp1 = model_prob(r, S, now, None, rv_cache[r['und']][2], hi, lo)
        rec = dict(r, **top, **mp, p_rv1d=mp1.get('p_rv'), rv1d=rv_cache[r['und']][2], S=S, S_after=spot2[r['und']],
                   hi_so_far=hi, lo_so_far=lo, snap=i, t=now, t_book=t_book)
        out.append(rec)
    res = pd.DataFrame(out)
    res.to_pickle(os.path.join(out_dir, f"snap_{now.strftime('%Y%m%dT%H%M%S')}.pkl"))
    res['mid'] = (res['bid'] + res['ask']) / 2
    g = res.dropna(subset=['mid', 'p_iv'])
    g = g[(g.ask - g.bid) <= 0.1]
    print(now, 'n=', len(res), 'twosided=', len(g), 'mean|mid-iv|=', round((g.mid - g.p_iv).abs().mean(), 4),
          {u: round(spot[u], 4) for u in UNDS}, flush=True)


for i in range(n_iter):
    t0 = time.time()
    try:
        snapshot(i)
    except Exception:
        traceback.print_exc()
    if i < n_iter - 1:
        time.sleep(max(5, interval - (time.time() - t0)))
