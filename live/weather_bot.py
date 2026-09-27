"""Live paper-trading bot: weather temperature markets, stacked (market + METAR nowcast) model.

Every LOOP seconds:
  1. refresh universe (active temperature events for the current local day of each city; NOAA/WU sources, no HKO)
  2. fetch latest METARs (aviationweather.gov) -> per-station history (resolution units)
  3. fetch new taker prints for all bucket markets (data-api, batched by condition id) -> last price, age, n1h
  4. compute nowcast bucket probabilities + stacked probability p per bucket
  5. read live order books; if p - best_ask(YES) >= EDGE (or (1-p) - best_ask(NO) >= EDGE) paper-buy at the
     displayed ask, size limited by displayed depth at that price and MAXUSD per bucket/side
  6. settle positions when markets resolve
State and a full signal/fill log go to data/live/weather/.
"""
import json, os, sys, time, pickle, traceback, math
import numpy as np, pandas as pd, datetime as dt
from zoneinfo import ZoneInfo
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'research', 'weather'))
from polylib import *
from stations import STATIONS
from wlib import parse_bucket, city_of, obs_date
from features_live import features_at
from train_nowcast import prep, FEATS, KMAX
from bt_nowcast import bucket_probs
from stack import SF
sys.path.insert(0, os.path.join(ROOT, 'live'))
import rules_guard

EDGE = float(os.environ.get('EDGE', 0.15)); MAXUSD = float(os.environ.get('MAXUSD', 50)); LOOP = 60
FEE = 0.05; PMIN, PMAX = 0.02, 0.98
MAX_AGE = float(os.environ.get('MAX_AGE', 60))   # minutes since last print
OUT = os.path.join(ROOT, 'data', 'live', 'weather'); os.makedirs(OUT, exist_ok=True)
NOW = lambda: time.time()
models = {k: pickle.load(open(os.path.join(ROOT, f'data/nowcast_{k}.pkl'), 'rb')) for k in ('high', 'low')}
stack = pickle.load(open(os.path.join(ROOT, os.environ.get('STACK_MODEL', 'data/stack_model_prod.pkl')), 'rb'))
state_path = os.path.join(OUT, 'state.pkl')
state = pickle.load(open(state_path, 'rb')) if os.path.exists(state_path) else {'positions': [], 'spent': {}, 'closed': []}


def log(name, rec):
    with open(os.path.join(OUT, name), 'a') as f:
        f.write(json.dumps(rec, default=float) + '\n')


def universe():
    evs = gamma_paginate('events', {'tag_slug': 'weather', 'active': 'true', 'closed': 'false'}, page=100)
    U = []
    for e in evs:
        if 'temperature' not in e['slug']:
            continue
        city = city_of(e['slug'])
        if city not in STATIONS:
            continue
        desc = e.get('description') or ''
        src = 'NOAA' if 'weather.gov/wrh/timeseries' in desc else ('WU' if 'wunderground' in desc else 'HKO')
        if src == 'HKO':
            continue
        try:   # Jev rules guard (cached per event text); unapproved events are never traded
            if not rules_guard.approve(e['slug'], desc, 'high' if e['slug'].startswith('highest') else 'low', allow_llm=True):
                continue
        except Exception as ex:
            print('rules_guard error', ex, flush=True)
            continue
        try:
            d = obs_date(e['slug'])
        except Exception:
            continue
        icao, unit, tz = STATIONS[city]
        z = ZoneInfo(tz)
        if dt.datetime.now(z).date() != d:          # model is trained on the observation day only
            continue
        bks = []
        for m in e['markets']:
            rng = parse_bucket(m.get('groupItemTitle') or m['question'])
            toks = jl(m.get('clobTokenIds'))
            if rng and toks and m.get('acceptingOrders', True):
                bks.append(dict(rng=rng, cid=m['conditionId'], yes=toks[0], no=toks[1], title=m.get('groupItemTitle'),
                                tick=float(m.get('orderPriceMinTickSize') or 0.01)))
        bks.sort(key=lambda b: b['rng'][0])
        if bks:
            U.append(dict(slug=e['slug'], city=city, kind='high' if e['slug'].startswith('highest') else 'low',
                          icao=icao, unit=unit, tz=tz, date=d.isoformat(), src=src, buckets=bks,
                          day_start=dt.datetime(d.year, d.month, d.day, tzinfo=z).timestamp()))
    return U


METAR = {}
BACKFILLED = set()


def _ingest(d):
    for x in d:
        if x.get('temp') is None:
            continue
        METAR.setdefault(x['icaoId'], {})[int(x['obsTime'])] = float(x['temp'])


def update_metar(icaos):
    """aviationweather.gov caps a response at 400 obs, so: backfill 40h per station individually once, then poll
    recent obs in small batches (20 stations x 3h stays far below the cap)."""
    ids = sorted(set(icaos))
    for icao in ids:
        if icao not in BACKFILLED:
            try:
                _ingest(get('https://aviationweather.gov/api/data/metar', {'ids': icao, 'format': 'json', 'hours': 40}))
                BACKFILLED.add(icao)
            except Exception:
                pass
    for i in range(0, len(ids), 20):
        try:
            _ingest(get('https://aviationweather.gov/api/data/metar', {'ids': ','.join(ids[i:i + 20]), 'format': 'json', 'hours': 3}))
        except Exception:
            continue


def day_coverage_ok(icao, day_start, now):
    """Require observations covering the local day so far: first obs within 90 min of day start and no gap > 3h."""
    ts = sorted(t for t in METAR.get(icao, {}) if day_start - 3600 <= t <= now)
    if not ts or icao not in BACKFILLED:
        return False
    day = [t for t in ts if t >= day_start]
    if not day or day[0] - day_start > 5400:
        return False
    gaps = [b - a for a, b in zip(day, day[1:])] + [now - day[-1]]
    return max(gaps) <= 3 * 3600


def obs_overdue(icao, now):
    """True if the station's usual reporting interval has elapsed since its last observation we hold, i.e. a newer
    METAR is probably out (the market may already know it while our feed has not delivered it yet)."""
    ts = sorted(METAR.get(icao, {}))
    if len(ts) < 3:
        return True
    gaps = [b - a for a, b in zip(ts[-7:], ts[-6:])]
    interval = sorted(gaps)[len(gaps) // 2]
    return now - ts[-1] >= interval - 360   # METARs are often published a few minutes BEFORE their nominal time


def in_blackout(icao, now, lat=300):
    """True while the market may know an observation our model does not use yet:
    (a) a newer METAR is probably published but not in our feed (reporting interval elapsed), or
    (b) we hold an observation younger than the model's 5-min availability lag (kept for backtest parity)."""
    ts = sorted(METAR.get(icao, {}))
    return obs_overdue(icao, now) or (bool(ts) and ts[-1] + lat > now)


def metar_frame(icao, unit):
    obs = METAR.get(icao, {})
    if not obs:
        return None
    ts = np.array(sorted(obs)); c = np.array([obs[t] for t in ts])
    tf = c * 9 / 5 + 32 if unit == 'F' else c
    t = np.round(tf + 1e-9).astype(int)
    return pd.DataFrame(dict(ts=ts, t=t, tf=tf))


PRINTS = {}


def update_prints(cids):
    cids = list(cids)
    for i in range(0, len(cids), 30):
        try:
            r = get(f'{DATA}/trades', {'market': ','.join(cids[i:i + 30]), 'limit': 500})
        except Exception:
            continue
        for x in r:
            p = float(x['price']); pyes = p if x['outcomeIndex'] == 0 else 1 - p
            PRINTS.setdefault(x['conditionId'], {})[(x['timestamp'], x['transactionHash'], x['asset'], x['price'], x['size'], x['side'])] = pyes


def market_feats(cid, now):
    pr = PRINTS.get(cid)
    if not pr:
        return np.nan, np.nan, 0
    ks = sorted(pr)
    last = ks[-1]
    n1h = sum(1 for k in ks if k[0] >= now - 3600)
    return pr[last], (now - last[0]) / 60, n1h


def best(levels, side):
    lv = [(float(x['price']), float(x['size'])) for x in levels]
    lv.sort(key=lambda x: -x[0] if side == 'bid' else x[0])
    return lv


def step(U):
    now = NOW()
    update_metar([u['icao'] for u in U])
    update_prints([b['cid'] for u in U for b in u['buckets']])
    toks = [b['yes'] for u in U for b in u['buckets']]
    BK = {}
    for i in range(0, len(toks), 400):
        for bk in books(toks[i:i + 400]):
            BK[bk['asset_id']] = (best(bk.get('bids', []), 'bid'), best(bk.get('asks', []), 'ask'))
    rows, meta = [], []
    for u in U:
        mf = metar_frame(u['icao'], u['unit'])
        if mf is None or not day_coverage_ok(u['icao'], u['day_start'], now):
            continue
        f = features_at(mf, u['city'], u['kind'], now)
        if f is None:
            continue
        F = prep(pd.DataFrame([f]))
        P = models[u['kind']].predict_proba(F[FEATS])
        # no tail gate: the stacked model was trained on all hours with `tail` as a feature (matches bt_stack.py)
        run = f['run'] if u['kind'] == 'high' else -f['run']
        rngs = [b['rng'] for b in u['buckets']]
        Q = bucket_probs(P, np.array([run]), rngs, u['kind'])[0]
        mk = [market_feats(b['cid'], now) for b in u['buckets']]
        refs = np.array([m[0] for m in mk], dtype=float); over = np.nansum(refs)
        for j, b in enumerate(u['buckets']):
            if np.isnan(refs[j]):
                continue
            lo, hi = max(b['rng'][0], -200), min(b['rng'][1], 200)
            rows.append(dict(hr=f['hr'], q=Q[j], tail=P[0, KMAX], ref=refs[j], age=mk[j][1], n1h=mk[j][2], over=over,
                             refn=refs[j] / over if over > 0 else np.nan,
                             dlo=(lo - run) if u['kind'] == 'high' else (run - hi),
                             dhi=(hi - run) if u['kind'] == 'high' else (run - lo),
                             gap=f['gap'], tr1=f['tr1'], tr3=f['tr3'], unitF=f['unitF'], kindH=int(u['kind'] == 'high'),
                             nb=len(rngs)))
            meta.append((u, b, f))
    if not rows:
        return 0
    PS = stack.predict_proba(pd.DataFrame(rows)[SF])[:, 1]
    for (u, b, f), row, p in zip(meta, rows, PS):
        p = float(p)
        bids, asks = BK.get(b['yes'], ([], []))
        ask_yes = asks[0] if asks else None
        ask_no = (1 - bids[0][0], bids[0][1]) if bids else None
        log('signals.jsonl', dict(ts=now, slug=u['slug'], cid=b['cid'], bucket=b['title'], p=p, q=row['q'], ref=row['ref'],
                                  age=row['age'], n1h=row['n1h'], ask_yes=ask_yes, ask_no=ask_no, hr=f['hr'],
                                  run=f['run'], src=u['src']))
        if not (row['age'] <= MAX_AGE):   # validated regime: 96% of backtest fills had a print within the last hour
            continue
        for side, a in (('YES', ask_yes), ('NO', ask_no)):
            if a is None:
                continue
            px, depth = a
            fair = p if side == 'YES' else 1 - p
            ref_side = row['ref'] if side == 'YES' else 1 - row['ref']
            if ref_side - px > 0.25 and fair - px >= EDGE and in_blackout(u['icao'], now):
                # quote far better than the last trade while the market may know an obs our model does not -> skip
                log('guard_skip.jsonl', dict(ts=now, slug=u['slug'], bucket=b['title'], side=side, px=px, fair=fair, ref=row['ref']))
                continue
            key = f"{b['cid']}|{side}"
            spent = state['spent'].get(key, 0.0)
            if fair - px >= EDGE and PMIN <= px <= PMAX and spent < MAXUSD:
                ckey = f"{key}|{px:.4f}"
                avail = depth - state.setdefault('consumed', {}).get(ckey, 0.0)
                shares = min(avail, (MAXUSD - spent) / px)
                if shares <= 0:
                    continue
                state['consumed'][ckey] = state['consumed'].get(ckey, 0.0) + shares
                if shares * px < 1:
                    continue
                state['spent'][key] = spent + shares * px
                pos = dict(ts=now, slug=u['slug'], city=u['city'], kind=u['kind'], cid=b['cid'], bucket=b['title'],
                           side=side, px=px, shares=shares, fee=FEE * px * (1 - px) * shares, p=p, fair=fair,
                           hr=f['hr'], src=u['src'], age=row['age'], model=os.environ.get('STACK_MODEL', 'prod'), v=2)
                state['positions'].append(pos); log('fills.jsonl', pos)
                print(time.strftime('%H:%M:%S'), 'PAPER BUY', side, u['slug'][:45], b['title'],
                      'px %.3f fair %.3f $%.1f' % (px, fair, shares * px), flush=True)
    return len(rows)


def settle():
    open_ = [p for p in state['positions'] if 'pnl' not in p]
    cids = list({p['cid'] for p in open_})
    for i in range(0, len(cids), 50):
        try:
            ms = get(f'{GAMMA}/markets', {'condition_ids': cids[i:i + 50], 'closed': 'true', 'limit': 50})
        except Exception:
            continue
        res = {m['conditionId']: jl(m.get('outcomePrices')) for m in ms if m.get('closed')}
        for p in open_:
            op = res.get(p['cid'])
            if op and len(op) == 2 and op[0] in ('0', '1'):
                y = float(op[0]); win = y if p['side'] == 'YES' else 1 - y
                p['win'] = win; p['pnl'] = p['shares'] * (win - p['px']) - p['fee']
                log('settled.jsonl', p)
                print(time.strftime('%H:%M:%S'), 'SETTLED', p['slug'][:40], p['bucket'], p['side'], 'pnl %.2f' % p['pnl'], flush=True)


if __name__ == '__main__':
    U, uts = [], 0
    while True:
        t0 = time.time()
        try:
            if time.time() - uts > 900:
                U = universe(); uts = time.time()
                print(time.strftime('%H:%M:%S'), 'universe events', len(U), flush=True)
            n = step(U)
            print(time.strftime('%H:%M:%S'), 'signals', n, 'open', sum(1 for p in state['positions'] if 'pnl' not in p), flush=True)
            if int(t0) % 900 < LOOP:
                settle()
            pickle.dump(state, open(state_path, 'wb'))
        except Exception:
            traceback.print_exc()
        time.sleep(max(1, LOOP - (time.time() - t0)))
