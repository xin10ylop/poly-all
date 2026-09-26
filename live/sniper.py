"""Minute-level 'dead bucket' sniper (paper). Polls NOAA's raw METAR server every POLL seconds; when a new
observation pushes the day's running max above a bucket (or running min below it), that bucket can no longer win,
so we buy NO at the displayed price (= 1 - best YES bid), limited to displayed depth and MAXUSD per bucket.
NOAA-resolved cities only (METAR == resolution source); unreliable cities excluded.
Backtest (research/weather/bt_dead.py, NOAA cities since Aug 24): ~$25/day at 5-min reaction, 97% positive days."""
import os, sys, re, json, time, pickle, traceback, datetime as dt
from zoneinfo import ZoneInfo
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, os.path.join(ROOT, 'research', 'weather')); sys.path.insert(0, os.path.dirname(__file__))
from polylib import *
from stations import STATIONS
from wlib import parse_bucket, city_of, obs_date
import rules_guard

POLL = 15; MAXUSD = float(os.environ.get('SNIPE_MAXUSD', 200)); PMIN = 0.01; FEE = 0.05
EXCL = {'moscow', 'manila', 'panama-city', 'hong-kong'}
OUT = os.path.join(ROOT, 'data', 'live', 'sniper'); os.makedirs(OUT, exist_ok=True)
state_path = os.path.join(OUT, 'state.pkl')
state = pickle.load(open(state_path, 'rb')) if os.path.exists(state_path) else {'positions': [], 'spent': {}}
OBS = {}          # icao -> {obs_ts: temp_c}
S = requests.Session() if False else None


def log(name, rec):
    with open(os.path.join(OUT, name), 'a') as f:
        f.write(json.dumps(rec, default=float) + '\n')


def parse_metar_temp(raw):
    """Temperature in C: T-group tenths (TsTTTsDDD) if present, else main TT/DD group."""
    m = re.search(r'\bT([01])(\d{3})([01])(\d{3})\b', raw)
    if m:
        return (-1 if m.group(1) == '1' else 1) * int(m.group(2)) / 10.0
    m = re.search(r'\s(M?\d{2})/(M?\d{2}|//)\s', ' ' + raw + ' ')
    if m:
        v = m.group(1)
        return -int(v[1:]) if v.startswith('M') else int(v)
    return None


def to_unit(c, unit):
    return round(c * 9 / 5 + 32 + 1e-9) if unit == 'F' else round(c + 1e-9)


def universe():
    evs = gamma_paginate('events', {'tag_slug': 'weather', 'active': 'true', 'closed': 'false'}, page=100)
    U = []
    for e in evs:
        if 'temperature' not in e['slug']:
            continue
        city = city_of(e['slug'])
        desc = e.get('description') or ''
        if city not in STATIONS or city in EXCL or 'weather.gov/wrh/timeseries' not in desc:
            continue
        try:
            d = obs_date(e['slug'])
        except Exception:
            continue
        icao, unit, tz = STATIONS[city]
        z = ZoneInfo(tz)
        if dt.datetime.now(z).date() != d:
            continue
        kind = 'high' if e['slug'].startswith('highest') else 'low'
        if not rules_guard.approve(e['slug'], desc, kind, allow_llm=False):
            continue
        bks = []
        for m in e['markets']:
            rng = parse_bucket(m.get('groupItemTitle') or m['question']); toks = jl(m.get('clobTokenIds'))
            if rng and toks:
                bks.append(dict(rng=rng, cid=m['conditionId'], yes=toks[0], no=toks[1], title=m.get('groupItemTitle')))
        U.append(dict(slug=e['slug'], city=city, kind=kind, icao=icao, unit=unit,
                      day_start=dt.datetime(d.year, d.month, d.day, tzinfo=z).timestamp(), buckets=bks))
    return U


def backfill(icao):
    for x in get('https://aviationweather.gov/api/data/metar', {'ids': icao, 'format': 'json', 'hours': 30}):
        if x.get('rawOb'):
            t = parse_metar_temp(x['rawOb'])
            if t is not None:
                OBS.setdefault(icao, {})[int(x['obsTime'])] = t


def poll(icao):
    """Latest METAR from NOAA tgftp; returns obs_ts if new."""
    r = requests.get(f'https://tgftp.nws.noaa.gov/data/observations/metar/stations/{icao}.TXT', timeout=10)
    lines = r.text.strip().splitlines()
    ts = dt.datetime.strptime(lines[0].strip(), '%Y/%m/%d %H:%M').replace(tzinfo=dt.timezone.utc).timestamp()
    t = parse_metar_temp(lines[1])
    if t is None or int(ts) in OBS.get(icao, {}):
        return None
    OBS.setdefault(icao, {})[int(ts)] = t
    return int(ts)


def check(u, obs_ts):
    day = {ts: to_unit(c, u['unit']) for ts, c in OBS.get(u['icao'], {}).items() if u['day_start'] <= ts < u['day_start'] + 86400}
    if not day:
        return
    run = max(day.values()) if u['kind'] == 'high' else min(day.values())
    dead = [b for b in u['buckets'] if (b['rng'][1] < run if u['kind'] == 'high' else b['rng'][0] > run)]
    if not dead:
        return
    bk = {x['asset_id']: x for x in books([b['yes'] for b in dead])}
    now = time.time()
    for b in dead:
        x = bk.get(b['yes'])
        if not x:
            continue
        bids = sorted([(float(l['price']), float(l['size'])) for l in x.get('bids', [])], reverse=True)
        key = b['cid']
        for pyes, size in bids:
            if pyes < PMIN or state['spent'].get(key, 0) >= MAXUSD:
                break
            px = 1 - pyes                                  # NO price
            shares = min(size, (MAXUSD - state['spent'].get(key, 0)) / px)
            if shares < 5:
                continue
            state['spent'][key] = state['spent'].get(key, 0) + shares * px
            pos = dict(ts=now, obs_ts=obs_ts, react_min=(now - obs_ts) / 60, slug=u['slug'], cid=b['cid'], bucket=b['title'],
                       side='NO', px=px, shares=shares, fee=FEE * px * (1 - px) * shares, run=run)
            state['positions'].append(pos); log('fills.jsonl', pos)
            print(time.strftime('%H:%M:%S'), 'SNIPE NO', u['slug'][:45], b['title'], 'run', run, 'px %.3f $%.0f react %.1fmin' % (px, shares * px, pos['react_min']), flush=True)


def settle():
    open_ = [p for p in state['positions'] if 'pnl' not in p]
    cids = list({p['cid'] for p in open_})
    for i in range(0, len(cids), 50):
        try:
            ms = get(f'{GAMMA}/markets', {'condition_ids': cids[i:i + 50], 'closed': 'true', 'limit': 50})
        except Exception:
            continue
        res = {m['conditionId']: jl(m.get('outcomePrices')) for m in ms}
        for p in open_:
            op = res.get(p['cid'])
            if op and op[0] in ('0', '1'):
                win = 1 - float(op[0]); p['pnl'] = p['shares'] * (win - p['px']) - p['fee']; p['win'] = win
                log('settled.jsonl', p)


if __name__ == '__main__':
    U, uts, last_settle = [], 0, 0
    while True:
        t0 = time.time()
        try:
            if time.time() - uts > 900:
                U = universe(); uts = time.time()
                for icao in {u['icao'] for u in U} - set(OBS):
                    backfill(icao)
                print(time.strftime('%H:%M:%S'), 'sniper universe', len(U), 'events', flush=True)
            for icao in sorted({u['icao'] for u in U}):
                try:
                    new = poll(icao)
                except Exception:
                    continue
                if new:
                    for u in U:
                        if u['icao'] == icao:
                            check(u, new)
            if time.time() - last_settle > 900:
                settle(); last_settle = time.time()
            pickle.dump(state, open(state_path, 'wb'))
        except Exception:
            traceback.print_exc()
        time.sleep(max(1, POLL - (time.time() - t0)))
