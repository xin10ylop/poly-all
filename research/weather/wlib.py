"""Weather market helpers: bucket parsing, event loading, METAR loading."""
import json, re, os, sys, io
import datetime as dt
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from stations import STATIONS

BUCKET_RE = re.compile(r'(-?\d+)(?:\s*-\s*(-?\d+))?\s*°\s*([CF])(?:\s+or\s+(below|higher|above|lower))?', re.I)


def parse_bucket(title):
    """'68-69°F' -> (68, 69); '57°F or below' -> (-inf, 57); '24°C' -> (24, 24)."""
    m = BUCKET_RE.search(title or '')
    if not m:
        return None
    a = int(m.group(1)); b = int(m.group(2)) if m.group(2) else a
    kind = (m.group(4) or '').lower()
    if kind in ('below', 'lower'):
        return (-999, b)
    if kind in ('higher', 'above'):
        return (a, 999)
    return (a, b)


def city_of(slug):
    s = re.sub(r'-on-.*', '', slug)
    return s.replace('highest-temperature-in-', '').replace('lowest-temperature-in-', '')


MONTHS = {m: i for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                                      'september', 'october', 'november', 'december'], 1)}


def obs_date(slug):
    m = re.search(r'-on-([a-z]+)-(\d+)-(\d{4})', slug)
    return dt.date(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))


def load_events(path='data/weather_events.jsonl'):
    out = []
    for l in open(path):
        e = json.loads(l)
        city = city_of(e['slug'])
        if city not in STATIONS:
            continue
        e['city'] = city
        e['kind'] = 'high' if e['slug'].startswith('highest') else 'low'
        e['date'] = obs_date(e['slug'])
        icao, unit, tz = STATIONS[city]
        e['icao'], e['unit'], e['tz'] = icao, unit, tz
        # local day window in UTC
        z = ZoneInfo(tz)
        d = e['date']
        e['day_start'] = dt.datetime(d.year, d.month, d.day, tzinfo=z).timestamp()
        e['day_end'] = (dt.datetime(d.year, d.month, d.day, tzinfo=z) + dt.timedelta(days=1)).timestamp()
        bk = []
        for m in e['markets']:
            rng = parse_bucket(m['title'] or m['q'])
            op = m.get('outcomePrices')
            won = None
            if op and len(op) == 2:
                try:
                    won = float(op[0]) > 0.5
                except Exception:
                    pass
            bk.append(dict(m, rng=rng, won=won))
        e['buckets'] = sorted([b for b in bk if b['rng']], key=lambda b: b['rng'][0])
        out.append(e)
    return out


def load_metar(icao, unit):
    path = f'data/metar/{icao}.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, na_values=['M'])
    df = df.dropna(subset=['tmpc'])
    df['ts'] = (pd.to_datetime(df['valid'], utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)
    if unit == 'F':
        df['t'] = np.round(df['tmpc'] * 9 / 5 + 32 + 1e-9).astype(int)
    else:
        df['t'] = np.round(df['tmpc'] + 1e-9).astype(int)
    return df[['ts', 't', 'tmpc', 'metar']].sort_values('ts').reset_index(drop=True)


def bucket_of(value, buckets):
    for i, b in enumerate(buckets):
        lo, hi = b['rng']
        if lo <= value <= hi:
            return i
    return None


def load_wtrades():
    """All weather taker prints: cid, ts, buy (taker side BUY), oi (0=YES token), px, sz, wallet; plus pyes and
    yes_buy (taker bought YES exposure)."""
    import glob
    parts = [pd.read_parquet(f) for f in sorted(glob.glob('data/wtrades/p*.parquet'))]
    for p in parts:
        p['cid'] = p.cid.astype(str); p['wallet'] = p.wallet.astype(str)
    df = pd.concat(parts, ignore_index=True)
    df['pyes'] = np.where(df.oi == 0, df.px, 1 - df.px)
    df['yes_buy'] = df.buy == (df.oi == 0)
    return df.sort_values(['cid', 'ts']).reset_index(drop=True)
