"""Empirical nowcast of the final daily max/min given observations so far.

For every station-day and every 10-minute step we compute features from METAR obs available
up to that time and the realized final extreme. Target: D = |final - running_extreme| (>=0, integer
in resolution units). A gradient-boosted classifier estimates P(D = k), k = 0..K (K = ">= K").
"""
import os, sys
import numpy as np, pandas as pd
import datetime as dt
from zoneinfo import ZoneInfo
sys.path.insert(0, os.path.dirname(__file__))
from stations import STATIONS
from wlib import load_metar

STEP = 600
KMAX = 8


def load_any(icao, unit):
    frames = []
    for d in ('data/metar_long', 'data/metar'):
        p = f'{d}/{icao}.csv'
        if os.path.exists(p):
            df = pd.read_csv(p, na_values=['M'], usecols=['station', 'valid', 'tmpc'])
            frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames).dropna(subset=['tmpc']).drop_duplicates(subset=['valid'])
    df['ts'] = (pd.to_datetime(df['valid'], utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)
    df['t'] = np.round(df['tmpc'] * 9 / 5 + 32 + 1e-9).astype(int) if unit == 'F' else np.round(df['tmpc'] + 1e-9).astype(int)
    df['tf'] = df['tmpc'] * 9 / 5 + 32 if unit == 'F' else df['tmpc']
    return df[['ts', 't', 'tf']].sort_values('ts').reset_index(drop=True)


def station_days(city, kind, lat=300):
    """Yield feature rows for every day and 10-min step (local day window)."""
    icao, unit, tz = STATIONS[city]
    df = load_any(icao, unit)
    if df is None or len(df) < 1000:
        return None
    z = ZoneInfo(tz)
    ts, tv, tf = df.ts.values, df.t.values, df.tf.values
    first = dt.datetime.fromtimestamp(ts[0], z).date() + dt.timedelta(days=1)
    last = dt.datetime.fromtimestamp(ts[-1], z).date() - dt.timedelta(days=1)
    rows = []
    d = first
    sign = 1 if kind == 'high' else -1
    while d <= last:
        s = dt.datetime(d.year, d.month, d.day, tzinfo=z).timestamp()
        e = (dt.datetime(d.year, d.month, d.day, tzinfo=z) + dt.timedelta(days=1)).timestamp()
        i0, i1 = np.searchsorted(ts, s), np.searchsorted(ts, e)
        if i1 - i0 >= 16:
            dts, dv, dfv = ts[i0:i1], tv[i0:i1] * sign, tf[i0:i1] * sign
            final = dv.max()
            run = np.maximum.accumulate(dv)
            grid = np.arange(s + STEP, e, STEP)
            oi = np.searchsorted(dts + lat, grid, side='right') - 1
            # previous-day stats (prior info available at start of day)
            p0, p1 = np.searchsorted(ts, s - 86400), i0
            prev_ext = (tv[p0:p1] * sign).max() if p1 - p0 > 8 else np.nan
            doy = d.timetuple().tm_yday
            ok = oi >= 0
            g = grid[ok]; j = oi[ok]
            j1 = np.searchsorted(dts, dts[j] - 3600, side='right') - 1
            j3 = np.searchsorted(dts, dts[j] - 3 * 3600, side='right') - 1
            tr1 = np.where(j1 >= 0, dfv[j] - dfv[np.maximum(j1, 0)], np.nan)
            tr3 = np.where(j3 >= 0, dfv[j] - dfv[np.maximum(j3, 0)], np.nan)
            n = len(g)
            rows.append(pd.DataFrame(dict(city=city, kind=kind, date=d.isoformat(), t=g, hr=(g - s) / 3600.0,
                                          doy=doy, run=run[j], cur=dv[j], curf=dfv[j], gap=run[j] - dv[j],
                                          tr1=tr1, tr3=tr3, age=(g - dts[j]) / 60.0,
                                          prevgap=prev_ext - run[j], D=final - run[j])))
        d += dt.timedelta(days=1)
    out = pd.concat(rows, ignore_index=True)
    out['unitF'] = int(unit == 'F')
    return out


if __name__ == '__main__':
    kind = sys.argv[1]
    os.makedirs('data/nowcast', exist_ok=True)
    for city in STATIONS:
        path = f'data/nowcast/{kind}_{city}.parquet'
        if os.path.exists(path):
            continue
        out = station_days(city, kind)
        if out is None:
            print('skip', city, flush=True); continue
        out.to_parquet(path)
        print(city, kind, len(out), flush=True)
