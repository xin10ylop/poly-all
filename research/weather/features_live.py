"""Compute nowcast features at an arbitrary time t from a METAR frame (same definitions as nowcast.py)."""
import numpy as np, datetime as dt
from zoneinfo import ZoneInfo
from stations import STATIONS


def features_at(df, city, kind, t, lat=300):
    """df: columns ts (unix), t (resolution-unit integer), tf (float in unit). Returns dict or None."""
    icao, unit, tz = STATIONS[city]
    z = ZoneInfo(tz)
    d = dt.datetime.fromtimestamp(t, z).date()
    s = dt.datetime(d.year, d.month, d.day, tzinfo=z).timestamp()
    sign = 1 if kind == 'high' else -1
    ts = df.ts.values
    avail = ts + lat <= t
    day = (ts >= s) & avail & (ts < s + 86400)
    if day.sum() == 0:
        return None
    dts = ts[day]; dv = df.t.values[day] * sign; dfv = df.tf.values[day] * sign
    run = dv.max(); j = len(dv) - 1
    prev = (ts >= s - 86400) & (ts < s)
    prev_ext = (df.t.values[prev] * sign).max() if prev.sum() > 8 else np.nan
    j1 = np.searchsorted(dts, dts[j] - 3600, side='right') - 1
    j3 = np.searchsorted(dts, dts[j] - 3 * 3600, side='right') - 1
    doy = d.timetuple().tm_yday
    return dict(city=city, kind=kind, date=d.isoformat(), t=t, hr=(t - s) / 3600.0, doy=doy, run=run, cur=dv[j],
                curf=dfv[j], gap=run - dv[j], tr1=dfv[j] - dfv[j1] if j1 >= 0 else np.nan,
                tr3=dfv[j] - dfv[j3] if j3 >= 0 else np.nan, age=(t - dts[j]) / 60.0,
                prevgap=prev_ext - run, unitF=int(unit == 'F'), D=0)
