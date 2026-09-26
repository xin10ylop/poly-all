"""Race public METAR sources: poll each every ~15 s and record when each new observation first appears."""
import time, json, requests, gzip, io, csv, datetime as dt
ST = ['KLGA', 'KORD', 'EGLC', 'LFPB', 'EDDM', 'RKSI', 'ZSPD', 'RJTT', 'SBGR', 'LTFM']
seen = {}   # (source, station, obsTime) -> first_seen
S = requests.Session()


def rec(src, st, obs):
    k = (src, st, int(obs))
    if k not in seen:
        seen[k] = time.time()


def awc_api():
    for x in S.get('https://aviationweather.gov/api/data/metar', params={'ids': ','.join(ST), 'format': 'json'}, timeout=20).json():
        rec('awc_api', x['icaoId'], x['obsTime'])


def awc_cache():
    r = S.get('https://aviationweather.gov/data/cache/metars.cache.csv.gz', timeout=30)
    txt = gzip.decompress(r.content).decode()
    for row in csv.reader(io.StringIO(txt)):
        if len(row) > 2 and row[1] in ST:
            rec('awc_cache', row[1], dt.datetime.fromisoformat(row[2].replace('Z', '+00:00')).timestamp())


def tgftp():
    for st in ST:
        t = S.get(f'https://tgftp.nws.noaa.gov/data/observations/metar/stations/{st}.TXT', timeout=15).text.splitlines()[0].strip()
        rec('tgftp', st, dt.datetime.strptime(t, '%Y/%m/%d %H:%M').replace(tzinfo=dt.timezone.utc).timestamp())


def iem():
    for st in ST:
        j = S.get('https://mesonet.agron.iastate.edu/api/1/currents.json', params={'station': st if not st.startswith('K') else st[1:]}, timeout=15).json()
        for x in j.get('data', []):
            if x.get('utc_valid'):
                rec('iem', st, dt.datetime.fromisoformat(x['utc_valid'].replace('Z', '+00:00')).timestamp())


end = time.time() + float(__import__('sys').argv[1]) * 60
while time.time() < end:
    for f in (awc_api, awc_cache, tgftp, iem):
        try:
            f()
        except Exception as ex:
            pass
    time.sleep(15)
rows = [dict(src=k[0], st=k[1], obs=k[2], seen=v, lag_min=(v - k[2]) / 60) for k, v in seen.items()]
json.dump(rows, open('data/feed_race.json', 'w'))
import pandas as pd
d = pd.DataFrame(rows)
start = min(seen.values()) + 60                   # only observations first seen after we started polling
d = d[d.groupby(['st', 'obs']).seen.transform('min') > start]
print(d.groupby('src').lag_min.describe().round(2))
