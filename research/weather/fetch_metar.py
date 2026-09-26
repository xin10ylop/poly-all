"""Download METAR archive from IEM for all stations (rate-limited)."""
import os, sys, time, io
import requests, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from stations import STATIONS
y1, m1, d1 = 2026, 6, 18
y2, m2, d2 = map(int, (sys.argv[1] if len(sys.argv) > 1 else '2026-09-27').split('-'))
os.makedirs('data/metar', exist_ok=True)
for city, (icao, unit, tz) in STATIONS.items():
    path = f'data/metar/{icao}.csv'
    if os.path.exists(path) and len(sys.argv) <= 2:
        continue
    url = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=%s&data=tmpc&data=metar'
           '&year1=%d&month1=%d&day1=%d&year2=%d&month2=%d&day2=%d&tz=Etc/UTC&format=onlycomma'
           '&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4') % (icao, y1, m1, d1, y2, m2, d2)
    for attempt in range(6):
        r = requests.get(url, timeout=120)
        if r.status_code == 200 and not r.text.startswith('Too many'):
            break
        time.sleep(10 * (attempt + 1))
    open(path, 'w').write(r.text)
    print(city, icao, len(r.text.splitlines()), flush=True)
    time.sleep(6)
