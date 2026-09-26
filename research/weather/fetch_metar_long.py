"""Long METAR history (2023-06-01 .. 2026-06-18) for nowcast climatology."""
import os, sys, time, requests
sys.path.insert(0, os.path.dirname(__file__))
from stations import STATIONS
os.makedirs('data/metar_long', exist_ok=True)
for city, (icao, unit, tz) in STATIONS.items():
    path = f'data/metar_long/{icao}.csv'
    if os.path.exists(path):
        continue
    url = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=%s&data=tmpc'
           '&year1=2023&month1=6&day1=1&year2=2026&month2=6&day2=18&tz=Etc/UTC&format=onlycomma'
           '&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4') % icao
    for attempt in range(8):
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 200 and not r.text.startswith('Too many'):
                break
        except Exception as ex:
            print('err', ex)
        time.sleep(15 * (attempt + 1))
    open(path, 'w').write(r.text)
    print(city, icao, len(r.text.splitlines()), flush=True)
    time.sleep(8)
