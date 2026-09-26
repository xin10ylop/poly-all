"""Bulk NBM (NBS) MOS archive from IEM for US stations: txn/xnd forecasts per runtime."""
import os, time, requests
ST = ['KLGA', 'KATL', 'KMIA', 'KHOU', 'KDAL', 'KORD', 'KAUS', 'KBKF', 'KSEA', 'KLAX', 'KSFO', 'CYYZ']
os.makedirs('data/nbm', exist_ok=True)
for st in ST:
    p = f'data/nbm/{st}.csv'
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        continue
    url = f'https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?station={st}&model=NBS&sts=2026-06-10T00:00Z&ets=2026-09-27T00:00Z&format=csv'
    for a in range(6):
        try:
            r = requests.get(url, timeout=300)
            if r.status_code == 200 and r.text.startswith('runtime'):
                open(p, 'w').write(r.text); print(st, len(r.text.splitlines()), flush=True); break
            print(st, 'status', r.status_code, r.text[:100], flush=True)
        except Exception as ex:
            print(st, 'err', ex, flush=True)
        time.sleep(20 * (a + 1))
    time.sleep(10)
