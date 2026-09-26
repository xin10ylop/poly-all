"""Resume weather trade fetch; writes parquet shards data/wtrades/pN_XXX.parquet (compact)."""
import json, sys, os, glob, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

min_end = sys.argv[1] if len(sys.argv) > 1 else '2026-06-20'
evs = [json.loads(l) for l in open('data/weather_events.jsonl')]
evs = [e for e in evs if e['endDate'] >= min_end]
done = set()
for f in glob.glob('data/wtrades/done*.parquet'):
    done |= set(pd.read_parquet(f).cid)
jobs = [m['cid'] for e in evs for m in e['markets'] if m['cid'] not in done and m['vol'] > 0]
print('jobs', len(jobs), flush=True)
tag = time.strftime('%H%M%S')


def work(cid):
    allt, off = [], 0
    try:
        while True:
            t = trades(market=cid, limit=500, offset=off)
            allt += [(cid, x['timestamp'], x['side'] == 'BUY', x['outcomeIndex'], float(x['price']), float(x['size']),
                      x['proxyWallet']) for x in t]
            if len(t) < 500 or off >= 9500:
                break
            off += 500
        return cid, allt
    except Exception:
        return cid, None


buf, bufc, part = [], [], 0
with ThreadPoolExecutor(16) as ex:
    for i, (cid, t) in enumerate(ex.map(work, jobs)):
        if t is not None:
            buf += t; bufc.append(cid)
        if len(bufc) >= 3000 or i == len(jobs) - 1:
            df = pd.DataFrame(buf, columns=['cid', 'ts', 'buy', 'oi', 'px', 'sz', 'wallet'])
            df['cid'] = df.cid.astype('category'); df['wallet'] = df.wallet.astype('category')
            df.to_parquet(f'data/wtrades/p{tag}_{part:03d}.parquet')
            pd.Series(bufc).to_frame('cid').to_parquet(f'data/wtrades/done{tag}_{part:03d}.parquet')
            print(i + 1, flush=True); buf, bufc = [], []; part += 1
print('done')
