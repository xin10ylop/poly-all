"""Fetch taker prints in the 12h window after each horizon (hd = 1, 3, 7 days before scheduled end).
Output: data/jev_tags/win_trades.pkl  {(mid, hd): [(ts, side, outcomeIndex, price, size), ...]}
Reuses data/longshot_trades.pkl when that cache provably covers the window."""
import os, sys, pickle, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
from polylib import trades

HDS = (1, 3, 7)
WIN = 12 * 3600
OUT = os.path.join(ROOT, 'data', 'jev_tags', 'win_trades.pkl')
U = pd.read_pickle(os.path.join(ROOT, 'data', 'jev_tags', 'universe.pkl'))
LT = pickle.load(open(os.path.join(ROOT, 'data', 'longshot_trades.pkl'), 'rb'))
S = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}


def windows(r):
    for hd in HDS:
        t0 = r.end - hd * 86400
        if t0 >= r.closed - 3600 or t0 < r.created:
            continue
        yield hd, int(t0), int(min(t0 + WIN, r.closed))


jobs, reused = [], 0
for r in U.itertuples():
    for hd, t0, t1 in windows(r):
        if (r.mid, hd) in S:
            continue
        lt = LT.get(r.mid)
        if lt and min(x[0] for x in lt) < t0:
            S[(r.mid, hd)] = [x for x in lt if t0 <= x[0] < t1]
            reused += 1
            continue
        jobs.append((r.mid, r.cid, hd, t0, t1))
print('reused', reused, 'jobs', len(jobs), flush=True)


def work(j):
    mid, cid, hd, t0, t1 = j
    out, off = [], 0
    try:
        while off <= 3000:
            t = trades(market=cid, limit=500, offset=off, start=t0, end=t1)
            out += [(x['timestamp'], x['side'], x['outcomeIndex'], float(x['price']), float(x['size'])) for x in t]
            if len(t) < 500:
                break
            off += 500
        return (mid, hd), [x for x in out if t0 <= x[0] < t1]
    except Exception as ex:
        return (mid, hd), None


t_start = time.time()
with ThreadPoolExecutor(8) as ex:
    for i, (k, v) in enumerate(ex.map(work, jobs)):
        if v is not None:
            S[k] = v
        if i % 5000 == 0:
            print(i, round(time.time() - t_start), flush=True)
            pickle.dump(S, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT)
pickle.dump(S, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT)
print('done', len(S), 'nonempty', sum(1 for v in S.values() if v))
