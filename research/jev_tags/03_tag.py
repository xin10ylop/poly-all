"""Tag the executable universe with Jev structural questions (market text + creation/end dates only).
Input : data/jev_tags/to_tag.txt (market ids, priority order)
Output: data/jev_tags/tags.pkl  (mid -> {question: value})"""
import os, sys, pickle, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from jevq import *

CAP = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0   # stop once cumulative spend reaches this
OUT = os.path.join(ROOT, 'data', 'jev_tags', 'tags.pkl')
U = pd.read_pickle(os.path.join(ROOT, 'data', 'jev_tags', 'universe.pkl')).set_index('mid')
mids = [l.strip() for l in open(os.path.join(ROOT, 'data', 'jev_tags', 'to_tag.txt')) if l.strip()]
T = pickle.load(open(OUT, 'rb')) if os.path.exists(OUT) else {}
prev = load_spend()
print('prev spend', prev, 'todo', sum(m not in T for m in mids), flush=True)


def work(mid):
    if prev['usd'] + jev.spent['usd'] >= CAP:
        return mid, None
    try:
        a = jev.decide(state(U.loc[mid]), QUESTIONS)
    except Exception as ex:
        return mid, None
    return mid, {k: v.get('noul', v.get('score')) for k, v in a.items()} | {k + '_conf': v.get('confidence') for k, v in a.items()}


t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    for i, (mid, v) in enumerate(ex.map(work, [m for m in mids if m not in T])):
        if v is not None:
            T[mid] = v
        if i % 1000 == 0:
            json.dump({'usd': prev['usd'] + jev.spent['usd'], 'calls': prev['calls'] + jev.spent['calls']}, open(SPEND, 'w'))
            print(i, round(time.time() - t0), 'spent this run', round(jev.spent['usd'], 4), flush=True)
            pickle.dump(T, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT)
pickle.dump(T, open(OUT + '.tmp', 'wb')); os.replace(OUT + '.tmp', OUT)
tot = save_spend(prev)
print('done', len(T), 'cumulative spend', tot)
