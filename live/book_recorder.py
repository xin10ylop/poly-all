"""Poll CLOB order books for a universe of markets and store top-of-book snapshots.

Usage: python live/book_recorder.py weather [interval_sec]
Writes data/books/<universe>/YYYYMMDDHH.jsonl.gz, one line per snapshot round:
  {"ts": ..., "b": {token: [[bid_px, bid_sz]x5, [ask_px, ask_sz]x5]}}
"""
import gzip, json, os, sys, time, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from polylib import *

UNIVERSE = sys.argv[1] if len(sys.argv) > 1 else 'weather'
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 60
OUT = f'data/books/{UNIVERSE}'
os.makedirs(OUT, exist_ok=True)


def load_universe():
    meta = {}
    if UNIVERSE == 'weather':
        evs = gamma_paginate('events', {'tag_slug': 'weather', 'active': 'true', 'closed': 'false'}, page=100)
        for e in evs:
            if 'temperature' not in e['slug']:
                continue
            for m in e['markets']:
                toks = jl(m.get('clobTokenIds'))
                if toks:
                    meta[toks[0]] = dict(slug=e['slug'], title=m.get('groupItemTitle'), cid=m['conditionId'],
                                         no=toks[1], tick=m.get('orderPriceMinTickSize'),
                                         rewMin=m.get('rewardsMinSize'), rewSpread=m.get('rewardsMaxSpread'),
                                         rew=sum(float(r.get('rewardsDailyRate') or 0) for r in (m.get('clobRewards') or [])))
    return meta


def top(levels, n=5, reverse=False):
    lv = [(float(x['price']), float(x['size'])) for x in levels]
    lv.sort(key=lambda x: x[0], reverse=reverse)
    return lv[:n]


meta, meta_ts = {}, 0
while True:
    t0 = time.time()
    try:
        if time.time() - meta_ts > 1800:
            meta = load_universe(); meta_ts = time.time()
            with open(f'{OUT}/meta_{time.strftime("%Y%m%d%H%M", time.gmtime())}.json', 'w') as f:
                json.dump(meta, f)
            print(time.strftime('%H:%M:%S'), 'universe', len(meta), flush=True)
        toks = list(meta)
        snap = {}
        for i in range(0, len(toks), 400):
            for bk in books(toks[i:i + 400]):
                snap[bk['asset_id']] = [top(bk.get('bids', []), reverse=True), top(bk.get('asks', []))]
        fn = f'{OUT}/{time.strftime("%Y%m%d%H", time.gmtime())}.jsonl.gz'
        with gzip.open(fn, 'at') as f:
            f.write(json.dumps({'ts': round(t0, 1), 'b': snap}) + '\n')
    except Exception:
        traceback.print_exc()
    time.sleep(max(1, INTERVAL - (time.time() - t0)))
