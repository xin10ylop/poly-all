"""High-frequency poll of the negRisk events closest to the arb boundary.

Picks events whose NO-basket net edge (sum top YES bids - fees - 1) came within HOT_THR of zero in the
main scanner's nrstats, plus exhaustive-looking buy-all-YES candidates, and re-scans them every ~POLL s.
Logs every net-positive detection with timestamps so we can measure how long windows stay open.

Usage: python research/arb/hot.py <scanner_run_dir> [minutes=25] [poll_s=1.5] [thr=-0.015]
"""
import gzip, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ARGV = sys.argv[:]
sys.argv = sys.argv[:1]  # scanner parses argv at import
RUN_DIR = ARGV[1]
os.environ['ARB_RUN'] = os.path.basename(RUN_DIR.rstrip('/'))[4:] + '_hot'
sys.path.insert(0, HERE)
import scanner as SC  # noqa: E402

MIN = float(ARGV[2]) if len(ARGV) > 2 else 25
POLL = float(ARGV[3]) if len(ARGV) > 3 else 1.5
THR = float(ARGV[4]) if len(ARGV) > 4 else -0.015
SC.LIM.gap = 1.0 / 10

best = {}
with gzip.open(os.path.join(RUN_DIR, 'nrstats.jsonl.gz'), 'rt') as f:
    for line in f:
        r = json.loads(line)
        for x in r['rows']:
            gid, nb_net, maxbid = x[0], x[6] - 1, x[9]
            if maxbid is not None and maxbid >= 0.97:
                continue  # effectively decided (winner bid ~0.999): pinned at the price cap, not contested
            best[gid] = max(best.get(gid, -9), nb_net)
blob = json.load(open(os.path.join(SC.DATA, 'universe_latest.json')))
groups, _ = SC.build(blob)
hot = [g for gid, g in groups.items() if best.get(gid, -9) > THR]
SC.say('hot groups', len(hot), 'markets', sum(len(g['markets']) for g in hot))
SC.log('hot_groups.jsonl', dict(gids=[g['gid'] for g in hot], titles=[g['title'] for g in hot]))
toks = [t for g in hot for m in g['markets'] for t in (m['yes'], m['no'])]
pool = ThreadPoolExecutor(6)
t_end = time.time() + MIN * 60
n = 0
while time.time() < t_end:
    t0 = time.time()
    bk = SC.fetch(toks, pool)
    t1 = time.time()
    rows, stats = [], []
    for g in hot:
        o, st = SC.scan_group(g, bk, f'hot{n}', fetch_ts=t0)
        if st:
            stats.append(st[:9])
        for r in o:
            if not r['fee_killed'] and r['units'] > 0 and not (r['type'] == 'nr_buy_yes' and r['exh'] == 'no'):
                rows.append({k: v for k, v in r.items() if k != 'legs'})
    for r in rows:
        SC.log('hot_opps.jsonl', r)
    SC.log('hot_polls.jsonl', dict(i=n, t0=t0, fetch_s=round(t1 - t0, 2), n_books=len(bk), n_opp=len(rows),
                                   nb_net_max=max((s[6] - 1 for s in stats), default=None)))
    if n % 40 == 0:
        SC.say(f'hot{n} fetch {t1-t0:.2f}s books {len(bk)} opps {len(rows)}')
    n += 1
    time.sleep(max(0.05, POLL - (time.time() - t0)))
