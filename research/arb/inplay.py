"""High-frequency probe of nested-ladder violations in sports games that are in play right now.

Selects ladders (O/U lines, spreads, team totals, props) whose markets' endDate (= scheduled start for
sports markets) lies in [now - 4h, now + 10min], and polls their books every ~POLL seconds, logging every
executable violation (ask YES_easy + ask NO_hard + fees < 1) with timestamps to measure window lifetimes.

Usage: python research/arb/inplay.py [minutes=25] [poll_s=2]
"""
import calendar, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

ARGV = sys.argv[:]
sys.argv = sys.argv[:1]
os.environ['ARB_RUN'] = time.strftime('%Y%m%d%H%M', time.gmtime()) + '_inplay'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scanner as SC  # noqa: E402
import ladders as LD  # noqa: E402

MIN = float(ARGV[1]) if len(ARGV) > 1 else 25
POLL = float(ARGV[2]) if len(ARGV) > 2 else 2.0
SC.LIM.gap = 1.0 / 15


def ts(s):
    try:
        return calendar.timegm(time.strptime(s[:19], '%Y-%m-%dT%H:%M:%S'))
    except Exception:
        return None


def select():
    blob = SC.universe.pull()
    now = time.time()
    lads = []
    for ev in blob['events']:
        if ev['negRisk']:
            continue
        if not any(m.get('sportsMarketType') for m in ev['markets']):
            continue
        for L in LD.ladders_for_event(ev):
            t = ts(L['markets'][0].get('endDate') or '')
            if t and now - 4 * 3600 <= t <= now + 600:
                lads.append(L)
    return lads


lads = select()
toks = sorted({t for L in lads for m in L['markets'] for t in (m['yes'], m['no'])})
SC.say('in-play ladders', len(lads), 'events', len({L['eid'] for L in lads}), 'tokens', len(toks))
SC.log('inplay_ladders.jsonl', dict(n=len(lads), events=sorted({L['title'] for L in lads})))
pool = ThreadPoolExecutor(6)
t_end = time.time() + MIN * 60
i = 0
while time.time() < t_end:
    t0 = time.time()
    bk = SC.fetch(toks, pool)
    t1 = time.time()
    n_net = n_exec = 0
    for L in lads:
        for o in LD.scan_ladder(L, bk, f'ip{i}'):
            if not o['fee_killed']:
                n_net += 1
                n_exec += o['exec']
                SC.log('inplay_opps.jsonl', o)
    SC.log('inplay_polls.jsonl', dict(i=i, t0=t0, fetch_s=round(t1 - t0, 2), n_books=len(bk), n_net=n_net, n_exec=n_exec))
    if i % 30 == 0:
        SC.say(f'ip{i} fetch {t1-t0:.2f}s books {len(bk)}/{len(toks)} net {n_net} exec {n_exec}')
    i += 1
    time.sleep(max(0.05, POLL - (time.time() - t0)))
