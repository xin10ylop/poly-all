"""Cross-market monotonicity ("nested ladder") arb scanner.

Within one event, markets that differ only in one numeric parameter (O/U line, spread line, price
threshold, "X+" threshold) or in a "by <date>" deadline are nested: event(hard) implies event(easy).
Holding YES(easy) + NO(hard) pays >= $1 in every state ($2 if easy happens and hard doesn't).
Arb if ask(YES_easy) + ask(NO_hard) + taker fees < 1 (walk both ladders, >= 5 shares).
Payout arrives at resolution (capital locked until then); this is NOT instantly mergeable.

Usage: python research/arb/ladders.py [minutes=60] [period_s=120]
Writes data/arb/run_<ts>_ladder/{ladders.json.gz, lad_opps.jsonl, lad_rechecks.jsonl, lad_sweeps.jsonl}
"""
import calendar, datetime, gzip, json, os, re, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor

from dateutil import parser as dparser

os.environ.setdefault('ARB_RUN', time.strftime('%Y%m%d%H%M', time.gmtime()) + '_ladder')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scanner as SC  # noqa: E402

MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 60
PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 120
SC.LIM.gap = 1.0 / 12  # keep this process at <= 12 req/s (the main scanner uses <= 30)
NUM = re.compile(r'(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?')
# bare "hit" is ambiguous (e.g. "approval rating hit 37%" is a *low*), so only "hit (HIGH)/(LOW)" count
DEC_KW = re.compile(r'↑|\breach\b|hit \(high\)|\babove\b|at least|or more|more than|greater than|\bexceed|higher than|≥|\d\+', re.I)
INC_KW = re.compile(r'↓|\bdip\b|hit \(low\)|\bbelow\b|less than|fewer than|or less|or fewer|lower than|≤|\bfall to\b', re.I)
MONTH = re.compile(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*$', re.I)


def parse_num(s):
    try:
        return float(s.replace(',', ''))
    except Exception:
        return None


def parse_date(s):
    if not s or not re.search(r'[A-Za-z]{3}', s) or not re.search(r'\d', s):
        return None
    try:
        return dparser.parse(s, fuzzy=False, default=dparser.parse('2026-12-31')).timestamp()
    except Exception:
        return None


def ladders_for_event(ev):
    live = [m for m in ev['markets'] if SC.is_live(m)]
    if len(live) < 2:
        return []
    out = {}
    for m in live:
        q, g = m.get('q') or '', m.get('git') or ''
        outs = tuple(m.get('outcomes') or [])
        # --- deadline ladders: groupItemTitle is a date and question says "by <date>" / "before <date>"
        d = parse_date(g)
        if d is not None and g in q and re.search(r'\b(by|before)\s+' + re.escape(g), q):
            # order by the deadline in the title; if the title omits the year take it from endDate
            # (endDate alone is unreliable: renamed/extended markets keep stale endDates)
            try:
                ed = datetime.datetime.strptime(m['endDate'][:10], '%Y-%m-%d')
                yr = (ed - datetime.timedelta(days=2)).year  # "Dec 31" deadlines end 04:59Z Jan 1
                d = dparser.parse(g, fuzzy=False, default=datetime.datetime(yr, 12, 31)).timestamp()
            except Exception:
                continue
            key = ('date', q.replace(g, '<D>'), outs)
            out.setdefault(key, []).append((d, m, 'inc'))
            continue
        # --- numeric ladders
        nums = NUM.findall(q)
        if not nums:
            continue
        tmpl = NUM.sub('#', q)
        low = q.lower()
        smt = (m.get('sportsMarketType') or '')
        if 'o/u' in low and outs[:1] == ('Over',):
            dirn = 'dec'
        elif 'spread' in smt or low.startswith('spread'):
            dirn = 'inc'  # signed line: Team(-2.5) harder than Team(-1.5)
        elif 'o/u' in low or 'odd' in low or 'even' in low:
            continue
        else:
            dk, ik = bool(DEC_KW.search(q + ' ' + g)), bool(INC_KW.search(q + ' ' + g))
            if dk == ik:
                continue
            dirn = 'dec' if dk else 'inc'
        key = ('num', tmpl, outs, dirn)
        out.setdefault(key, []).append((tuple(nums), m, dirn))
    lads = []
    for key, items in out.items():
        if len(items) < 2:
            continue
        if key[0] == 'date':
            pts = [(x, m) for x, m, _ in items]
            dirn = 'inc'
        else:
            dirn = key[3]
            L = {len(n) for n, _, _ in items}
            if len(L) != 1:
                continue
            # exactly one numeric position must vary
            var = [i for i in range(L.pop()) if len({n[i] for n, _, _ in items}) > 1]
            if len(var) != 1:
                continue
            vals = [parse_num(n[var[0]]) for n, _, _ in items]
            if all(v is not None and v == int(v) and 1990 <= v <= 2100 for v in vals):
                continue  # varying number is a year -> deadline, not a threshold
            q0 = items[0][1]['q']
            pos = list(NUM.finditer(q0))[var[0]].start()
            if MONTH.search(q0[:pos]):
                continue  # varying number is a day-of-month
            pts = [(parse_num(n[var[0]]), m) for n, m, _ in items]
            if any(x is None for x, _ in pts):
                continue
        if len({x for x, _ in pts}) != len(pts):
            continue
        # order easiest -> hardest
        pts.sort(key=lambda t: t[0], reverse=(dirn == 'inc'))
        lads.append(dict(eid=ev['id'], slug=ev['slug'], title=ev['title'], kind=key[0], tmpl=key[1][:140], dirn=dirn,
                         params=[x for x, _ in pts], markets=[m for _, m in pts]))
    return lads


def scan_ladder(L, bk, sweep):
    out = []
    ms = L['markets']
    if any(m['yes'] not in bk or m['no'] not in bk for m in ms):
        return out
    ay = [SC.top(bk[m['yes']][1], 'asks') for m in ms]
    an = [SC.top(bk[m['no']][1], 'asks') for m in ms]
    for i in range(len(ms)):          # easy
        for j in range(i + 1, len(ms)):  # hard
            if ay[i] is None or an[j] is None or ay[i] + an[j] >= 1.0:
                continue
            e, h = ms[i], ms[j]
            legs = [SC.mk_leg(e, bk[e['yes']][1], 'asks'), SC.mk_leg(h, bk[h['no']][1], 'asks')]
            tn, tg = SC.top_edge(legs, 'buy', lambda k: 1.0)
            r, _ = SC.evaluate(legs, 'buy', lambda k: 1.0, False)
            rec = dict(sweep=sweep, type='ladder', kind=L['kind'], eid=L['eid'], slug=L['slug'], title=L['title'],
                       tmpl=L['tmpl'], dirn=L['dirn'], easy=e['q'], hard=h['q'], easy_id=e['id'], hard_id=h['id'],
                       p_easy=L['params'][i], p_hard=L['params'][j], ask_yes_easy=ay[i], ask_no_hard=an[j],
                       rate=[e['rate'], h['rate']], top_net=tn, top_gross=tg, ts=time.time(),
                       fetch_ts=min(bk[e['yes']][0], bk[h['no']][0]), endDate=[e.get('endDate'), h.get('endDate')],
                       n_viol_pairs=None)
            if r is None or r['units'] <= 0:
                rec.update(units=0, profit=0.0, exec=False, fee_killed=True)
            else:
                rec.update(units=round(r['units'], 3), profit=round(r['profit'], 4), notional=round(r['notional'], 2),
                           exec=r['exec'], fee_killed=False, vwap=r['vwap'])
            out.append(rec)
    npairs = len(ms) * (len(ms) - 1) // 2
    for r in out:
        r['n_viol_pairs'] = len(out)
        r['n_pairs'] = npairs
    return out


def main():
    blob = json.load(open(os.path.join(SC.DATA, 'universe_latest.json')))
    lads = []
    for ev in blob['events']:
        if ev['negRisk']:
            continue
        try:
            lads.extend(ladders_for_event(ev))
        except Exception:
            traceback.print_exc()
    with gzip.open(os.path.join(SC.OUTDIR, 'ladders.json.gz'), 'wt') as f:
        json.dump([{k: v for k, v in L.items() if k != 'markets'} | {'mids': [m['id'] for m in L['markets']]}
                   for L in lads], f)
    toks = sorted({t for L in lads for m in L['markets'] for t in (m['yes'], m['no'])})
    SC.say('ladders', len(lads), 'markets', sum(len(L['markets']) for L in lads), 'tokens', len(toks))
    pool = ThreadPoolExecutor(4)
    rpool = ThreadPoolExecutor(2)
    tracked = {}
    t_end = time.time() + MINUTES * 60
    k = 0

    def recheck(key, L, rec):
        for d in SC.RECHECK_DELAYS:
            wait = rec['fetch_ts'] + d - time.time()
            if wait > 0:
                time.sleep(wait)
            mk = [m for m in L['markets'] if m['id'] in (rec['easy_id'], rec['hard_id'])]
            sub = dict(L, markets=mk, params=[L['params'][L['markets'].index(m)] for m in mk])
            t0 = time.time()
            bk = SC.fetch([t for m in mk for t in (m['yes'], m['no'])], rpool)
            res = scan_ladder(sub, bk, rec['sweep'])
            r = res[0] if res else None
            SC.log('lad_rechecks.jsonl', dict(key=key, delay=d, lag=round(t0 - rec['fetch_ts'], 2), ts=t0,
                                              present=bool(r and r['units'] > 0), exec=bool(r and r.get('exec')),
                                              units=r['units'] if r else 0, profit=r['profit'] if r else 0.0,
                                              top_net=r['top_net'] if r else None,
                                              orig_units=rec['units'], orig_profit=rec['profit']))

    while time.time() < t_end:
        c0 = time.time()
        sweep = f'lad{k}'
        bk = SC.fetch(toks, pool)
        t1 = time.time()
        opps = []
        for L in lads:
            opps.extend(scan_ladder(L, bk, sweep))
        t2 = time.time()
        for o in opps:
            SC.log('lad_opps.jsonl', o)
            key = f"{o['easy_id']}>{o['hard_id']}"
            if o['units'] > 0 and not o['fee_killed'] and (key not in tracked or time.time() - tracked[key] > 240):
                tracked[key] = time.time()
                L = next(x for x in lads if x['eid'] == o['eid'] and x['tmpl'] == o['tmpl'])
                threading.Thread(target=recheck, args=(key, L, o), daemon=True).start()
        SC.log('lad_sweeps.jsonl', dict(sweep=sweep, t0=c0, fetch_s=round(t1 - c0, 1), scan_s=round(t2 - t1, 1),
                                        n_tok=len(toks), n_books=len(bk), n_opp=len(opps),
                                        n_net=sum(not o['fee_killed'] for o in opps),
                                        n_exec=sum(o['exec'] for o in opps)))
        SC.say(sweep, f'fetch {t1-c0:.1f}s scan {t2-t1:.1f}s books {len(bk)}/{len(toks)} gross {len(opps)} '
                      f'net {sum(not o["fee_killed"] for o in opps)} exec {sum(o["exec"] for o in opps)}')
        del bk
        k += 1
        time.sleep(max(1.0, PERIOD - (time.time() - c0)))
    time.sleep(max(SC.RECHECK_DELAYS) + 10)


if __name__ == '__main__':
    main()
