"""Polymarket executable-arbitrage scanner (negRisk baskets + binary YES/NO pairs).

Usage: python research/arb/scanner.py [minutes=90] [nr_period_s=30] [bin_every=4]

Each cycle:
  * nr sweep : YES+NO books of every live market in every negRisk event (~115k tokens, ~290 POSTs)
  * bin sweep: YES+NO books of every live non-negRisk market, every `bin_every` cycles
Opportunities (net of per-share taker fees, walking the ask/bid ladders, min 5 shares per leg)
are logged to data/arb/run_<ts>/opps.jsonl; each newly-seen opportunity is re-fetched
5/15/30/60/180 s after the snapshot to measure persistence (rechecks.jsonl).

Arb types
  bin_buy     ask(YES)+ask(NO)+fees < 1                -> buy both, merge for $1
  bin_sell    bid(YES)+bid(NO)-fees > 1                -> split $1, sell both
  nr_buy_yes  sum_i ask(YES_i)+fees < 1                -> needs exhaustive outcome set (flagged)
  nr_buy_no   sum_{i in S} ask(NO_i)+fees < |S|-1      -> riskless (negRisk: at most one YES);
              realise instantly: convert one NO -> YES of others, merge pairs
  nr_sell_yes sum_{i in S} bid(YES_i)-fees > 1          -> split & sell YES (== buy NO via conversion)
"""
import calendar, gzip, heapq, json, os, re, sys, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor

import orjson

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'src'))
sys.path.insert(0, HERE)
import polylib  # noqa: E402
import universe  # noqa: E402

MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 90
NR_PERIOD = float(sys.argv[2]) if len(sys.argv) > 2 else 30
BIN_EVERY = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MIN_SZ = 5.0
MAX_RPS = 30.0
RECHECK_DELAYS = [5, 15, 30, 60, 180]
UNIVERSE_TTL = 1800

DATA = os.path.join(HERE, '..', '..', 'data', 'arb')
RUN = os.environ.get('ARB_RUN') or time.strftime('%Y%m%d%H%M', time.gmtime())
OUTDIR = os.path.join(DATA, f'run_{RUN}')
os.makedirs(OUTDIR, exist_ok=True)

_wlock = threading.Lock()


def log(name, rec):
    with _wlock:
        with open(os.path.join(OUTDIR, name), 'ab') as f:
            f.write(orjson.dumps(rec) + b'\n')


def say(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- rate-limited fetching
class Limiter:
    def __init__(self, rps):
        self.gap, self.next, self.lock, self.n = 1.0 / rps, 0.0, threading.Lock(), 0

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.gap
            self.n += 1
        if t > now:
            time.sleep(t - now)


LIM = Limiter(MAX_RPS)
S = polylib._s


def fetch_chunk(ch):
    err = None
    for attempt in range(5):
        LIM.wait()
        try:
            r = S.post(polylib.CLOB + '/books', data=orjson.dumps([{'token_id': t} for t in ch]),
                       headers={'Content-Type': 'application/json'}, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return time.time(), orjson.loads(r.content)
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(1 + attempt)
    say('fetch failed', repr(err))
    return time.time(), []


def fetch(tokens, pool, chunk=400):
    chunks = [tokens[i:i + chunk] for i in range(0, len(tokens), chunk)]
    out = {}
    for t, res in pool.map(fetch_chunk, chunks):
        for b in res:
            out[b['asset_id']] = (t, b)
    return out


def ladder(book, side):
    """side 'asks' -> ascending price (best first); 'bids' -> descending."""
    lv = [(float(x['price']), float(x['size'])) for x in book.get(side) or []]
    lv = [x for x in lv if x[1] > 0]
    lv.sort(reverse=(side == 'bids'))
    return lv


def top(book, side):
    lv = book.get(side) or []
    if not lv:
        return None
    ps = [float(x['price']) for x in lv]
    return min(ps) if side == 'asks' else max(ps)


def fee(p, rate, exp):
    if not rate:
        return 0.0
    x = p * (1.0 - p)
    return rate * (x if exp in (1, None) else x ** exp)


# ---------------------------------------------------------------- basket walking
def walk(legs, kind, payout, allow_drop):
    """legs: list of dict(lad=[(p,s)...] best-first, rate, exp).
    kind 'buy': per-unit profit = payout(k) - sum(p+fee); 'sell': sum(p-fee) - payout(k).
    Returns dict(units, profit, notional, fills, vwap) walking until marginal profit <= 0."""
    n = len(legs)
    idx = [0] * n
    rem = [lg['lad'][0][1] if lg['lad'] else 0.0 for lg in legs]
    active = [i for i in range(n) if legs[i]['lad']]
    if not allow_drop and len(active) < n:
        return None
    fills, notional_leg = [0.0] * n, [0.0] * n
    units = profit = notional = 0.0
    steps = 0
    while active and steps < 100000:
        steps += 1
        px = {i: legs[i]['lad'][idx[i]][0] for i in active}
        if kind == 'buy':
            c = sum(p + fee(p, legs[i]['rate'], legs[i]['exp']) for i, p in px.items())
            pu = payout(len(active)) - c
        else:
            c = sum(p - fee(p, legs[i]['rate'], legs[i]['exp']) for i, p in px.items())
            pu = c - payout(len(active))
        if pu <= 1e-12:
            break
        q = min(rem[i] for i in active)
        units += q
        profit += q * pu
        notional += q * sum(px.values())
        stop = False
        for i in list(active):
            fills[i] += q
            notional_leg[i] += q * px[i]
            rem[i] -= q
            if rem[i] <= 1e-9:
                idx[i] += 1
                if idx[i] < len(legs[i]['lad']):
                    rem[i] = legs[i]['lad'][idx[i]][1]
                elif allow_drop:
                    active.remove(i)
                else:
                    stop = True
        if stop:
            break
    return dict(units=units, profit=profit, notional=notional, fills=fills,
                vwap=[(notional_leg[i] / fills[i]) if fills[i] else None for i in range(n)])


def top_edge(legs, kind, payout):
    """Per-unit edge at top of book (net, gross) over legs that have a top level."""
    act = [lg for lg in legs if lg['lad']]
    if not act:
        return None, None
    ps = [(lg['lad'][0][0], lg['rate'], lg['exp']) for lg in act]
    k = len(act)
    if kind == 'buy':
        g = payout(k) - sum(p for p, _, _ in ps)
        n_ = g - sum(fee(p, r, e) for p, r, e in ps)
    else:
        g = sum(p for p, _, _ in ps) - payout(k)
        n_ = g - sum(fee(p, r, e) for p, r, e in ps)
    return n_, g


def evaluate(legs, kind, payout, allow_drop):
    """Walk the basket, enforcing >= MIN_SZ shares on every leg that trades."""
    excl = set()
    for _ in range(4):
        use = [lg for j, lg in enumerate(legs) if j not in excl]
        if not use:
            return None, excl
        r = walk(use, kind, payout, allow_drop)
        if r is None:
            return None, excl
        small = [j for j, f in zip([j for j in range(len(legs)) if j not in excl], r['fills']) if 0 < f < MIN_SZ]
        if not small or not allow_drop or r['units'] < MIN_SZ:
            r['exec'] = r['units'] >= MIN_SZ and not small
            r['used'] = [j for j in range(len(legs)) if j not in excl]
            return r, excl
        excl.update(small)
    r['exec'] = False
    r['used'] = [j for j in range(len(legs)) if j not in excl]
    return r, excl


# ---------------------------------------------------------------- universe
def is_live(m):
    return bool(m['active'] and not m['closed'] and m['accepting'] and m['yes'] and m['no']
                and m.get('enableOrderBook', True) is not False and not m.get('archived'))


def resolved(m):
    op = m.get('op') or []
    try:
        op = [float(x) for x in op]
    except Exception:
        return None
    if m['closed'] and len(op) == 2:
        if op[0] >= 0.99:
            return 'YES'
        if op[1] >= 0.99:
            return 'NO'
    return None


RANGE_PAT = re.compile(r'(or (below|less|lower|under|fewer|more|above|higher|greater|over))|(\bless than\b|\bmore than\b|\bunder\b|\bover\b|<|>|≤|≥|\+$)', re.I)
OTHER_PAT = re.compile(r'^(other|others|none|no one|nobody|neither|draw|tie|another|field|no winner|none of the above)\b', re.I)


def classify_group(ev, ms_all, live):
    names = [(m.get('git') or m.get('q') or '') for m in live]
    res = [resolved(m) for m in ms_all]
    missing = [m for m, r in zip(ms_all, res) if not is_live(m) and r != 'NO']
    flags = []
    if 'YES' in res:
        flags.append('resolved_yes')
    if ev.get('negRiskAugmented'):
        flags.append('augmented')
    if missing:
        flags.append(f'missing{len(missing)}')
    has_other = any(OTHER_PAT.search(x.strip()) for x in names)
    has_range = sum(bool(RANGE_PAT.search(x)) for x in names) >= 2
    if has_other:
        flags.append('has_other')
    if has_range:
        flags.append('range_buckets')
    if 'resolved_yes' in flags or 'augmented' in flags or missing:
        exh = 'no'
    elif has_other or has_range:
        exh = 'likely'
    else:
        exh = 'unknown'
    return exh, flags


def build(blob):
    now = time.time()
    groups, bins = {}, []
    for ev in blob['events']:
        ms_all = ev['markets']
        if ev['negRisk']:
            by = {}
            for m in ms_all:
                by.setdefault(m.get('nrid') or ev.get('negRiskMarketID') or ev['id'], []).append(m)
            for nrid, grp in by.items():
                live = [m for m in grp if is_live(m)]
                if len(live) < 2:
                    bins.extend(m | {'_slug': ev['slug'], '_title': ev['title']} for m in live)
                    continue
                exh, flags = classify_group(ev, grp, live)
                past = False
                try:
                    past = bool(ev.get('endDate')) and calendar.timegm(time.strptime(ev['endDate'][:19], '%Y-%m-%dT%H:%M:%S')) < now
                except Exception:
                    pass
                if past:
                    flags.append('past_end')
                umas = sorted({m.get('uma') for m in live if m.get('uma')})
                gid = f"{ev['id']}:{nrid[:10] if nrid else ''}"
                groups[gid] = dict(gid=gid, eid=ev['id'], slug=ev['slug'], title=ev['title'], exh=exh, flags=flags,
                                   umas=umas, tags=ev.get('tags', [])[:6], n_all=len(grp), vol24=ev.get('volume24hr'),
                                   markets=live)
        else:
            bins.extend(m | {'_slug': ev['slug'], '_title': ev['title']} for m in ms_all if is_live(m))
    for g in groups.values():
        for m in g['markets']:
            m['_slug'], m['_title'] = g['slug'], g['title']
    return groups, bins


# ---------------------------------------------------------------- scanning
def mk_leg(m, book, side):
    return dict(lad=ladder(book, side) if book else [], rate=m['rate'] or 0.0, exp=m.get('exp') or 1,
                name=(m.get('git') or m.get('q') or '')[:60], mid=m['id'])


def scan_group(g, bk, sweep, fetch_ts=None, record=True):
    """Return list of opportunity dicts for a negRisk group given books dict token->(t, book)."""
    out = []
    ms = g['markets']
    have = [m for m in ms if m['yes'] in bk and m['no'] in bk]
    if len(have) < len(ms):
        return out, None
    yb = [bk[m['yes']][1] for m in ms]
    nb = [bk[m['no']][1] for m in ms]
    fts = [bk[m['yes']][0] for m in ms]
    # near-miss stats (top of book)
    ay = [top(b, 'asks') for b in yb]
    byb = [top(b, 'bids') for b in yb]
    an = [top(b, 'asks') for b in nb]
    rates = [(m['rate'] or 0.0, m.get('exp') or 1) for m in ms]
    s_ay = sum(p for p in ay if p is not None)
    s_ayf = sum(p + fee(p, *r) for p, r in zip(ay, rates) if p is not None)
    s_by = sum(p for p in byb if p is not None)
    s_byf = sum(p - fee(p, *r) for p, r in zip(byb, rates) if p is not None)
    k_an = [p for p in an if p is not None]
    s_anf = sum(p + fee(p, *r) for p, r in zip(an, rates) if p is not None)
    stat = [g['gid'], len(ms), round(s_ay, 4), round(s_ayf, 4), sum(p is None for p in ay),
            round(s_by, 4), round(s_byf, 4), sum(p is None for p in byb),
            round(len(k_an) - 1 - s_anf, 4), max([p for p in byb if p is not None], default=None)]
    ctx = dict(sweep=sweep, gid=g['gid'], eid=g['eid'], slug=g['slug'], title=g['title'], N=len(ms), n_all=g['n_all'],
               exh=g['exh'], flags=g['flags'], umas=g['umas'], fetch_ts=fetch_ts or min(fts),
               max_yes_bid=stat[-1])
    checks = []
    if all(p is not None for p in ay) and s_ay < 1.0:
        checks.append(('nr_buy_yes', [mk_leg(m, b, 'asks') for m, b in zip(ms, yb)], 'buy', lambda k: 1.0, False))
    if len(k_an) >= 2 and sum(k_an) < len(k_an) - 1:
        checks.append(('nr_buy_no', [mk_leg(m, b, 'asks') for m, b in zip(ms, nb)], 'buy', lambda k: k - 1.0, True))
    if s_by > 1.0:
        checks.append(('nr_sell_yes', [mk_leg(m, b, 'bids') for m, b in zip(ms, yb)], 'sell', lambda k: 1.0, True))
    for typ, legs, kind, payout, drop in checks:
        tn, tg = top_edge(legs, kind, payout)
        r, excl = evaluate(legs, kind, payout, drop)
        rec = dict(ctx, type=typ, top_net=tn, top_gross=tg, ts=time.time())
        if r is None or r['units'] <= 0:
            rec.update(units=0, profit=0.0, exec=False, fee_killed=True)
        else:
            rec.update(units=round(r['units'], 3), profit=round(r['profit'], 4), notional=round(r['notional'], 2),
                       exec=r['exec'], fee_killed=False,
                       legs=[dict(name=legs[j]['name'], mid=legs[j]['mid'], top=legs[j]['lad'][0][0] if legs[j]['lad'] else None,
                                  top_sz=legs[j]['lad'][0][1] if legs[j]['lad'] else None,
                                  fill=round(f, 3), vwap=round(v, 5) if v else None)
                             for j, f, v in zip(r['used'], r['fills'], r['vwap'])])
        out.append(rec)
    return out, stat


def scan_binary(m, bk, sweep):
    out = []
    if m['yes'] not in bk or m['no'] not in bk:
        return out
    (ty, yb), (tn_, nb) = bk[m['yes']], bk[m['no']]
    ay, an, by, bn = top(yb, 'asks'), top(nb, 'asks'), top(yb, 'bids'), top(nb, 'bids')
    ctx = dict(sweep=sweep, mid=m['id'], cid=m['cid'], slug=m.get('_slug'), title=m.get('_title'), q=m.get('q'),
               negRisk=m.get('negRisk'), fetch_ts=min(ty, tn_), uma=m.get('uma'), endDate=m.get('endDate'),
               book_ts=[yb.get('timestamp'), nb.get('timestamp')])
    checks = []
    if ay is not None and an is not None and ay + an < 1.0:
        checks.append(('bin_buy', 'asks', 'buy'))
    if by is not None and bn is not None and by + bn > 1.0:
        checks.append(('bin_sell', 'bids', 'sell'))
    for typ, side, kind in checks:
        legs = [mk_leg(m, yb, side), mk_leg(m, nb, side)]
        tn, tg = top_edge(legs, kind, lambda k: 1.0)
        r, _ = evaluate(legs, kind, lambda k: 1.0, False)
        rec = dict(ctx, type=typ, top_net=tn, top_gross=tg, ts=time.time(), tops=[ay, an, by, bn])
        if r is None or r['units'] <= 0:
            rec.update(units=0, profit=0.0, exec=False, fee_killed=True)
        else:
            rec.update(units=round(r['units'], 3), profit=round(r['profit'], 4), notional=round(r['notional'], 2),
                       exec=r['exec'], fee_killed=False, fills=r['fills'], vwap=r['vwap'])
        out.append(rec)
    return out


# ---------------------------------------------------------------- rechecks
class Rechecker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.heap, self.lock, self.seq = [], threading.Lock(), 0
        self.tracked = {}  # key -> first detection ts
        self.pool = ThreadPoolExecutor(4)

    def schedule(self, key, opp, obj, is_group):
        with self.lock:
            if key in self.tracked and time.time() - self.tracked[key] < max(RECHECK_DELAYS) + 30:
                return False
            self.tracked[key] = time.time()
            base = opp['fetch_ts']
            for d in RECHECK_DELAYS:
                self.seq += 1
                heapq.heappush(self.heap, (base + d, self.seq, d, key, opp, obj, is_group))
            return True

    def run(self):
        while True:
            with self.lock:
                item = self.heap[0] if self.heap else None
                if item and item[0] <= time.time():
                    heapq.heappop(self.heap)
                else:
                    item = None
            if not item:
                time.sleep(0.2)
                continue
            try:
                self.do(item)
            except Exception:
                traceback.print_exc()

    def do(self, item):
        due, _, d, key, opp, obj, is_group = item
        toks = [t for m in (obj['markets'] if is_group else [obj]) for t in (m['yes'], m['no'])]
        t0 = time.time()
        bk = fetch(toks, self.pool)
        if is_group:
            res, _ = scan_group(obj, bk, opp['sweep'], fetch_ts=t0)
        else:
            res = scan_binary(obj, bk, opp['sweep'])
        same = [r for r in res if r['type'] == opp['type']]
        r = same[0] if same else None
        log('rechecks.jsonl', dict(key=key, type=opp['type'], delay=d, lag=round(t0 - opp['fetch_ts'], 2), ts=t0,
                                   present=bool(r and r['units'] > 0), exec=bool(r and r.get('exec')),
                                   units=r['units'] if r else 0, profit=r['profit'] if r else 0.0,
                                   top_net=r['top_net'] if r else None, orig_units=opp['units'], orig_profit=opp['profit']))


# ---------------------------------------------------------------- main loop
SLIM_KEYS = ('id', 'cid', 'q', 'git', 'yes', 'no', 'rate', 'exp', 'negRisk', 'uma', 'endDate', '_slug', '_title')


def slim(groups, bins):
    for g in groups.values():
        g['markets'] = [{k: m.get(k) for k in SLIM_KEYS} for m in g['markets']]
    return groups, [{k: m.get(k) for k in SLIM_KEYS} for m in bins]


def build_to_pickle(out):
    """Run in a subprocess: pull universe, build groups/bins, pickle a slim copy (bounds parent memory)."""
    import pickle
    blob = universe.pull()
    g, b = slim(*build(blob))
    with open(out, 'wb') as f:
        pickle.dump((g, b), f)


def load_universe():
    import pickle, subprocess
    out = os.path.join(OUTDIR, 'universe_slim.pkl')
    code = (f"import sys; sys.argv=['x']; sys.path.insert(0, {HERE!r}); import scanner; "
            f"scanner.build_to_pickle({out!r})")
    subprocess.run([sys.executable, '-c', code], check=True, env=dict(os.environ, ARB_RUN=RUN))
    with open(out, 'rb') as f:
        g, b = pickle.load(f)
    meta = {gid: {k: v for k, v in x.items() if k != 'markets'} | {'names': [m.get('git') for m in x['markets']]}
            for gid, x in g.items()}
    with gzip.open(os.path.join(OUTDIR, f'groups_{int(time.time())}.json.gz'), 'wt') as f:
        json.dump(meta, f)
    say('universe', len(g), 'negRisk groups', sum(len(x['markets']) for x in g.values()), 'nr markets',
        len(b), 'binary markets')
    return g, b


def pack(items, ntok, limit=400):
    out, cur, n = [], [], 0
    for it in items:
        k = ntok(it)
        if cur and n + k > limit:
            out.append(cur)
            cur, n = [], 0
        cur.append(it)
        n += k
    if cur:
        out.append(cur)
    return out


def nr_chunk(args):
    gs, sweep = args
    toks = [t for g in gs for m in g['markets'] for t in (m['yes'], m['no'])]
    t, res = fetch_chunk(toks)
    bk = {b['asset_id']: (t, b) for b in res}
    opps, stats = [], []
    for g in gs:
        o, st = scan_group(g, bk, sweep)
        opps.extend(o)
        if st:
            stats.append(st)
        for m in g['markets']:
            opps.extend(scan_binary(m, bk, sweep))
    return opps, stats, len(bk), len(toks)


def bin_chunk(args):
    ms, sweep = args
    toks = [t for m in ms for t in (m['yes'], m['no'])]
    t, res = fetch_chunk(toks)
    bk = {b['asset_id']: (t, b) for b in res}
    opps = []
    for m in ms:
        opps.extend(scan_binary(m, bk, sweep))
    return opps, len(bk), len(toks)


def main():
    t_end = time.time() + MINUTES * 60
    pool = ThreadPoolExecutor(8)
    rc = Rechecker()
    rc.start()
    uni = {'loading': False}

    def reload():
        uni['loading'] = True
        try:
            g, b = load_universe()
            uni.update(groups=g, bins=b, ts=time.time())
        except Exception:
            traceback.print_exc()
            uni['ts'] = time.time()
        finally:
            uni['loading'] = False

    reload()
    cycle = 0
    while time.time() < t_end:
        c0 = time.time()
        if time.time() - uni['ts'] > UNIVERSE_TTL and not uni['loading']:
            threading.Thread(target=reload, daemon=True).start()
        groups, bins = uni['groups'], uni['bins']
        # ---- negRisk sweep (whole groups packed into <=400-token requests, scanned as they arrive)
        sweep = f'nr{cycle}'
        r0 = LIM.n
        chunks = pack(list(groups.values()), lambda g: 2 * len(g['markets']))
        opps, stats, nb, nt = [], [], 0, 0
        for o, st, a, b in pool.map(nr_chunk, [(c, sweep) for c in chunks]):
            opps.extend(o)
            stats.extend(st)
            nb += a
            nt += b
        t1 = time.time()
        with gzip.open(os.path.join(OUTDIR, 'nrstats.jsonl.gz'), 'ab') as f:
            f.write(orjson.dumps(dict(sweep=sweep, ts=c0, rows=stats)) + b'\n')
        handle(opps, sweep, groups, rc, is_group=True)
        log('sweeps.jsonl', dict(sweep=sweep, kind='nr', t0=c0, fetch_s=round(t1 - c0, 1), scan_s=0,
                                 n_tok=nt, n_books=nb, n_req=LIM.n - r0, n_groups=len(groups),
                                 n_opp=len(opps), n_exec=sum(o.get('exec', False) for o in opps),
                                 by_type=count(opps)))
        say(sweep, f'fetch+scan {t1-c0:.1f}s books {nb}/{nt} opps {len(opps)} '
                   f'exec {sum(o.get("exec", False) for o in opps)} {count(opps)}')
        # ---- binary sweep
        if cycle % BIN_EVERY == 0:
            sweep_b = f'bin{cycle}'
            b0 = time.time()
            r0 = LIM.n
            opps, nb, nt = [], 0, 0
            for o, a, b in pool.map(bin_chunk, [(c, sweep_b) for c in pack(bins, lambda m: 2)]):
                opps.extend(o)
                nb += a
                nt += b
            b1 = time.time()
            handle(opps, sweep_b, None, rc, is_group=False, bins={m['id']: m for m in bins})
            log('sweeps.jsonl', dict(sweep=sweep_b, kind='bin', t0=b0, fetch_s=round(b1 - b0, 1), scan_s=0,
                                     n_tok=nt, n_books=nb, n_req=LIM.n - r0, n_opp=len(opps),
                                     n_exec=sum(o.get('exec', False) for o in opps), by_type=count(opps)))
            say(sweep_b, f'fetch+scan {b1-b0:.1f}s books {nb}/{nt} opps {len(opps)} '
                         f'exec {sum(o.get("exec", False) for o in opps)} {count(opps)}')
        cycle += 1
        time.sleep(max(1.0, NR_PERIOD - (time.time() - c0)))
    say('done; waiting for pending rechecks')
    time.sleep(max(RECHECK_DELAYS) + 20)


def count(opps):
    c = {}
    for o in opps:
        k = o['type'] + (':exec' if o.get('exec') else (':net' if not o.get('fee_killed') else ':gross'))
        c[k] = c.get(k, 0) + 1
    return c


def handle(opps, sweep, groups, rc, is_group, bins=None):
    for o in opps:
        nonexh = o['type'] == 'nr_buy_yes' and o.get('exh') == 'no'
        if nonexh:  # known non-exhaustive outcome set: not an arb; log compactly, no rechecks
            log('opps_nonexh.jsonl', {k: v for k, v in o.items() if k != 'legs'})
            continue
        log('opps.jsonl', o)
        if o.get('units', 0) > 0 and not o.get('fee_killed'):
            if 'gid' in o:
                key = f"{o['type']}|{o['gid']}"
                obj, grp = groups[o['gid']], True
            else:
                key = f"{o['type']}|{o['mid']}"
                obj, grp = (bins or {}).get(o['mid']), False
                if obj is None and groups:
                    obj = next((m for g in groups.values() for m in g['markets'] if m['id'] == o['mid']), None)
                if obj is None:
                    continue
            rc.schedule(key, o, obj, grp)


if __name__ == '__main__':
    main()
