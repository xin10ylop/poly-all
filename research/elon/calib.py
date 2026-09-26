"""Calibrate the Elon nowcast (NB size r, rate weights) on resolved windows; report calibration by time-left."""
import json, re, sys, os, itertools
import numpy as np, pandas as pd, datetime as dt
sys.path.insert(0, os.path.dirname(__file__))
from model import load_posts, Nowcast


def brackets(e):
    out = []
    for m in e['markets']:
        t = (m['title'] or m['q'] or '').replace(',', '')
        nums = [int(x) for x in re.findall(r'\d+', t)]
        if not nums:
            continue
        if t.strip().startswith('<') or 'less than' in t.lower():
            lo, hi = 0, nums[0] - 1
        elif t.strip().endswith('+') or 'or more' in t.lower():
            lo, hi = nums[0], 10**6
        elif len(nums) >= 2:
            lo, hi = nums[0], nums[1]
        else:
            continue
        won = bool(m['outcomePrices']) and m['outcomePrices'][0] in ('1', '1.0')
        out.append(dict(lo=lo, hi=hi, cid=m['cid'], won=won, tok=m['toks'][0] if m['toks'] else None, title=m['title']))
    return sorted(out, key=lambda b: b['lo'])


def windows():
    ev = {e['slug']: e for e in json.load(open('data/elon_events.json'))}
    W = []
    for slug, s, e in json.load(open('data/elon_windows.json')):
        s = dt.datetime.fromisoformat(s).timestamp(); e = dt.datetime.fromisoformat(e).timestamp()
        bk = brackets(ev[slug])
        if bk and sum(b['won'] for b in bk) == 1:
            W.append(dict(slug=slug, s=s, e=e, bk=bk, end=ev[slug]['end']))
    return W


def evaluate(nc, W, fracs=(0.1, 0.3, 0.5, 0.7, 0.85, 0.95)):
    rows = []
    for w in W:
        for f in fracs:
            t = w['s'] + f * (w['e'] - w['s'])
            if t < nc.ct.min() + 8 * 86400:
                continue
            C, mu, q = nc.bucket_probs(w['s'], t, w['e'], [(b['lo'], b['hi']) for b in w['bk']])
            k = [i for i, b in enumerate(w['bk']) if b['won']][0]
            rows.append(dict(slug=w['slug'], f=f, hours_left=(w['e'] - t) / 3600, C=C, mu=mu, qwin=q[k],
                             ll=np.log(max(q[k], 1e-6)), qmax=q.max(), argmax_hit=int(q.argmax() == k), end=w['end']))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    P = load_posts(); W = windows()
    print('windows', len(W))
    best = None
    for r, wts in itertools.product([4, 8, 12, 20, 40], [(0.3, 0.3, 0.4), (0.5, 0.3, 0.2), (0.1, 0.3, 0.6), (0.0, 0.5, 0.5)]):
        nc = Nowcast(P, *wts, r=r)
        d = evaluate(nc, W)
        tr = d[d.end < '2026-06-01']
        print(r, wts, 'train ll %.4f  all ll %.4f  hit %.3f' % (tr.ll.mean(), d.ll.mean(), d.argmax_hit.mean()), flush=True)
        if best is None or tr.ll.mean() > best[0]:
            best = (tr.ll.mean(), r, wts)
    print('best', best)
    nc = Nowcast(P, *best[2], r=best[1])
    d = evaluate(nc, W)
    d.to_pickle('data/elon_calib.pkl')
    print(d.groupby('f').agg(n=('ll', 'size'), ll=('ll', 'mean'), qwin=('qwin', 'mean'), hit=('argmax_hit', 'mean')).round(3))
