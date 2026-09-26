"""Estimate liquidity-reward capture for a small two-sided quote in each rewarded market.

Scoring (docs.polymarket.com/programs/liquidity-rewards): S = ((v - s)/v)^2 * size for orders within v cents
of the (size-adjusted) midpoint. Two-sided combination Q = max(min(Q1,Q2), max(Q1,Q2)/3) if 0.10<=mid<=0.90
else min(Q1,Q2). Rewards per day split pro-rata by Q across makers (sampled each minute).
"""
import gzip, json, glob, sys
import numpy as np


def side_q(levels, mid, v, minsize):
    q = 0.0
    for px, sz in levels:
        s = abs(px - mid) * 100
        if s < v and sz >= minsize:
            q += ((v - s) / v) ** 2 * sz
    return q


def combine(q1, q2, mid):
    if 0.10 <= mid <= 0.90:
        return max(min(q1, q2), max(q1, q2) / 3)
    return min(q1, q2)


def estimate(bids, asks, rate, v, minsize, tick, our_size=None, improve=True):
    if not bids or not asks:
        return None
    bb, ba = bids[0][0], asks[0][0]
    mid = (bb + ba) / 2
    our_size = our_size or minsize
    q_b = side_q(bids, mid, v, minsize); q_a = side_q(asks, mid, v, minsize)
    others = combine(q_b, q_a, mid)
    # our quote: join best bid/ask (or improve by one tick if spread allows)
    ob = bb + tick if improve and ba - bb > 2 * tick else bb
    oa = ba - tick if improve and ba - bb > 2 * tick else ba
    qb = side_q([(ob, our_size)], mid, v, minsize); qa = side_q([(oa, our_size)], mid, v, minsize)
    ours = combine(qb, qa, mid)
    share = ours / (ours + others) if ours + others > 0 else 0
    capital = our_size * ob + our_size * (1 - oa)
    return dict(mid=mid, spread=ba - bb, others=others, ours=ours, share=share, usd_day=share * rate,
                capital=capital, bid=ob, ask=oa)


if __name__ == '__main__':
    uni = sys.argv[1] if len(sys.argv) > 1 else 'weather'
    meta = json.load(open(sorted(glob.glob(f'data/books/{uni}/meta_*.json'))[-1]))
    last = None
    for f in sorted(glob.glob(f'data/books/{uni}/*.jsonl.gz')):
        for l in gzip.open(f, 'rt'):
            last = l
    snap = json.loads(last)['b']
    res = []
    for tok, m in meta.items():
        if m['rew'] <= 0 or tok not in snap:
            continue
        bids, asks = snap[tok]
        r = estimate(bids, asks, m['rew'], float(m['rewSpread'] or 4.5), float(m['rewMin'] or 20),
                     float(m['tick'] or 0.01))
        if r:
            r.update(slug=m['slug'], title=m['title'], rate=m['rew'])
            res.append(r)
    res.sort(key=lambda r: -r['usd_day'] / max(r['capital'], 1))
    tot = sum(r['usd_day'] for r in res); cap = sum(r['capital'] for r in res)
    print(f'markets {len(res)}  est reward/day with min-size quotes everywhere: ${tot:.0f} on ${cap:.0f} capital')
    for r in res[:25]:
        print('%-55s %-10s rate %5.0f mid %.3f spr %.3f share %.3f $/day %6.2f cap %5.1f' % (
            r['slug'][:55], r['title'], r['rate'], r['mid'], r['spread'], r['share'], r['usd_day'], r['capital']))
