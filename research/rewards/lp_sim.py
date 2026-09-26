"""Simulate a small two-sided liquidity-rewards quoter on recorded 1-minute books.

Each minute, for each rewarded market: quote bid/ask at distance S (cents) from the size-adjusted mid (clamped
inside the spread, never crossing), size = min qualifying size. Reward share per Polymarket formula vs the other
resting orders. Fill when the next snapshot's opposite best price trades through our quote (conservative).
Inventory capped; P&L marked to resolution when known, else last mid.
"""
import gzip, json, glob, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from reward_share import side_q, combine

UNI = os.environ.get('UNI', 'weather')
S = float(os.environ.get('S', 2.0))          # target distance from mid, cents
INV = float(os.environ.get('INV', 2))        # max inventory in units of min size per side
HOURS_CUTOFF = float(os.environ.get('CUT', 0))  # stop quoting this many hours before endDate (0=never)


def adj_mid(bids, asks, minsize):
    bb = next((p for p, s in bids if s >= minsize), bids[0][0] if bids else None)
    ba = next((p for p, s in asks if s >= minsize), asks[0][0] if asks else None)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2


def main():
    metas = sorted(glob.glob(f'data/books/{UNI}/meta_*.json'))
    meta = {}
    for m in metas:
        meta.update(json.load(open(m)))
    snaps = []
    for f in sorted(glob.glob(f'data/books/{UNI}/*.jsonl.gz')):
        for l in gzip.open(f, 'rt'):
            try:
                snaps.append(json.loads(l))
            except Exception:
                pass
    print('snapshots', len(snaps), 'hours %.1f' % ((snaps[-1]['ts'] - snaps[0]['ts']) / 3600))
    state = {}
    reward = 0.0
    per_mkt = {}
    for k in range(len(snaps) - 1):
        cur, nxt = snaps[k]['b'], snaps[k + 1]['b']
        for tok, (bids, asks) in cur.items():
            m = meta.get(tok)
            if not m or m['rew'] <= 0 or not bids or not asks:
                continue
            tick = float(m['tick'] or 0.01); v = float(m['rewSpread'] or 4.5); ms = float(m['rewMin'] or 20)
            mid = adj_mid(bids, asks, ms)
            if mid is None:
                continue
            bb, ba = bids[0][0], asks[0][0]
            st = state.setdefault(tok, dict(inv=0.0, cash=0.0, fills=0, rew=0.0))
            ob = np.floor((mid - S / 100) / tick + 1e-9) * tick
            oa = np.ceil((mid + S / 100) / tick - 1e-9) * tick
            ob = min(ob, ba - tick); oa = max(oa, bb + tick)
            qb_on = st['inv'] < INV * ms and ob > 0
            qa_on = st['inv'] > -INV * ms and oa < 1
            q1 = side_q([(ob, ms)], mid, v, ms) if qb_on else 0.0
            q2 = side_q([(oa, ms)], mid, v, ms) if qa_on else 0.0
            ours = combine(q1, q2, mid)
            others = combine(side_q(bids, mid, v, ms), side_q(asks, mid, v, ms), mid)
            if ours > 0:
                r = m['rew'] / 1440 * ours / (ours + others)
                st['rew'] += r; reward += r
            nb = nxt.get(tok)
            if not nb or not nb[0] or not nb[1]:
                continue
            nbb, nba = nb[0][0][0], nb[1][0][0]
            if qb_on and nba <= ob:          # market traded down through our bid -> we bought YES at ob
                st['inv'] += ms; st['cash'] -= ms * ob; st['fills'] += 1
            if qa_on and nbb >= oa:          # traded up through our ask -> we sold YES at oa
                st['inv'] -= ms; st['cash'] += ms * oa; st['fills'] += 1
            st['last_mid'] = (nbb + nba) / 2
    # mark to market
    tot_mtm = 0.0; fills = 0
    for tok, st in state.items():
        mark = st.get('last_mid', 0.5)
        if mark > 0.97: mark = 1.0
        if mark < 0.03: mark = 0.0
        st['mtm'] = st['cash'] + st['inv'] * mark
        tot_mtm += st['mtm']; fills += st['fills']
    hours = (snaps[-1]['ts'] - snaps[0]['ts']) / 3600
    print(f'S={S}c INV={INV}: markets quoted {len(state)}  rewards ${reward:.1f} ({reward / hours * 24:.0f}/day)  '
          f'fills {fills}  trading mtm ${tot_mtm:.1f} ({tot_mtm / hours * 24:.0f}/day)')
    df = pd.DataFrame([dict(tok=t, slug=meta[t]['slug'], title=meta[t]['title'], **{k: v for k, v in s.items()}) for t, s in state.items()])
    df.to_pickle(f'data/lp_sim_{UNI}_S{S}.pkl')
    # capital: max simultaneous quote notional ~ sum over markets of ms*(ob + 1-oa) ~ ms per market
    print('approx capital needed $%.0f' % sum(float(meta[t]['rewMin'] or 20) for t in state))


if __name__ == '__main__':
    main()
