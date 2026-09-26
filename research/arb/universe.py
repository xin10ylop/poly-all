"""Pull all active, open Gamma events and build the arb-scanner universe.

Writes data/arb/universe_latest.json with a compact structure:
  events: [{id, slug, title, negRisk, negRiskAugmented, enableNegRisk, tags, n_markets_total,
            markets: [{id, cid, q, git, yes, no, rate, tick, minsz, active, closed, accepting,
                       nro (negRiskOther), endDate, umaStatus, outcomePrices, bestBid, bestAsk}]}]
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

OUT = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'arb')
os.makedirs(OUT, exist_ok=True)


def fnum(x, d=None):
    try:
        return float(x)
    except Exception:
        return d


def compact_market(m):
    toks = jl(m.get('clobTokenIds')) or []
    fs = m.get('feeSchedule') or {}
    return dict(
        id=m.get('id'), cid=m.get('conditionId'), q=m.get('question'), git=m.get('groupItemTitle'),
        yes=toks[0] if len(toks) > 0 else None, no=toks[1] if len(toks) > 1 else None,
        rate=fnum(fs.get('rate'), 0.0) if m.get('feesEnabled', True) else 0.0,
        feesEnabled=m.get('feesEnabled'), feeType=m.get('feeType'),
        exp=fs.get('exponent'), tick=fnum(m.get('orderPriceMinTickSize')), minsz=fnum(m.get('orderMinSize'), 5),
        active=m.get('active'), closed=m.get('closed'), accepting=m.get('acceptingOrders'),
        nro=m.get('negRiskOther'), negRisk=m.get('negRisk'), nrid=m.get('negRiskMarketID'), endDate=m.get('endDate'),
        uma=m.get('umaResolutionStatus'), umas=m.get('umaResolutionStatuses'),
        op=jl(m.get('outcomePrices')), bb=fnum(m.get('bestBid')), ba=fnum(m.get('bestAsk')),
        archived=m.get('archived'), ready=m.get('ready'), enableOrderBook=m.get('enableOrderBook'),
        git_thr=m.get('groupItemThreshold'), sportsMarketType=m.get('sportsMarketType'),
        outcomes=jl(m.get('outcomes')),
    )


def pull():
    t0 = time.time()
    evs = gamma_keyset('events/keyset', {'active': 'true', 'closed': 'false'}, limit=100, key='events')
    out = []
    for e in evs:
        out.append(dict(
            id=e.get('id'), slug=e.get('slug'), title=e.get('title'), negRisk=bool(e.get('negRisk')),
            negRiskAugmented=e.get('negRiskAugmented'), enableNegRisk=e.get('enableNegRisk'),
            negRiskMarketID=e.get('negRiskMarketID'),
            tags=[t.get('slug') for t in (e.get('tags') or [])], endDate=e.get('endDate'),
            volume24hr=e.get('volume24hr'), liquidity=e.get('liquidity'),
            markets=[compact_market(m) for m in (e.get('markets') or [])],
        ))
    ts = time.strftime('%Y%m%d%H%M', time.gmtime())
    blob = dict(ts=time.time(), events=out)
    tmp = f'{OUT}/universe_latest.json.tmp'
    with open(tmp, 'w') as f:
        json.dump(blob, f)
    os.replace(tmp, f'{OUT}/universe_latest.json')
    print(f'pulled {len(out)} events in {time.time()-t0:.0f}s', flush=True)
    return blob


if __name__ == '__main__':
    pull()
