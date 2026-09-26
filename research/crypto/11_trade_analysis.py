"""Analyse trade_level.pkl: taker vs maker PnL, favourite-longshot on actual prints,
and a model-anchored continuously re-quoted maker (quote YES bid at q-thr / ask at q+thr)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 30)
D = pd.read_pickle(os.path.join(DATA, 'trade_level.pkl'))
D = D[(D.p > 0) & (D.p < 1)].copy()
D['date'] = pd.to_datetime(D['T'], unit='s', utc=True).dt.floor('D')
# token the taker bought and its price
D['bp'] = np.where(D.ask, D.p, 1 - D.p)
D['bwin'] = np.where(D.ask, D.y, 1 - D.y)
D['fee'] = taker_fee(D.bp)
D['taker_ps'] = D.bwin - D.bp - D.fee            # per share, net of fee
D['maker_ps'] = D.bp - D.bwin                    # counterparty (gross, no rebate)
D['usd'] = D.bp * D.sz
D['hb'] = pd.cut(D.tau_s / 60, [0, 15, 60, 240, 720, 1440, 2880, 1e9],
                 labels=['<15m', '15-60m', '1-4h', '4-12h', '12-24h', '1-2d', '>2d'])
D['bpb'] = pd.cut(D.bp, [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0])
D['szb'] = pd.cut(D.usd, [0, 10, 50, 200, 1000, 1e9], labels=['<$10', '$10-50', '$50-200', '$200-1k', '>$1k'])
print('prints', len(D), 'markets', D.mid_id.nunique(), 'events', D.event.nunique(), 'taker $', round(D.usd.sum() / 1e6, 2), 'M')


def agg(g):
    byday = (g.taker_ps * g.sz).groupby(g.date).sum()
    return pd.Series(dict(n=len(g), usd_k=g.usd.sum() / 1e3, taker_ret=(g.taker_ps * g.sz).sum() / g.usd.sum(),
                          taker_ret_gross=((g.bwin - g.bp) * g.sz).sum() / g.usd.sum(),
                          fee_pct=(g.fee * g.sz).sum() / g.usd.sum(),
                          maker_pnl_k=(g.maker_ps * g.sz).sum() / 1e3,
                          taker_c_per_share=100 * g.taker_ps.mean(),
                          t_day=-byday.mean() / (byday.std() / np.sqrt(len(byday))) if len(byday) > 2 else np.nan))


print('\n=== 1. Taker returns (share-weighted, net of taker fee) and maker gross PnL; t_day is for MAKER side')
print(agg(D).round(4).to_string())
for by in ['kind', 'und', 'hb', 'bpb', 'szb']:
    print(D.groupby(by, observed=True).apply(agg).round(4).to_string())
print(D.groupby(['kind', 'bpb'], observed=True).apply(agg).round(4).to_string())

print('\n=== 2. Calibration of actual prints (share-weighted): YES-equivalent price bucket vs outcome, by kind')
D['pb'] = pd.cut(D.p, [0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.0])
cal = D.groupby(['kind', 'pb'], observed=True).apply(lambda g: pd.Series(dict(
    n=len(g), mkts=g.mid_id.nunique(), p=np.average(g.p, weights=g.sz), y=np.average(g.y, weights=g.sz),
    q=np.average(g.q, weights=g.sz))))
cal['y-p'] = cal.y - cal.p
print(cal.round(4).to_string())


def mm(D, thr, cap=50, model='q', rebate=False):
    """Continuously re-quoted maker at model +/- thr. A print that traded through our level fills us
    at our level (size capped)."""
    q = D[model]
    fa = D.ask & (D.p > q + thr) & (q + thr < 0.995)       # taker bought YES above our ask -> we sold YES at q+thr
    fb = ~D.ask & (D.p < q - thr) & (q - thr > 0.005)      # taker sold YES below our bid -> we bought YES at q-thr
    s = np.minimum(D.sz, cap)
    pnl_ps = np.where(fa, (q + thr) - D.y, np.where(fb, D.y - (q - thr), np.nan))
    if rebate:
        pnl_ps = pnl_ps + 0.2 * taker_fee(np.where(fa, q + thr, q - thr))
    cost = np.where(fa, 1 - (q + thr), np.where(fb, q - thr, np.nan))  # capital: NO @1-ask or YES @bid
    r = D.assign(pnl_ps=pnl_ps, fsz=s, cost=cost, fside=np.where(fa, 'sellYES', np.where(fb, 'buyYES', '')))
    return r[r.fside != '']


def summ(r):
    byday = (r.pnl_ps * r.fsz).groupby(r.date).sum()
    return pd.Series(dict(fills=len(r), mkts=r.mid_id.nunique(), days=len(byday), shares_k=r.fsz.sum() / 1e3,
                          pnl_c_ps=100 * np.average(r.pnl_ps, weights=r.fsz), pnl_usd=(r.pnl_ps * r.fsz).sum(),
                          roi=(r.pnl_ps * r.fsz).sum() / (r.cost * r.fsz).sum(),
                          usd_per_day=(r.pnl_ps * r.fsz).sum() / max(1, D.date.nunique()),
                          t_day=byday.mean() / (byday.std() / np.sqrt(len(byday))) if len(byday) > 2 else np.nan))


print('\n=== 3. Model-anchored maker (quote at q_emp +/- thr, requoted continuously; fills = prints through our level; cap 50 sh/fill)')
out = {}
for model in ['q', 'q_rv']:
    for thr in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
        out[(model, thr)] = summ(mm(D, thr, model=model))
print(pd.DataFrame(out).T.round(4).to_string())
for thr in [0.02, 0.05]:
    r = mm(D, thr)
    print(f'\n-- thr={thr} split')
    for by in ['kind', 'fside', 'hb', 'und', 'bpb']:
        print(r.groupby(by, observed=True).apply(summ).round(4).to_string())

print('\n=== 4. Naive maker: always be the counterparty of every print (i.e. what liquidity providers earned), by kind x horizon')
print(D.groupby(['kind', 'hb'], observed=True).apply(lambda g: pd.Series(dict(
    n=len(g), maker_c_ps=100 * np.average(g.maker_ps, weights=g.sz), maker_usd_k=(g.maker_ps * g.sz).sum() / 1e3))).round(3).to_string())
