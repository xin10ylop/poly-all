"""Is the maker adverse selection latency-driven? Re-price the model at each Polymarket print with Binance spot
from 1 SECOND before the print (1s klines) instead of the last 1-minute close, for BTC/ETH markets
resolving Sep 13-25, and compare the model-anchored maker PnL.
Output: data/crypto/kl1s_<SYM>.pkl, printed table."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa
from bt_models import *  # noqa

S0, S1 = pd.Timestamp('2026-09-11', tz='UTC'), pd.Timestamp('2026-09-25 16:05', tz='UTC')


def get1s(sym):
    p = os.path.join(DATA, f'kl1s_{sym}.pkl')
    if os.path.exists(p):
        return pd.read_pickle(p)
    starts = np.arange(S0.value // 10**6, S1.value // 10**6, 1000 * 1000)

    def f(s):
        for a in range(4):
            try:
                d = get(f'{BV}/klines', {'symbol': sym, 'interval': '1s', 'startTime': int(s), 'limit': 1000})
                return [(r[0] // 1000, float(r[4])) for r in d]
            except Exception:
                time.sleep(1 + a)
        return []
    with ThreadPoolExecutor(8) as ex:
        rows = [x for part in ex.map(f, starts) for x in part]
    s = pd.Series(dict(rows)).sort_index()
    s.to_pickle(p)
    return s


D = pd.read_pickle(os.path.join(DATA, 'trade_level.pkl'))
D = D[D.und.isin(['BTCUSDT', 'ETHUSDT']) & (D['T'] >= S0.timestamp() + 2 * 86400) & (D['T'] <= S1.timestamp())
      & (D.p > 0) & (D.p < 1)].copy()
print('prints in window', len(D), D.kind.value_counts().to_dict(), flush=True)
emp = build_empirical()
for u in ['BTCUSDT', 'ETHUSDT']:
    s = get1s(u)
    print(u, '1s bars', len(s), flush=True)
    m = D.und == u
    tt = D.loc[m, 't'].values - 1
    idx = np.searchsorted(s.index.values, tt, side='right') - 1
    D.loc[m, 'S1s'] = s.values[np.clip(idx, 0, None)]
    D.loc[m, 'age1s'] = tt - s.index.values[np.clip(idx, 0, None)]
D = D[D.age1s <= 5]
D['q1s'] = [emp_prob(emp, pf, S, ta, sg, K=K, lo=lo, hi=hi) for pf, S, ta, sg, K, lo, hi in
            zip(D.payoff, D.S1s, D.tau_s, D.rv, D.K, D['lo'], D['hi'])]
D['date'] = pd.to_datetime(D['T'], unit='s', utc=True).dt.floor('D')
print('spot move between 1m-stale and 1s-fresh: median |dS/S| bp', round(1e4 * np.median(np.abs(D.S1s / D.S - 1)), 2))
print('corr(print price, q_1m)', round(D[['p', 'q']].corr().iloc[0, 1], 4), ' corr(print, q_1s)', round(D[['p', 'q1s']].corr().iloc[0, 1], 4))
print('Brier: print', round(np.mean((D.p - D.y) ** 2), 5), ' q_1m', round(np.mean((D.q - D.y) ** 2), 5),
      ' q_1s', round(np.mean((D.q1s - D.y) ** 2), 5))
rows = []
for model in ['q', 'q1s']:
    for thr in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]:
        q = D[model]
        fa = D.ask & (D.p > q + thr) & (q + thr < 0.995)
        fb = ~D.ask & (D.p < q - thr) & (q - thr > 0.005)
        sz = np.minimum(D.sz, 50)
        pnl = np.where(fa, q + thr - D.y, np.where(fb, D.y - (q - thr), np.nan))
        ok = np.isfinite(pnl)
        byday = pd.Series(pnl[ok] * sz[ok]).groupby(D.date.values[ok]).sum()
        rows.append(dict(model=model, thr=thr, fills=int(ok.sum()), pnl_c_ps=100 * np.average(pnl[ok], weights=sz[ok]),
                         usd=(pnl[ok] * sz[ok]).sum(), days=len(byday),
                         t_day=byday.mean() / (byday.std() / np.sqrt(len(byday)))))
        for k in ['above_daily', 'hit_daily', 'range_daily']:
            mk = ok & (D.kind == k).values
            if mk.sum():
                rows[-1][k + '_c'] = 100 * np.average(pnl[mk], weights=sz[mk])
print(pd.DataFrame(rows).round(3).to_string())
