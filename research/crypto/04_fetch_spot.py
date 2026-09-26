"""Fetch Binance spot klines (1m recent, 5m long) and Deribit DVOL history."""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa

UNDS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
now = pd.Timestamp.now(tz='UTC').floor('min')
s1 = pd.Timestamp('2026-06-25', tz='UTC')
s5 = pd.Timestamp('2025-09-01', tz='UTC')


def chunked(sym, interval, start, end, days=10):
    parts, cur = [], start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=days), end)
        parts.append((sym, interval, cur.value // 10**6, nxt.value // 10**6 - 1))
        cur = nxt
    with ThreadPoolExecutor(8) as ex:
        dfs = list(ex.map(lambda a: klines(*a), parts))
    return pd.concat(dfs).sort_index().loc[lambda d: ~d.index.duplicated()]


for u in UNDS:
    k1 = chunked(u, '1m', s1, now)
    k1.to_pickle(os.path.join(DATA, f'kl1m_{u}.pkl'))
    k5 = chunked(u, '5m', s5, now, days=30)
    k5.to_pickle(os.path.join(DATA, f'kl5m_{u}.pkl'))
    print(u, len(k1), k1.index.min(), k1.index.max(), len(k5), k5.index.min(), flush=True)

for c in ['BTC', 'ETH']:
    d = dvol_history(c, pd.Timestamp('2025-09-01', tz='UTC'), now, 3600)
    d.to_pickle(os.path.join(DATA, f'dvol_{c}.pkl'))
    print(c, len(d), d.index.min(), d.index.max(), flush=True)
