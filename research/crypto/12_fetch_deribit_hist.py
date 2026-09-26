"""Fetch historical Deribit option trades (history.deribit.com) for BTC/ETH (and SOL/XRP USDC options):
each trade carries mark_price, index_price, iv -> used to reconstruct hourly implied-vol smiles.
Output: data/crypto/deribit_trades_<CCY>.pkl"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa

H = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'
start = pd.Timestamp('2026-07-12', tz='UTC')
end = pd.Timestamp.now(tz='UTC').floor('h')


def fetch_window(args):
    ccy, s, e = args
    out, cur = [], s
    while True:
        for attempt in range(5):
            try:
                d = get(H, {'currency': ccy, 'kind': 'option', 'start_timestamp': cur, 'end_timestamp': e,
                            'count': 1000, 'sorting': 'asc'})['result']
                break
            except Exception:
                time.sleep(1 + 2 * attempt)
        else:
            print('fail', ccy, s, flush=True)
            return out
        tr = d['trades']
        out.extend((x['timestamp'], x['instrument_name'], x['price'], x['mark_price'], x['iv'], x['index_price'],
                    x['amount']) for x in tr)
        if not d.get('has_more') or not tr:
            return out
        cur = tr[-1]['timestamp'] + 1


for ccy in ['BTC', 'ETH', 'USDC']:
    wins, c = [], start
    while c < end:
        n = min(c + pd.Timedelta(hours=6), end)
        wins.append((ccy, int(c.value // 10**6), int(n.value // 10**6) - 1))
        c = n
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch_window, wins))
    df = pd.DataFrame([r for w in res for r in w], columns=['ts', 'inst', 'price', 'mark', 'iv', 'index', 'amount'])
    df = df.drop_duplicates()
    df.to_pickle(os.path.join(DATA, f'deribit_trades_{ccy}.pkl'))
    print(ccy, len(df), round(time.time() - t0), flush=True)
