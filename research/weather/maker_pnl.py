"""Aggregate maker P&L (at resolution) on weather markets from the taker trade tape."""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(__file__))
from wlib import *

evs = load_events()
T = pickle.load(open('data/weather_trades.pkl', 'rb'))
rows = []
for e in evs:
    for b in e['buckets']:
        tr = T.get(b['cid'])
        if not tr:
            continue
        Y = 1.0 if b['won'] else 0.0
        for x in tr:
            p, sz = float(x['price']), float(x['size'])
            yes = (x['outcomeIndex'] == 0)
            taker_buys_yes = (x['side'] == 'BUY') == yes
            px_yes = p if yes else 1 - p
            # maker position in YES: +sz if maker bought YES
            mk = -sz if taker_buys_yes else sz
            rows.append((e['city'], e['kind'], e['date'], x['timestamp'], (e['day_end'] - x['timestamp']) / 3600.0,
                         px_yes, mk, mk * (Y - px_yes), sz * px_yes if yes else sz * p, p, sz, x['proxyWallet']))
df = pd.DataFrame(rows, columns=['city', 'kind', 'date', 'ts', 'h2end', 'pyes', 'mkpos', 'mkpnl', 'notional', 'p', 'sz', 'taker'])
df['fee'] = 0.05 * df.p * (1 - df.p) * df.sz
df.to_pickle('data/weather_tape.pkl')
print(len(df), 'trades; notional $%.0f; maker pnl $%.0f; taker fees $%.0f' % (df.notional.sum(), df.mkpnl.sum(), df.fee.sum()))
df['hb'] = pd.cut(df.h2end, [-100, 0, 3, 6, 12, 24, 36, 48, 1000])
df['pb'] = pd.cut(df.pyes, [0, .02, .05, .1, .2, .4, .6, .8, .9, .95, .98, 1])
g = df.groupby('hb', observed=True).agg(n=('mkpnl', 'size'), notional=('notional', 'sum'), mkpnl=('mkpnl', 'sum'), fee=('fee', 'sum'))
g['mk_bps'] = 1e4 * g.mkpnl / g.notional
print(g.round(1))
g = df.groupby('pb', observed=True).agg(n=('mkpnl', 'size'), notional=('notional', 'sum'), mkpnl=('mkpnl', 'sum'))
g['mk_bps'] = 1e4 * g.mkpnl / g.notional
print(g.round(1))
g = df.groupby('kind').agg(n=('mkpnl', 'size'), notional=('notional', 'sum'), mkpnl=('mkpnl', 'sum'))
g['mk_bps'] = 1e4 * g.mkpnl / g.notional
print(g.round(1))
