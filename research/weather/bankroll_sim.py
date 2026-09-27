"""Cash-constrained bankroll simulation on the walk-forward backtest fills (Aug 1 - Sep 25).
Orders = fills grouped per (event, bucket, side, signal time); an order needs >= 5 shares (Polymarket minimum) and
enough free cash; cash (and payout) returns ~3h after the local day ends (resolution). Optional compounding: the
per-bucket cap tier grows with equity (cap3 -> cap5 -> cap10 -> cap20 -> cap50 backtest fill sets)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from wlib import load_events

ev = {e['slug']: e['day_end'] for e in load_events()}
TIERS = [(0, 'cap3'), (250, 'cap5'), (600, 'cap10'), (1500, 'cap20'), (3500, 'wf0801_e15')]


def orders(tag):
    df = pd.read_pickle(f'data/bt_stack_{tag}.pkl')
    o = df.groupby(['slug', 'j', 'side', 'tsig']).agg(t=('tfill', 'min'), shares=('shares', 'sum'), cost=('cost', 'sum'),
                                                      pnl=('pnl', 'sum'), date=('date', 'first')).reset_index()
    o = o[o.shares >= 5]
    o['tres'] = o.slug.map(ev) + 3 * 3600
    return o.sort_values('t').reset_index(drop=True)


def simulate(start, compound, fixed_tag='cap3'):
    sets = {tag: orders(tag) for _, tag in TIERS} if compound else {fixed_tag: orders(fixed_tag)}
    events = []
    for tag, o in sets.items():
        for r in o.itertuples():
            events.append((r.t, tag, r.cost, r.pnl, r.tres, r.date))
    events.sort()
    cash, pending, equity_curve, taken, skipped = start, [], [], 0, 0
    daily = {}
    for t, tag, cost, pnl, tres, date in events:
        # release resolved positions
        still = []
        for (tr, c, p, d) in pending:
            if tr <= t:
                cash += c + p; daily[d] = daily.get(d, 0) + p
            else:
                still.append((tr, c, p, d))
        pending = still
        equity = cash + sum(c for _, c, _, _ in pending)
        active = max(tg for th, tg in [(th, tg) for th, tg in TIERS] if equity >= th) if compound else fixed_tag
        active = [tg for th, tg in TIERS if equity >= th][-1] if compound else fixed_tag
        if tag != active:
            continue
        if cost <= cash:
            cash -= cost; pending.append((tres, cost, pnl, date)); taken += 1
        else:
            skipped += 1
        equity_curve.append((t, equity))
    for (tr, c, p, d) in pending:
        cash += c + p; daily[d] = daily.get(d, 0) + p
    eq = pd.Series([e for _, e in equity_curve])
    d = pd.Series(daily).sort_index()
    return dict(final=cash, min_equity=eq.min() if len(eq) else start, taken=taken, skipped=skipped,
                pos_days=(d > 0).mean(), days=len(d), worst_day=d.min(), best_day=d.max())


if __name__ == '__main__':
    for start in (100, 200, 500):
        r = simulate(start, compound=False, fixed_tag='cap3')
        print(f'start ${start:4d} fixed $3 cap : final ${r["final"]:8.0f}  min equity ${r["min_equity"]:6.0f}  orders taken {r["taken"]} skipped(no cash) {r["skipped"]}  pos days {r["pos_days"]:.0%}  worst day ${r["worst_day"]:.0f}')
        r = simulate(start, compound=True)
        print(f'start ${start:4d} compounding  : final ${r["final"]:8.0f}  min equity ${r["min_equity"]:6.0f}  orders taken {r["taken"]} skipped {r["skipped"]}  pos days {r["pos_days"]:.0%}  worst day ${r["worst_day"]:.0f}')
