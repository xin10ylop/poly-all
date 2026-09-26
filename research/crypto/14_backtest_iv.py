"""Backtest with the point-in-time Deribit smile model (p_iv) vs the market, incl. drift-neutral splits.
Reads data/crypto/panel_iv.pkl (from 13_deribit_hist_model.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa
from sklearn.linear_model import LogisticRegression

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 40)
EPS = 1e-3
P = pd.read_pickle(os.path.join(DATA, 'panel_iv.pkl'))
P['date'] = pd.to_datetime(P['T'], unit='s', utc=True).dt.floor('D')
P['hb'] = pd.cut(P.h_min, [0, 45, 90, 240, 720, 1500, 3000, 1e6],
                 labels=['30m', '1h', '2-3h', '6-12h', '18-24h', '2d', '3-5d'])
P['p_mb'] = pd.cut(P.p_mkt, [-0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.01])
P['liquid'] = (P.last_age <= 7200) & ((P.p_mkt - P.last_p).abs() <= 0.04)
P = P.dropna(subset=['p_iv'])
P['nondeg'] = (P[['p_mkt', 'p_iv', 'p_emp']].min(axis=1) < 0.995) & (P[['p_mkt', 'p_iv', 'p_emp']].max(axis=1) > 0.005)
L = P[P.liquid & P.nondeg].copy()
print('rows with p_iv', len(P), 'liquid non-degenerate', len(L), L.groupby('und').size().to_dict())


def brier(p, y):
    return float(np.mean((np.clip(p, 0, 1) - y) ** 2))


def lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


print('\n=== A. Brier x100 (liquid, non-degenerate): market vs Deribit smile (iv), implied-ATM lognormal, implied-ATM x empirical shape, RV-empirical')
rows = []
for k, g in list(L.groupby('kind')) + [('ALL', L)]:
    rows.append(dict(kind=k, n=len(g), ev=g.event.nunique(), mkt=100 * brier(g.p_mkt, g.y), iv=100 * brier(g.p_iv, g.y),
                     ivatm=100 * brier(g.p_ivatm, g.y), ivemp=100 * brier(g.p_ivemp.fillna(g.p_iv), g.y),
                     emp=100 * brier(g.p_emp, g.y), mix_mkt_iv=100 * brier(0.5 * g.p_mkt + 0.5 * g.p_iv, g.y),
                     mean_mkt=g.p_mkt.mean(), mean_iv=g.p_iv.mean(), mean_y=g.y.mean()))
print(pd.DataFrame(rows).set_index('kind').round(3).to_string())
for u, g in L.groupby('und'):
    print(u, 'n', len(g), 'mkt', round(100 * brier(g.p_mkt, g.y), 3), 'iv', round(100 * brier(g.p_iv, g.y), 3),
          'ivemp', round(100 * brier(g.p_ivemp.fillna(g.p_iv), g.y), 3))
print('\nby horizon (ALL kinds)')
print(L.groupby('hb', observed=True).apply(lambda x: pd.Series(dict(n=len(x), mkt=100 * brier(x.p_mkt, x.y),
      iv=100 * brier(x.p_iv, x.y), ivemp=100 * brier(x.p_ivemp.fillna(x.p_iv), x.y), emp=100 * brier(x.p_emp, x.y)))).round(3).T.to_string())

print('\n=== B. Where do market and Deribit disagree? mean(mkt - iv) and realized (y - iv), by kind x market-price bucket')
t = L.groupby(['kind', 'p_mb'], observed=True).agg(n=('y', 'size'), mkt=('p_mkt', 'mean'), iv=('p_iv', 'mean'),
                                                   y=('y', 'mean'))
t['mkt-iv'] = t.mkt - t.iv
t['y-iv'] = t.y - t.iv
t['y-mkt'] = t.y - t.mkt
print(t.round(4).to_string())

print('\n=== C. Stacking (train first half of dates, test second half)')
split = L.date.quantile(0.5)
for k, g in list(L.groupby('kind')) + [('ALL', L)]:
    tr, te = g[g.date <= split], g[g.date > split]
    if len(tr) < 150 or len(te) < 150:
        continue
    out = {'n_test': len(te), 'raw_mkt': 100 * brier(te.p_mkt, te.y), 'raw_iv': 100 * brier(te.p_iv, te.y)}
    for name, cols in [('mkt', ['p_mkt']), ('iv', ['p_iv']), ('mkt+iv', ['p_mkt', 'p_iv'])]:
        m = LogisticRegression(C=1e4, max_iter=2000).fit(np.column_stack([lg(tr[c]) for c in cols]), tr.y)
        out[name] = 100 * brier(m.predict_proba(np.column_stack([lg(te[c]) for c in cols]))[:, 1], te.y.values)
        out[name + '_b'] = np.round(m.coef_[0], 2).tolist()
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in out.items()})

# ---------------------------------------------------------------- trading
ls = pd.read_pickle(os.path.join(DATA, 'live_panel.pkl'))
ls = ls[ls.two].copy()
ls['p_mb'] = pd.cut(ls.mid, P.p_mb.cat.categories)
SPR = ls.groupby(['kind2', 'p_mb'], observed=True).spr.median()
SPR_K = ls.groupby('kind2').spr.median()
P['spr_est'] = [float(SPR.get((k, mb), SPR_K.get(k, 0.03))) for k, mb in zip(P.kind, P.p_mb)]
P['spr_est'] = P.spr_est.clip(lower=0.002)


def bull(r):
    """+1 if the bet profits from the underlying going up, -1 if down, 0 for range buckets."""
    up = np.where(r.payoff.isin(['digital', 'touch_up']), 1, np.where(r.payoff == 'touch_down', -1, 0))
    return up * np.where(r.side == 'YES', 1, -1)


def taker(df, model, thr, mode):
    sig = df[model] - df.p_mkt
    if mode == 'print':
        pY, pN = df.ask_next, 1 - df.bid_next
    else:
        pY = (df.p_mkt + df.spr_est / 2).clip(upper=0.999)
        pN = (1 - df.p_mkt + df.spr_est / 2).clip(upper=0.999)
    buyY = (sig > thr) & (df[model] - pY - taker_fee(pY) > 0)
    buyN = (sig < -thr) & ((1 - df[model]) - pN - taker_fee(pN) > 0)
    pnl = np.where(buyY, df.y - pY - taker_fee(pY), np.where(buyN, (1 - df.y) - pN - taker_fee(pN), np.nan))
    cost = np.where(buyY, pY, np.where(buyN, pN, np.nan))
    r = df.assign(pnl=pnl, cost=cost, side=np.where(buyY, 'YES', np.where(buyN, 'NO', '')))
    r = r[(r.side != '') & np.isfinite(r.pnl)]
    return r.assign(dirn=bull(r))


def summ(r):
    if not len(r):
        return pd.Series(dict(n=0))
    byday = r.groupby('date').pnl.sum()
    return pd.Series(dict(n=len(r), mkts=r.mid_id.nunique(), days=len(byday), avg_pnl_c=100 * r.pnl.mean(),
                          roi=r.pnl.sum() / r.cost.sum(), win=np.mean(r.pnl > 0), tot=r.pnl.sum(),
                          t_day=byday.mean() / (byday.std() / np.sqrt(len(byday))) if len(byday) > 2 and byday.std() > 0 else np.nan,
                          avg_cost=r.cost.mean()))


B = P[P.nondeg]
print('\n=== D. TAKER vs Deribit smile (exec=print: next same-side print <=30m; exec=half: liquid rows, mid+half spread). fee incl.')
out = {}
for model in ['p_iv', 'p_ivemp']:
    for thr in [0.02, 0.03, 0.05, 0.08, 0.12]:
        for mode, df in [('print', B), ('half', B[B.liquid])]:
            out[(model, thr, mode)] = summ(taker(df.dropna(subset=[model]), model, thr, mode))
print(pd.DataFrame(out).T.round(4).to_string())
for thr in [0.03, 0.05]:
    r = taker(B, 'p_iv', thr, 'print')
    print(f'\n-- p_iv>{thr} exec=print, splits')
    for by in ['kind', 'side', 'dirn', 'hb', 'und', 'p_mb']:
        print(r.groupby(by, observed=True).apply(summ).round(4).to_string())
    r.to_pickle(os.path.join(DATA, f'bt_taker_iv{int(thr * 100):02d}.pkl'))

print('\n=== E. Drift-neutral check: PnL of bullish vs bearish bets for p_iv>0.03 (exec=half, liquid)')
r = taker(B[B.liquid], 'p_iv', 0.03, 'half')
print(r.groupby(['kind', 'dirn']).apply(summ).round(4).to_string())
P.to_pickle(os.path.join(DATA, 'panel_iv_bt.pkl'))
