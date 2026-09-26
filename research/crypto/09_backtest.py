"""Backtest on resolved markets (Jul 20 - Sep 25 2026): calibration of market vs models,
favourite-longshot, and taker / maker trading rules including fees.
Reads data/crypto/panel_tr.pkl (panel + trade features) and live_panel.pkl (spread estimates)."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa
from sklearn.linear_model import LogisticRegression

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 40)
EPS = 1e-3

P = pd.read_pickle(os.path.join(DATA, 'panel_tr.pkl'))
P['date'] = pd.to_datetime(P['T'], unit='s', utc=True).dt.floor('D')
P['hb'] = pd.cut(P.h_min, [0, 45, 90, 240, 720, 1500, 3000, 1e6],
                 labels=['30m', '1h', '2-3h', '6-12h', '18-24h', '2d', '3-5d'])
P['p_mb'] = pd.cut(P.p_mkt, [-0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 1.01])
P['blend'] = 0.5 * P.p_rv + 0.5 * P.p_emp
# liquidity filter: a print within 2h and the history mid consistent with it
P['liquid'] = (P.last_age <= 7200) & ((P.p_mkt - P.last_p).abs() <= 0.04)
P['nondeg'] = (P[['p_mkt', 'p_rv', 'p_emp']].min(axis=1) < 0.995) & (P[['p_mkt', 'p_rv', 'p_emp']].max(axis=1) > 0.005)
print('rows', len(P), 'liquid', int(P.liquid.sum()), 'liquid & non-degenerate', int((P.liquid & P.nondeg).sum()))
print(P[P.nondeg].groupby(['kind', 'und']).liquid.mean().unstack().round(3))


def brier(p, y):
    return float(np.mean((np.clip(p, 0, 1) - y) ** 2))


def logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


L = P[P.liquid & P.nondeg].copy()

print('\n=== A. Brier (x100) & log-loss, liquid non-degenerate rows')
rows = []
for k, g in list(L.groupby('kind')) + [('ALL', L)]:
    gd = g.dropna(subset=['p_dvol'])
    rows.append(dict(kind=k, n=len(g), events=g.event.nunique(), mkt=100 * brier(g.p_mkt, g.y),
                     emp=100 * brier(g.p_emp, g.y), rv=100 * brier(g.p_rv, g.y), blend=100 * brier(g.blend, g.y),
                     mix=100 * brier(0.5 * g.p_mkt + 0.5 * g.p_emp, g.y),
                     LLmkt=logloss(g.p_mkt, g.y), LLemp=logloss(g.p_emp, g.y), LLrv=logloss(g.p_rv, g.y),
                     n_btceth=len(gd), mkt_be=100 * brier(gd.p_mkt, gd.y), dvol_be=100 * brier(gd.p_dvol, gd.y)))
print(pd.DataFrame(rows).set_index('kind').round(3).to_string())

print('\n=== A2. Brier (x100) by kind x horizon (liquid): mkt / emp / rv / dvol')
for k, g in L.groupby('kind'):
    t = g.groupby('hb', observed=True).apply(lambda x: pd.Series(dict(
        n=len(x), mkt=100 * brier(x.p_mkt, x.y), emp=100 * brier(x.p_emp, x.y), rv=100 * brier(x.p_rv, x.y),
        dvol=100 * brier(x.p_dvol.dropna(), x.y[x.p_dvol.notna()]) if x.p_dvol.notna().any() else np.nan)))
    print(k)
    print(t.round(3).T.to_string())

print('\n=== B. Reliability (liquid): realized YES rate vs market price bucket')
rel = L.groupby(['kind', 'p_mb'], observed=True).agg(n=('y', 'size'), ev=('event', 'nunique'), p_mkt=('p_mkt', 'mean'),
                                                     y=('y', 'mean'), p_emp=('p_emp', 'mean'), p_rv=('p_rv', 'mean'))
rel['y-mkt'] = rel.y - rel.p_mkt
print(rel.round(4).to_string())
rel2 = L.groupby('p_mb', observed=True).agg(n=('y', 'size'), ev=('event', 'nunique'), p_mkt=('p_mkt', 'mean'),
                                             y=('y', 'mean'), p_emp=('p_emp', 'mean'))
rel2['y-mkt'] = rel2.y - rel2.p_mkt
print('pooled\n', rel2.round(4).to_string())
tt = L[L.payoff.str.startswith('touch')]
print('touch markets by direction')
print(tt.groupby(['payoff', 'p_mb'], observed=True).agg(n=('y', 'size'), ev=('event', 'nunique'), p_mkt=('p_mkt', 'mean'),
                                                          y=('y', 'mean'), p_emp=('p_emp', 'mean')).round(4).to_string())


def lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


print('\n=== C. Does the model add information beyond the market? logistic stacking, train=first half of dates, test=second')
split = L.date.quantile(0.5)
for k, g in list(L.groupby('kind')) + [('ALL', L)]:
    tr, te = g[g.date <= split], g[g.date > split]
    if len(tr) < 150 or len(te) < 150:
        continue
    out = {'n_test': len(te), 'raw_mkt': 100 * brier(te.p_mkt, te.y)}
    for name, cols in [('mkt', ['p_mkt']), ('mkt+emp', ['p_mkt', 'p_emp']), ('mkt+rv', ['p_mkt', 'p_rv'])]:
        X = np.column_stack([lg(tr[c]) for c in cols])
        m = LogisticRegression(C=1e4, max_iter=2000).fit(X, tr.y)
        out[name] = 100 * brier(m.predict_proba(np.column_stack([lg(te[c]) for c in cols]))[:, 1], te.y.values)
        out[name + '_b'] = np.round(m.coef_[0], 2).tolist()
    print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in out.items()})

# ---------------------------------------------------------------- trading rules
ls = pd.read_pickle(os.path.join(DATA, 'live_panel.pkl'))
ls = ls[ls.two].copy()
ls['p_mb'] = pd.cut(ls.mid, P.p_mb.cat.categories)
SPR = ls.groupby(['kind2', 'p_mb'], observed=True).spr.median()
SPR_K = ls.groupby('kind2').spr.median()
P['spr_est'] = [float(SPR.get((k, mb), SPR_K.get(k, 0.03))) for k, mb in zip(P.kind, P.p_mb)]
P['spr_est'] = P.spr_est.clip(lower=0.002)


def summarize(r, label=''):
    if not len(r):
        return dict(rule=label, n=0)
    byday = r.groupby('date').pnl.sum()
    t = byday.mean() / (byday.std(ddof=1) / np.sqrt(len(byday))) if len(byday) > 2 and byday.std() > 0 else np.nan
    return dict(rule=label, n=len(r), mkts=r.mid_id.nunique(), events=r.event.nunique(), days=len(byday),
                avg_pnl_c=100 * r.pnl.mean(), roi=r.pnl.sum() / r.cost.sum(), win=np.mean(r.pnl > 0),
                tot=r.pnl.sum(), t_day=t, yes_share=np.mean(r.side == 'YES'), avg_cost=r.cost.mean())


def taker(df, model, thr, exec_mode):
    """exec_mode 'print': pay the next same-side taker print within 30min (skip if none);
    'half': pay history mid +/- half of the live median spread for that kind/price bucket."""
    sig = df[model] - df.p_mkt
    if exec_mode == 'print':
        pY, pN = df.ask_next, 1 - df.bid_next
    else:
        pY = (df.p_mkt + df.spr_est / 2).clip(upper=0.999)
        pN = (1 - df.p_mkt + df.spr_est / 2).clip(upper=0.999)
    # require the edge to survive the actual price paid, not just the mid
    buyY = (sig > thr) & (df[model] - pY - taker_fee(pY) > 0)
    buyN = (sig < -thr) & ((1 - df[model]) - pN - taker_fee(pN) > 0)
    pnl = np.where(buyY, df.y - pY - taker_fee(pY), np.where(buyN, (1 - df.y) - pN - taker_fee(pN), np.nan))
    cost = np.where(buyY, pY, np.where(buyN, pN, np.nan))
    r = df.assign(pnl=pnl, cost=cost, side=np.where(buyY, 'YES', np.where(buyN, 'NO', '')))
    return r[(r.side != '') & np.isfinite(r.pnl)]


def maker(df, model, thr, w):
    """Post YES bid at q-thr and NO bid at (1-q)-thr (only if passive vs current mid); filled if a taker print
    within w minutes traded through our price. Maker pays no fee."""
    q = df[model]
    tick = df.tick.fillna(0.01)
    b = np.floor((q - thr) / tick) * tick
    nb = np.floor(((1 - q) - thr) / tick) * tick
    yb_ok = (b > 0) & (b < df.p_mkt - tick / 2)
    nb_ok = (nb > 0) & (1 - nb > df.p_mkt + tick / 2)
    fillY = yb_ok & (df[f'minbid_{w}'] < b - 1e-9)
    fillN = nb_ok & (df[f'maxask_{w}'] > 1 - nb + 1e-9)
    rY = df[fillY].assign(pnl=lambda x: x.y - b[fillY], cost=b[fillY], side='YES')
    rN = df[fillN].assign(pnl=lambda x: (1 - x.y) - nb[fillN], cost=nb[fillN], side='NO')
    return pd.concat([rY, rN]), int(yb_ok.sum() + nb_ok.sum())


B = P[P.nondeg].copy()
Bl = B[B.liquid]
print('\n=== D. TAKER rules. exec=print: next same-side print within 30m (+fee). exec=half: mid+half live spread (+fee), liquid rows only')
res = []
for model in ['p_emp', 'p_rv', 'blend', 'p_dvol']:
    for thr in [0.03, 0.05, 0.10, 0.15]:
        for mode, df in [('print', B), ('half', Bl)]:
            for k, g in list(df.groupby('kind')) + [('ALL', df)]:
                s = summarize(taker(g.dropna(subset=[model]), model, thr, mode), f'{model}>{thr:.2f}')
                s.update(kind=k, exec=mode)
                res.append(s)
D = pd.DataFrame(res)
show = ['rule', 'exec', 'kind', 'n', 'mkts', 'events', 'days', 'avg_pnl_c', 'roi', 'win', 'tot', 't_day', 'yes_share', 'avg_cost']
print(D[D.kind == 'ALL'][show].round(4).to_string())
print(D[(D.kind != 'ALL') & D.rule.isin(['p_emp>0.05', 'p_emp>0.10', 'blend>0.05', 'blend>0.10'])][show].round(4).to_string())

print('\n=== D2. taker p_emp>0.05 (exec=print) split')
r = taker(B.dropna(subset=['p_emp']), 'p_emp', 0.05, 'print')
for by in ['side', 'hb', 'kind', 'und', 'p_mb']:
    print(pd.DataFrame({k: summarize(x) for k, x in r.groupby(by, observed=True)}).T[show[3:]].round(4).to_string())
r.to_pickle(os.path.join(DATA, 'bt_taker_emp05.pkl'))

print('\n=== E. MAKER rules (post at model -/+ thr, fill if a later print trades through within w min)')
res = []
for model in ['p_emp', 'blend']:
    for thr in [0.02, 0.04, 0.07, 0.10]:
        for w in [15, 60]:
            for k, g in list(B.groupby('kind')) + [('ALL', B)]:
                rr, nq = maker(g, model, thr, w)
                s = summarize(rr, f'{model}-{thr:.2f}/{w}m')
                s.update(kind=k, quotes=nq)
                res.append(s)
E = pd.DataFrame(res)
print(E[E.kind == 'ALL'][['rule', 'quotes'] + show[2:]].round(4).to_string())
print(E[(E.kind != 'ALL') & E.rule.isin(['p_emp-0.04/60m', 'p_emp-0.07/60m', 'blend-0.04/60m'])][['rule', 'quotes'] + show[2:]].round(4).to_string())
rr, _ = maker(B, 'p_emp', 0.04, 60)
print('maker p_emp-0.04/60m split by side / horizon / und')
for by in ['side', 'hb', 'und']:
    print(pd.DataFrame({k: summarize(x) for k, x in rr.groupby(by, observed=True)}).T[show[3:]].round(4).to_string())

print('\n=== F. Structural (model-free) rules on liquid rows, taker at mid+half-spread+fee')
for k in ['above_daily', 'range_daily', 'hit_daily', 'hit_weekly', 'hit_monthly']:
    g = P[(P.kind == k) & P.liquid]
    for lo_, hi_ in [(0.0, 0.03), (0.03, 0.1), (0.1, 0.25), (0.75, 0.9), (0.9, 0.97), (0.97, 1.0)]:
        s = g[(g.p_mkt > lo_) & (g.p_mkt <= hi_)]
        if len(s) < 30:
            continue
        pN = (1 - s.p_mkt + s.spr_est / 2).clip(upper=0.999)
        pY = (s.p_mkt + s.spr_est / 2).clip(upper=0.999)
        pnlN = (1 - s.y) - pN - taker_fee(pN)
        pnlY = s.y - pY - taker_fee(pY)
        print(f'{k:12s} p in ({lo_:.2f},{hi_:.2f}] n={len(s):5d} ev={s.event.nunique():3d} yes_rate={s.y.mean():.4f} '
              f'mkt={s.p_mkt.mean():.4f} emp={s.p_emp.mean():.4f} | buy NO pnl={100 * pnlN.mean():+.2f}c  buy YES pnl={100 * pnlY.mean():+.2f}c')
P.to_pickle(os.path.join(DATA, 'panel_bt.pkl'))
