"""Analyse live snapshots: Polymarket book vs Deribit-smile (p_iv), flat ATM IV, realized-vol and
empirical models; executable edges after fees; persistence across snapshots; model-free cross-market checks.
Writes data/crypto/live_panel.pkl and prints tables used in REPORT.md."""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa
from bt_models import *  # noqa

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 40)

fs = sorted(glob.glob(os.path.join(DATA, 'snapshots', 'snap_*.pkl')))
d = pd.concat([pd.read_pickle(f) for f in fs], ignore_index=True)
d['snap'] = d['t'].rank(method='dense').astype(int)
print('snapshots', d.snap.nunique(), 'rows', len(d), d.t.min(), d.t.max())

# empirical + blended-RV model evaluated at snapshot time (same estimator as the backtest)
emp = build_empirical()
now = pd.Timestamp.now(tz='UTC')
rvb = {}
for u in UNDS:
    k5 = klines(u, '5m', (now - pd.Timedelta(days=32)).value // 10**6, now.value // 10**6)
    rvb[u] = rv_frame(k5)['rv']
d['rv_blend'] = [rvb[u].asof(t) for u, t in zip(d.und, d.t)]
d['tau_s'] = d.tau_h * 3600
d['p_emp'] = [emp_prob(emp, r.payoff, r.S, r.tau_s, r.rv_blend, K=r.K, lo=r.lo, hi=r.hi)
              if r.p_iv not in (0.0, 1.0) or True else np.nan for r in d.itertuples()]
d['p_rvb'] = [gbm_prob(r.payoff, r.S, r.tau_s, r.rv_blend, K=r.K, lo=r.lo, hi=r.hi) for r in d.itertuples()]
# already-touched barriers -> 1
tch = ((d.payoff == 'touch_up') & (d.hi_so_far >= d.K)) | ((d.payoff == 'touch_down') & (d.lo_so_far <= d.K))
d.loc[tch, ['p_emp', 'p_rvb']] = 1.0

# fix: snapshot-time p_iv/p_atm/p_rv for open-low range buckets ('<X') were mis-evaluated (NaN bound);
# P(S_T < X) = 1 - P(above X) from the same snapshot's above-ladder when that strike exists
low = (d.kind == 'range_daily') & d['lo'].isna()
ab = d[d.kind == 'above_daily'].set_index(['snap', 'und', 'T', 'K'])
for c in ['p_iv', 'p_atm', 'p_rv', 'p_rv1d']:
    key = list(zip(d.loc[low, 'snap'], d.loc[low, 'und'], d.loc[low, 'T'], d.loc[low, 'hi']))
    vals = [1 - ab[c].get(k, np.nan) if k in ab.index else np.nan for k in key]
    d.loc[low, c] = vals
d['mid'] = (d.bid + d.ask) / 2
d['spr'] = d.ask - d.bid
d['two'] = d.bid.notna() & d.ask.notna()
kk = d.K.fillna(d.lo).fillna(d.hi)
d['zmon'] = np.log(kk / d.S) / (d.atm_iv * np.sqrt(d.tau_h / 8760))
d['tau_b'] = pd.cut(d.tau_h, [0, 2, 12, 36, 96, 24 * 8, 24 * 40, 1e6],
                    labels=['<2h', '2-12h', '12-36h', '1.5-4d', '4-8d', '8-40d', '>40d'])
d['mon_b'] = pd.cut(d.zmon.abs(), [-1, 0.5, 1, 2, 3, 1e9], labels=['|z|<.5', '.5-1', '1-2', '2-3', '>3'])
d['kind2'] = d.kind.replace({'hit_weekly2': 'hit_weekly'})
d.to_pickle(os.path.join(DATA, 'live_panel.pkl'))

g = d[d.two & (d.spr <= 0.10)].copy()
print('\n== two-sided books (spread<=10c):', len(g), 'of', len(d), '; median spread by kind')
print(g.groupby('kind2').spr.describe()[['count', 'mean', '50%']].round(4))
print('\n== mean(mid - model) and mean|mid - model| by kind (pp)')
for col in ['p_iv', 'p_atm', 'p_rvb', 'p_emp']:
    g['e_' + col] = g.mid - g[col]
agg = g.groupby('kind2').agg(n=('mid', 'size'), iv=('e_p_iv', 'mean'), iv_abs=('e_p_iv', lambda x: x.abs().mean()),
                             rvb=('e_p_rvb', 'mean'), rvb_abs=('e_p_rvb', lambda x: x.abs().mean()),
                             emp=('e_p_emp', 'mean'), emp_abs=('e_p_emp', lambda x: x.abs().mean()))
print((agg * [1, 100, 100, 100, 100, 100, 100]).round(2))
print('\n== mid - p_iv (pp) by kind x tau bucket')
print((g.pivot_table(index='kind2', columns='tau_b', values='e_p_iv', aggfunc='mean', observed=False) * 100).round(2))
print('\n== mid - p_iv (pp) by kind x |moneyness| bucket (z = ln(K/S)/(atm*sqrt(T)))')
print((g.pivot_table(index='kind2', columns='mon_b', values='e_p_iv', aggfunc='mean', observed=False) * 100).round(2))
print((g.pivot_table(index='kind2', columns='mon_b', values='e_p_iv', aggfunc='size', observed=False)))
print('\n== mid - p_iv (pp) for touch markets, by direction x tau')
t_ = g[g.payoff.str.startswith('touch')]
print((t_.pivot_table(index=['kind2', 'payoff'], columns='tau_b', values='e_p_iv', aggfunc='mean', observed=False) * 100).round(2))

# executable edges vs Deribit smile model
for col in ['p_iv', 'p_emp']:
    d['eY_' + col] = d[col] - d.ask - taker_fee(d.ask)          # buy YES at ask (taker)
    d['eN_' + col] = d.bid - d[col] - taker_fee(d.bid)          # buy NO at 1-bid (taker) == sell YES at bid
e = d[d.two].copy()
e['edge_iv'] = e[['eY_p_iv', 'eN_p_iv']].max(axis=1)
e['side_iv'] = np.where(e.eY_p_iv >= e.eN_p_iv, 'YES', 'NO')
e['edge_both'] = np.where(e.side_iv == 'YES', np.minimum(e.eY_p_iv, e.eY_p_emp), np.minimum(e.eN_p_iv, e.eN_p_emp))
print('\n== taker-executable edge vs Deribit smile after fee: rows with edge > x (per snapshot avg)')
ns = d.snap.nunique()
for x in [0.0, 0.01, 0.02, 0.05]:
    s = e[e.edge_iv > x]
    s2 = e[e.edge_both > x]
    print(f'edge>{x:.2f}: {len(s) / ns:.1f} mkts/snap (iv only), {len(s2) / ns:.1f} (iv AND empirical agree);'
          f' by kind: {s.groupby("kind2").size().div(ns).round(1).to_dict()}')
top = e[e.edge_both > 0.02].sort_values('edge_both', ascending=False)
top = top.drop_duplicates('yes')
cols = ['kind2', 'event', 'title', 'tau_h', 'S', 'bid', 'ask', 'bid_sz', 'ask_sz', 'p_iv', 'p_emp', 'p_rvb', 'side_iv',
        'edge_iv', 'edge_both']
print(top[cols].head(40).round(4).to_string())

# persistence: for each market, edge in consecutive snapshots
p = e.pivot_table(index='yes', columns='snap', values='edge_iv')
flag = p > 0.02
cont = []
for i in range(1, p.shape[1]):
    a, b = flag.iloc[:, i - 1], flag.iloc[:, i]
    if a.sum():
        cont.append((b & a).sum() / a.sum())
print('\n== persistence: P(edge>2c at snap i+1 | edge>2c at snap i) =', round(np.mean(cont), 3) if cont else None)
first, last = p.columns.min(), p.columns.max()
both = p[[first, last]].dropna()
print('corr(edge first snap, edge last snap) =', round(both.corr().iloc[0, 1], 3), 'n=', len(both))
# does the gap close via the market moving to the model?
m1 = d[d.snap == first].set_index('yes')
m2 = d[d.snap == last].set_index('yes')
j = m1.join(m2[['mid', 'p_iv', 'S']], rsuffix='_2', how='inner').dropna(subset=['mid', 'mid_2', 'p_iv', 'p_iv_2'])
j = j[(j.spr <= 0.1)]
j['gap1'] = j.mid - j.p_iv
j['dmid'] = j.mid_2 - j.mid
j['dmodel'] = j.p_iv_2 - j.p_iv
print('regress d(mid) on gap1: slope', round(np.polyfit(j.gap1, j.dmid, 1)[0], 3), ' d(model) on gap1: slope',
      round(np.polyfit(j.gap1, j.dmodel, 1)[0], 3), 'n', len(j))

# ---------------- model-free consistency: range buckets vs above ladder (same Binance 12:00 ET close)
print('\n== model-free: range bucket vs above-ladder difference (same date/underlying)')
out = []
for sn in sorted(d.snap.unique()):
    s = d[(d.snap == sn)]
    A = s[s.kind == 'above_daily']
    R = s[s.kind == 'range_daily']
    for (u, T), a in A.groupby(['und', 'T']):
        r = R[(R.und == u) & (R['T'] == T)]
        a = a.set_index('K')
        for x in r.itertuples():
            if x.lo is None or x.hi is None or x.lo not in a.index or x.hi not in a.index:
                continue
            al, ah = a.loc[x.lo], a.loc[x.hi]
            # synthetic range = long above(lo) + short above(hi): buy YES lo at ask, sell YES hi at bid
            syn_ask = al.ask - ah.bid
            syn_bid = al.bid - ah.ask
            out.append(dict(snap=sn, und=u, T=T, lo=x.lo, hi=x.hi, r_bid=x.bid, r_ask=x.ask, syn_bid=syn_bid,
                            syn_ask=syn_ask, arb1=x.bid - syn_ask, arb2=syn_bid - x.ask))
X = pd.DataFrame(out)
if len(X):
    X['arb'] = X[['arb1', 'arb2']].max(axis=1)
    print('pairs', len(X), ' pairs with gross arb > 0:', (X.arb > 0).sum(), ' > 1c:', (X.arb > 0.01).sum())
    print(X.sort_values('arb', ascending=False).head(10).round(4).to_string())
# hit_daily(up K) >= above_daily(K) same day  / weekly hit >= above for dates in week
print('\n== model-free: touch >= digital bound violations (ask_touch < bid_digital)')
out = []
for sn in sorted(d.snap.unique()):
    s = d[d.snap == sn]
    A = s[s.kind == 'above_daily']
    H = s[s.payoff == 'touch_up']
    for x in H.itertuples():
        a = A[(A.und == x.und) & (A.K == x.K) & (A['T'] <= x.T) & (A['T'] >= x.win_start)]
        for y in a.itertuples():
            out.append(dict(snap=sn, und=x.und, K=x.K, hit_ev=x.event, above_ev=y.event, hit_ask=x.ask, above_bid=y.bid,
                            viol=y.bid - x.ask))
V = pd.DataFrame(out)
if len(V):
    print('pairs', len(V), 'violations (above_bid > hit_ask):', (V.viol > 0).sum())
    print(V.sort_values('viol', ascending=False).head(8).round(4).to_string())
