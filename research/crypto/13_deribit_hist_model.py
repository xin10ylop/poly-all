"""Reconstruct point-in-time Deribit implied-vol smiles from historical option trades (mark prices)
and price every backtest-panel row with them.
  p_iv    : smile-consistent digital / range (-dC/dK), one-touch with sigma(H) (same as the live model)
  p_ivatm : lognormal with implied ATM vol at the target tenor (no smile)
  p_ivemp : empirical (fat-tailed, drift-neutral) shape scaled by implied ATM vol
Output: data/crypto/panel_iv.pkl (panel_tr + columns)"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from clib import *  # noqa
from bt_models import *  # noqa

WIN_H = 6          # trades from the last 6h build the smile
UMAP = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL_USDC': 'SOLUSDT', 'XRP_USDC': 'XRPUSDT'}


def load_trades():
    out = []
    for ccy in ['BTC', 'ETH', 'USDC']:
        p = os.path.join(DATA, f'deribit_trades_{ccy}.pkl')
        if not os.path.exists(p):
            continue
        d = pd.read_pickle(p)
        parts = d.inst.str.split('-', expand=True)
        d['base'] = parts[0]
        d = d[d.base.isin(UMAP)].copy()
        parts = parts.loc[d.index]
        d['und'] = d.base.map(UMAP)
        d['exp'] = parts[1].map(lambda s: parse_deribit_exp(s))
        d['K'] = parts[2].str.replace('d', '.').astype(float)
        d['cp'] = parts[3]
        d['t'] = pd.to_datetime(d.ts, unit='ms', utc=True)
        d['tau'] = (d.exp - d.t).dt.total_seconds() / YEAR_S
        d = d[d.tau > 0.5 / 8760]
        inverse = ~d.base.str.contains('USDC')
        d['F'] = d['index']
        d['px'] = np.where(inverse, d['mark'] * d['F'], d['mark'])
        out.append(d)
    d = pd.concat(out, ignore_index=True)
    # vectorised Black-76 inversion (bisection) of the mark price
    F, K, tau, px = d.F.values, d.K.values, d.tau.values, d.px.values
    call = (d.cp == 'C').values
    lo, hi = np.full(len(d), 0.01), np.full(len(d), 6.0)

    def price(sig):
        s = sig * np.sqrt(tau)
        d1 = (np.log(F / K) + 0.5 * s * s) / s
        c = F * norm.cdf(d1) - K * norm.cdf(d1 - s)
        return np.where(call, c, c - F + K)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        pm = price(mid)
        up = pm < px
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    d['sig'] = 0.5 * (lo + hi)
    intrinsic = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    ok = (px > intrinsic + 1e-9) & (d.sig > 0.02) & (d.sig < 5.5)
    otm = (call & (K >= F)) | (~call & (K < F))
    d = d[ok & otm].copy()
    d['x'] = np.log(d.K / d.F) / np.sqrt(d.tau)
    return d.sort_values('t').reset_index(drop=True)


class HistSurface:
    """Same interface as clib.Surface, fitted from trades in (t-WIN_H, t]."""

    def __init__(self, tr, t):
        self.exps = []
        w = tr[(tr.t > t - pd.Timedelta(hours=WIN_H)) & (tr.t <= t)]
        for exp, g in w.groupby('exp'):
            tau = (exp - t).total_seconds() / YEAR_S
            if tau <= 0.5 / 8760 or len(g) < 3:
                continue
            age_h = (t - g.t).dt.total_seconds().values / 3600
            wt = np.exp(-age_h / 3.0)
            x, s = g.x.values, g.sig.values
            xg = np.linspace(max(x.min(), -6), min(x.max(), 6), 25)
            if g.K.nunique() >= 4 and x.max() - x.min() > 0.5:
                c = np.polyfit(x, s, 2, w=np.sqrt(wt))
                iv = np.clip(np.polyval(c, xg), 0.05, 5)
            else:
                iv = np.full(len(xg), np.average(s, weights=wt))
            if len(xg) < 2 or not np.isfinite(iv).all():
                continue
            self.exps.append(dict(exp=exp, tau=tau, F=None, x=xg, iv=iv, atm=float(np.interp(0, xg, iv))))
        self.taus = np.array([e['tau'] for e in self.exps])

    def ok(self):
        return len(self.exps) > 0

    def fwd_ratio(self, tau):
        return 1.0

    vol = Surface.vol
    atm = Surface.atm


if __name__ == '__main__':
    tr = load_trades()
    print('deribit OTM marks', len(tr), tr.groupby('und').size().to_dict(), flush=True)
    P = pd.read_pickle(os.path.join(DATA, 'panel_tr.pkl'))
    emp = build_empirical()
    cache = {}
    res = {c: np.full(len(P), np.nan) for c in ['p_iv', 'p_ivatm', 'p_ivemp', 'iv_atm', 'iv_k']}
    trs = {u: g for u, g in tr.groupby('und')}
    t0 = time.time()
    for n, (i, r) in enumerate(P.iterrows()):
        u = r.und
        if u not in trs:
            continue
        t = pd.Timestamp(int(r.t), unit='s', tz='UTC')
        key = (u, int(r.t) // 300)
        if key not in cache:
            cache[key] = HistSurface(trs[u], t)
        surf = cache[key]
        if not surf.ok():
            continue
        rr = dict(payoff=r.payoff, K=r.K, lo=r['lo'], hi=r['hi'], T=pd.Timestamp(int(r['T']), unit='s', tz='UTC'))
        mp = model_prob(rr, r.S, t, surf, None, r.hi_sf if np.isfinite(r.hi_sf) else None,
                        r.lo_sf if np.isfinite(r.lo_sf) else None)
        res['p_iv'][i] = mp.get('p_iv', np.nan)
        res['p_ivatm'][i] = mp.get('p_atm', np.nan)
        res['iv_atm'][i] = mp.get('atm_iv', np.nan)
        res['iv_k'][i] = mp.get('sig_iv', np.nan)
        if np.isfinite(res['iv_atm'][i]):
            res['p_ivemp'][i] = emp_prob(emp, r.payoff, r.S, r.tau_s, res['iv_atm'][i], K=r.K, lo=r['lo'], hi=r['hi'])
        if n % 20000 == 0:
            print(n, len(P), round(time.time() - t0), len(cache), flush=True)
    for c, v in res.items():
        P[c] = v
    P.to_pickle(os.path.join(DATA, 'panel_iv.pkl'))
    print(P[['p_iv', 'p_ivatm', 'p_ivemp', 'iv_atm']].describe().round(3), flush=True)
