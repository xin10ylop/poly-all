"""Backtest-side models: trailing realized vol, DVOL lookup, and an empirical
(filtered-historical-simulation) model for terminal / max / min distributions."""
import math
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import norm

from clib import DATA, YEAR_S, p_touch

UNDS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
HGRID_MIN = np.array([5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360, 480, 720, 960, 1440, 2160, 2880, 4320,
                      5760, 7200, 10080, 14400, 20160, 30240, 43200])
TRAIN_END = pd.Timestamp('2026-07-13', tz='UTC')
PER5 = 105120.0


def load_k(u, iv):
    return pd.read_pickle(os.path.join(DATA, f'kl{iv}_{u}.pkl'))


def rv_frame(k5):
    """Trailing vol estimates (annualized) available at time index (= bar close time)."""
    r = np.log(k5['c']).diff()
    r2 = r ** 2
    f = pd.DataFrame(index=k5.index + pd.Timedelta(minutes=5))
    f['rv1d'] = np.sqrt(r2.rolling(288, min_periods=200).mean() * PER5).values
    f['rv7d'] = np.sqrt(r2.rolling(2016, min_periods=1500).mean() * PER5).values
    f['rv30d'] = np.sqrt(r2.rolling(8640, min_periods=6000).mean() * PER5).values
    f['rv'] = np.sqrt(0.45 * f.rv1d ** 2 + 0.35 * f.rv7d ** 2 + 0.2 * f.rv30d ** 2)
    return f


def build_empirical(force=False):
    """Standardized terminal return z, running max u, running min d for each grid horizon,
    pooled over the 4 underlyings, training window [2025-10-01, TRAIN_END)."""
    path = os.path.join(DATA, 'emp_dist.pkl')
    if os.path.exists(path) and not force:
        return pickle.load(open(path, 'rb'))
    out = {int(h): {'z': [], 'u': [], 'd': []} for h in HGRID_MIN}
    for u in UNDS:
        k5 = load_k(u, '5m')
        rv = rv_frame(k5)['rv'].values          # at close of bar i (index shifted)
        c, hi, lo = np.log(k5['c'].values), np.log(k5['h'].values), np.log(k5['l'].values)
        idx = k5.index
        ok = (idx >= pd.Timestamp('2025-10-01', tz='UTC'))
        for h in HGRID_MIN:
            n = int(h // 5)
            s_hi = pd.Series(hi).rolling(n).max().shift(-n).values
            s_lo = pd.Series(lo).rolling(n).min().shift(-n).values
            ret = np.roll(c, -n) - c
            end_t = idx + pd.Timedelta(minutes=5 * (n + 1))
            sel = ok & (end_t < TRAIN_END) & np.isfinite(rv) & np.isfinite(s_hi)
            sel_idx = np.where(sel)[0][::6]          # start every 30 min
            sc = rv[sel_idx] * math.sqrt(h * 60 / YEAR_S)
            out[int(h)]['z'].append(ret[sel_idx] / sc)
            out[int(h)]['u'].append((s_hi[sel_idx] - c[sel_idx]) / sc)
            out[int(h)]['d'].append((s_lo[sel_idx] - c[sel_idx]) / sc)
    for h in out:
        z, u, d = (np.concatenate(out[h][key]) for key in ('z', 'u', 'd'))
        # symmetrize (drift-neutral): training window had a strong trend
        out[h] = {'z': np.sort(np.concatenate([z, -z])), 'u': np.sort(np.concatenate([u, -d])),
                  'd': np.sort(np.concatenate([d, -u]))}
    pickle.dump(out, open(path, 'wb'))
    return out


def _grid_h(tau_min):
    i = int(np.argmin(np.abs(np.log(HGRID_MIN) - math.log(max(tau_min, 5)))))
    return int(HGRID_MIN[i])


def emp_prob(emp, payoff, S, tau_s, sig, K=None, lo=None, hi=None):
    """P(YES) from pooled standardized empirical distributions."""
    if tau_s <= 0:
        return np.nan
    lo, hi, K = _nn(lo), _nn(hi), _nn(K)
    h = _grid_h(tau_s / 60)
    sc = sig * math.sqrt(tau_s / YEAR_S)
    E = emp[h]

    def frac_ge(arr, x):
        return 1.0 - np.searchsorted(arr, x, side='left') / len(arr)

    if payoff == 'digital':
        return frac_ge(E['z'], math.log(K / S) / sc)
    if payoff == 'range':
        a = 1.0 if lo is None else frac_ge(E['z'], math.log(lo / S) / sc)
        b = 0.0 if hi is None else frac_ge(E['z'], math.log(hi / S) / sc)
        return a - b
    if payoff == 'touch_up':
        return frac_ge(E['u'], math.log(K / S) / sc)
    if payoff == 'touch_down':
        return np.searchsorted(E['d'], math.log(K / S) / sc, side='right') / len(E['d'])
    return np.nan


def _nn(x):
    """NaN -> None (pandas turns None into NaN in float columns)."""
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def gbm_prob(payoff, S, tau_s, sig, K=None, lo=None, hi=None):
    lo, hi, K = _nn(lo), _nn(hi), _nn(K)
    tau = tau_s / YEAR_S
    if tau <= 0 or not np.isfinite(sig):
        return np.nan
    st = sig * math.sqrt(tau)

    def above(x):
        return float(norm.cdf((math.log(S / x) - 0.5 * st * st) / st))

    if payoff == 'digital':
        return above(K)
    if payoff == 'range':
        return (1.0 if lo is None else above(lo)) - (0.0 if hi is None else above(hi))
    if payoff in ('touch_up', 'touch_down'):
        return p_touch(S, K, sig, tau, 0.0)
    return np.nan
