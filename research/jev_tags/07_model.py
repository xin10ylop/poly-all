"""Model-based out-of-sample test (less multiple-testing than the cell scan).
Fit P(side wins) = logistic(logit(price), tags, tags x logit(price), horizon) on disc, per side; on val buy the side
whenever model prob - all-in cost > margin. Compare with a price-only (+meta) recalibration model."""
import os, sys, re
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('NPERM', '0')
from stats import cl_stats, fmt
import importlib.util
spec = importlib.util.spec_from_file_location('an', os.path.join(os.path.dirname(__file__), '06_analyze.py'))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
P, NOUL, SCORE, LATE, U = an.P, an.NOUL, an.SCORE, an.LATE, an.U

P = P.copy()
P['lp'] = np.log(P.px / (1 - P.px))
P['nr_group'] = (P.negRisk & (P.n_ev_mkts > 1)).astype(float)
P['life'] = np.log1p((P.end - P.created) / 86400)
P['lvol'] = np.log10(P.vol)
P['h3'] = (P.hd == 3).astype(float); P['h7'] = (P.hd == 7).astype(float)
META = ['nr_group', 'life', 'lvol', 'h3', 'h7']
JEV = NOUL + SCORE


def design(df, feats):
    X = df[['lp'] + feats].copy()
    for f in feats:
        X[f + '_x_lp'] = df[f] * df.lp
    return X


def evaluate(feats, side, margins=(0.0, 0.02, 0.05, 0.1)):
    d = P[(P.side == side) & (P.split == 'disc')]
    v = P[(P.side == side) & (P.split == 'val')].copy()
    Xd, Xv = design(d, feats), design(v, feats)
    mu, sd = Xd.mean(), Xd.std().replace(0, 1)
    m = LogisticRegression(C=0.3, max_iter=2000).fit((Xd - mu) / sd, d.win)
    v['phat'] = m.predict_proba((Xv - mu) / sd)[:, 1]
    v['edge'] = v.phat - v.cost
    out = []
    for mg in margins:
        s = v[v.edge > mg]
        if len(s) >= 10:
            st = cl_stats(s)
            out.append(dict(side=side, margin=mg, **fmt(st, ('n', 'mkts', 'evs', 'px', 'hit', 'mean', 't', 'pnl_sh', 'mean_ex_top1'))))
            sl = s[s.end >= LATE]
            if len(sl) >= 10:
                stl = cl_stats(sl)
                out[-1].update(late_n=stl['n'], late_mean=round(stl['mean'], 3), late_t=round(stl['t'], 2))
    ll = -np.mean(v.win * np.log(v.phat) + (1 - v.win) * np.log(1 - v.phat))
    return out, ll, v


if __name__ == '__main__':
    res = []
    for name, feats in [('price+meta', META), ('price+meta+jev', META + JEV), ('price+jev', JEV)]:
        for side in ['YES', 'NO']:
            out, ll, v = evaluate(feats, side)
            print(f'{name:16s} {side:3s} val logloss={ll:.4f}  (entries {len(v)})')
            for o in out:
                res.append(dict(model=name, **o))
    print(pd.DataFrame(res).to_string(index=False))
