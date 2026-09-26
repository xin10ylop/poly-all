"""Calibration / executable-return analysis of Jev-tagged segments with a time split.
disc = markets ending before 2026-05-01; val = ending on/after; late = ending on/after 2026-07-01 (subset of val)."""
import os, sys, pickle, itertools, json, re
import numpy as np, pandas as pd
import scipy.sparse as sp
sys.path.insert(0, os.path.dirname(__file__))
from stats import cl_stats, fmt

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
D = os.path.join(ROOT, 'data', 'jev_tags')
pd.set_option('display.width', 250)
RET = os.environ.get('RET', 'ret')      # ret (conservative 4% default fee) | ret_a (listed fee only) | ret_first
LO, HI = 0.03, 0.97

U = pd.read_pickle(f'{D}/universe.pkl').set_index('mid')
P = pd.read_pickle(f'{D}/panel.pkl')
T = pd.DataFrame(pickle.load(open(f'{D}/tags.pkl', 'rb'))).T
T.index.name = 'mid'
NOUL = ['sq_no', 'person', 'sched', 'num', 'mention', 'ent', 'ambig', 'conflict', 'govt']
SCORE = ['drama', 'hype']
P = P[(P.px >= LO) & (P.px <= HI)].copy()
P = P[P.mid.isin(T.index)].join(T[NOUL + SCORE].astype(float), on='mid')
P = P.join(U[['created', 'ev_slug']], on='mid')
MONTHS = 'january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec|week|weekly|q[1-4]'
P['series'] = P.ev_slug.str.replace(r'\d+', '', regex=True).str.replace(rf'(^|-)({MONTHS})(?=-|$)', '-', regex=True).str.replace(r'-+', '-', regex=True).str.strip('-')
P['ret'] = P[RET]
DISC_END = pd.Timestamp('2026-05-01', tz='UTC').timestamp()
LATE = pd.Timestamp('2026-07-01', tz='UTC').timestamp()
P['split'] = np.where(P.end < DISC_END, 'disc', 'val')
BK = [0.03, 0.15, 0.35, 0.65, 0.85, 0.9701]
P['bk'] = pd.cut(P.px, BK, right=False, labels=['03-15', '15-35', '35-65', '65-85', '85-97'])

# ---- binary literals -------------------------------------------------------------------------
disc = P[P.split == 'disc'].drop_duplicates('mid')
Lit = {}
BIN = [f for f in NOUL if f != 'ambig']
TERC = SCORE + ['ambig']            # ambig rarely > 0.5, so use discovery terciles like the scores
for f in BIN:
    Lit[f] = P[f] > 0.5
    Lit['no_' + f] = P[f] <= 0.5
for f in TERC:
    lo, hi = disc[f].quantile([1 / 3, 2 / 3])
    Lit[f + '_hi'] = P[f] >= hi
    Lit[f + '_lo'] = P[f] <= lo
Lit['nr_group'] = P.negRisk & (P.n_ev_mkts > 1)
Lit['standalone'] = ~Lit['nr_group']
dur = (P.end - P.created) / 86400
Lit['short_life'] = dur <= 14
Lit['long_life'] = dur > 90
Lit['liquid'] = P.vol >= 50000
Lit['illiquid'] = P.vol < 50000
LIT = pd.DataFrame(Lit)
FAM = {k: k.replace('no_', '').replace('_hi', '').replace('_lo', '') for k in LIT}
FAM.update(nr_group='grp', standalone='grp', short_life='life', long_life='life', liquid='liq', illiquid='liq')


def scan(df, lit, min_ev=20, pairs=True):
    """Vectorized cluster-robust mean/t for every literal and literal pair within df (one side x bucket)."""
    r = df.ret.to_numpy(float)
    ev_codes, ev_idx = np.unique(df.eid.to_numpy(), return_inverse=True)
    E = sp.csr_matrix((np.ones(len(df)), (ev_idx, np.arange(len(df)))), shape=(len(ev_codes), len(df)))
    L = lit.loc[df.index].to_numpy(bool)
    names = list(lit.columns)
    cols, labels = [], []
    for i, a in enumerate(names):
        cols.append(L[:, i]); labels.append(a)
    if pairs:
        for i, j in itertools.combinations(range(len(names)), 2):
            if FAM[names[i]] == FAM[names[j]]:
                continue
            cols.append(L[:, i] & L[:, j]); labels.append(names[i] + '&' + names[j])
    M = np.column_stack(cols).astype(float)
    N = M.sum(0)
    S = M.T @ r
    mean = np.divide(S, N, out=np.full_like(S, np.nan), where=N > 0)
    Sg = (E @ (M * r[:, None]))
    Cg = (E @ M)
    Gn = (Cg > 0).sum(0)
    res = Sg - Cg * mean[None, :]
    var = (res ** 2).sum(0) * Gn / np.maximum(Gn - 1, 1) / np.maximum(N, 1) ** 2
    t = mean / np.sqrt(var)
    out = pd.DataFrame({'cell': labels, 'n': N, 'evs': Gn, 'mean': mean, 't': t})
    return out[out.evs >= min_ev]


def run_scan(P, lit, tag=''):
    rows = []
    for side in ['YES', 'NO']:
        for bk in list(P.bk.cat.categories) + ['all']:
            sub = P[(P.side == side) & ((P.bk == bk) if bk != 'all' else True)]
            d, v = sub[sub.split == 'disc'], sub[sub.split == 'val']
            if len(d) < 50:
                continue
            sd = scan(d, lit).set_index('cell')
            sv = scan(v, lit, min_ev=1).set_index('cell')
            base_d, base_v = d.ret.mean(), v.ret.mean()
            x = sd.join(sv, rsuffix='_v', how='left')
            x['side'], x['bk'], x['base_d'], x['base_v'] = side, bk, base_d, base_v
            rows.append(x.reset_index())
    return pd.concat(rows, ignore_index=True)


if __name__ == '__main__':
    print('ret column:', RET, '| entries', len(P), 'markets', P.mid.nunique(), 'events', P.eid.nunique())
    print(P.groupby(['split', 'side']).agg(n=('ret', 'size'), mkts=('mid', 'nunique'), evs=('eid', 'nunique')).to_string())

    # ---- A. baseline calibration by side x price bucket (no tags) ------------------------------
    print('\n=== A. Baseline: buy side at executable VWAP (+fee), by side x side-price bucket x split ===')
    rows = []
    for (side, bk, split), g in P.groupby(['side', 'bk', 'split'], observed=True):
        rows.append(dict(side=side, bk=bk, split=split, **fmt(cl_stats(g))))
    for (side, bk), g in P[P.end >= LATE].groupby(['side', 'bk'], observed=True):
        rows.append(dict(side=side, bk=bk, split='late', **fmt(cl_stats(g))))
    A = pd.DataFrame(rows).sort_values(['side', 'bk', 'split'])
    print(A.to_string(index=False))

    # ---- B. single-literal table --------------------------------------------------------------
    print('\n=== B. Single tags, all prices in [0.03,0.97], by side (disc vs val) ===')
    rows = []
    for side in ['YES', 'NO']:
        for lit in LIT.columns:
            for split in ['disc', 'val']:
                g = P[(P.side == side) & (P.split == split) & LIT.loc[P.index, lit]]
                if len(g) > 20:
                    rows.append(dict(side=side, tag=lit, split=split, **fmt(cl_stats(g), ('n', 'evs', 'px', 'hit', 'mean', 't'))))
    B = pd.DataFrame(rows).pivot_table(index=['side', 'tag'], columns='split', values=['n', 'evs', 'px', 'mean', 't'])
    print(B.round(3).to_string())

    # ---- C. scan singles + pairs x side x bucket on disc, validate -----------------------------
    S = run_scan(P, LIT)
    S['excess_d'] = S['mean'] - S.base_d
    S['excess_v'] = S['mean_v'] - S.base_v
    S.to_pickle(f'{D}/scan_{RET}.pkl')
    sel = S[(S.t >= 3) & (S['mean'] > 0) & (S.evs >= 20)].sort_values('t', ascending=False)
    print(f'\n=== C. Discovery scan: {len(S)} cells tested; {len(sel)} with disc t>=3, mean>0, >=20 events ===')
    print(sel[['side', 'bk', 'cell', 'n', 'evs', 'mean', 't', 'excess_d', 'n_v', 'evs_v', 'mean_v', 't_v', 'excess_v']]
          .head(60).round(3).to_string(index=False))
    ok = sel[(sel.mean_v > 0) & (sel.t_v >= 2)]
    print(f'validated (val mean>0 & t_v>=2): {len(ok)} of {len(sel)}; val mean>0: {(sel.mean_v > 0).sum()}')
    print('median val mean of selected:', sel.mean_v.median(), ' weighted by n_v:', np.average(sel.mean_v.fillna(0), weights=sel.n_v.fillna(0)) if sel.n_v.sum() else None)

    # ---- D. permutation null: shuffle tag profiles across events (keep within-event structure) --
    rng = np.random.default_rng(0)
    null = []
    for it in range(int(os.environ.get('NPERM', 10))):
        evs = P.eid.unique()
        donor = dict(zip(evs, rng.permutation(evs)))
        mk_by_ev = P.groupby('eid').mid.unique()
        mapping = {}
        for e, ms in mk_by_ev.items():
            dm = mk_by_ev[donor[e]]
            for m in ms:
                mapping[m] = dm[rng.integers(len(dm))]
        Tm = T[NOUL + SCORE].astype(float)
        Pp = P.copy()
        Pp[NOUL + SCORE] = Tm.loc[Pp.mid.map(mapping)].to_numpy()
        Lp = LIT.copy()
        for f in BIN:
            Lp[f] = Pp[f] > 0.5; Lp['no_' + f] = Pp[f] <= 0.5
        for f in TERC:
            lo, hi = disc[f].quantile([1 / 3, 2 / 3])
            Lp[f + '_hi'] = Pp[f] >= hi; Lp[f + '_lo'] = Pp[f] <= lo
        Sp = run_scan(Pp, Lp)
        # only cells involving a (shuffled) Jev literal are comparable
        jevcell = Sp.cell.str.contains('|'.join(NOUL + SCORE))
        sp_sel = Sp[jevcell & (Sp.t >= 3) & (Sp['mean'] > 0)]
        null.append(dict(it=it, n_sel=len(sp_sel), n_val=int(((sp_sel.mean_v > 0) & (sp_sel.t_v >= 2)).sum())))
    jevcell = S.cell.str.contains('|'.join(NOUL + SCORE))
    real = dict(n_sel=int((jevcell & (S.t >= 3) & (S['mean'] > 0)).sum()),
                n_val=int((jevcell & (S.t >= 3) & (S['mean'] > 0) & (S.mean_v > 0) & (S.t_v >= 2)).sum()))
    print('\n=== D. Permutation null (Jev-tag cells only) ===')
    print('real:', real)
    print(pd.DataFrame(null).describe().loc[['mean', 'std', 'max']].round(2).to_string())
