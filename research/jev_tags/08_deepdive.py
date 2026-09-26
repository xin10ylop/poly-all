"""Deep-dive on candidate segments: robustness by split / horizon / price mode, concentration, capacity."""
import os, sys, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from stats import cl_stats, fmt
pd.set_option('display.width', 250)


def load(px):
    os.environ['PX'] = px
    spec = importlib.util.spec_from_file_location('an_' + px, os.path.join(os.path.dirname(__file__), '06_analyze.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    P = m.P.copy()
    P = P.join(m.LIT.add_prefix('L_'))
    return P, m


CELLS = [  # (name, side, bucket or None, literal list)
    ('hype_hi&illiquid NO 65-85 [scan winner, first-print]', 'NO', '65-85', ['hype_hi', 'illiquid']),
    ('hype_hi NO 65-85', 'NO', '65-85', ['hype_hi']),
    ('person&hype_hi NO 65-85', 'NO', '65-85', ['person', 'hype_hi']),
    ('mention&hype_hi NO 65-85', 'NO', '65-85', ['mention', 'hype_hi']),
    # a-priori hypotheses from the task: YES needs something new + hype/drama => YES overpriced => buy NO
    ('sq_no NO all', 'NO', None, ['sq_no']),
    ('sq_no&hype_hi NO all', 'NO', None, ['sq_no', 'hype_hi']),
    ('sq_no&drama_hi NO all', 'NO', None, ['sq_no', 'drama_hi']),
    ('sq_no&drama_hi NO 65-85', 'NO', '65-85', ['sq_no', 'drama_hi']),
    ('sq_no&drama_hi NO 85-97', 'NO', '85-97', ['sq_no', 'drama_hi']),
    ('sq_no&hype_hi NO 85-97', 'NO', '85-97', ['sq_no', 'hype_hi']),
    ('person NO all', 'NO', None, ['person']),
    ('mention NO all', 'NO', None, ['mention']),
    ('ent NO all', 'NO', None, ['ent']),
    ('conflict NO all', 'NO', None, ['conflict']),
    ('sq_no YES all (mirror)', 'YES', None, ['sq_no']),
]
LATE = pd.Timestamp('2026-07-01', tz='UTC').timestamp()

if __name__ == '__main__':
    out = []
    for px in ['vwap', 'first']:
        P, m = load(px)
        for name, side, bk, lits in CELLS:
            mask = (P.side == side) & ((P.bk == bk) if bk else True)
            for l in lits:
                mask &= P['L_' + l]
            g = P[mask]
            for split, s in [('disc', g[g.split == 'disc']), ('val', g[g.split == 'val']), ('late', g[g.end >= LATE])]:
                if len(s) >= 10:
                    st = cl_stats(s)
                    out.append(dict(mode=px, cell=name, split=split, **fmt(st, ('n', 'evs', 'series', 'px', 'hit', 'mean', 't', 'pnl_sh', 'top_series_share', 'mean_ex_top1'))))
    R = pd.DataFrame(out)
    print(R.to_string(index=False))

    # focus: scan winner, per horizon + concentration + capacity (first-print prices)
    P, m = load('first')
    g = P[(P.side == 'NO') & (P.bk == '65-85') & P.L_hype_hi & P.L_illiquid]
    print('\n--- hype_hi&illiquid NO 65-85 (first print) by horizon / split')
    for (sp, hd), s in g.groupby(['split', 'hd']):
        print(sp, hd, fmt(cl_stats(s), ('n', 'evs', 'px', 'hit', 'mean', 't', 'pnl_sh')))
    v = g[g.split == 'val']
    weeks = (v.end.max() - v.end.min()) / 86400 / 7
    print('val weeks %.1f  entries/week %.1f  distinct markets/week %.1f' % (weeks, len(v) / weeks, v.mid.nunique() / weeks))
    print('window $ on NO side (all takers, 12h): median %.0f  mean %.0f  p90 %.0f' % (v.usd.median(), v.usd.mean(), v.usd.quantile(.9)))
    print('top series by #entries:', g.series.value_counts().head(10).to_dict())
    print('top series by val pnl:', v.groupby('series').pnl.sum().sort_values().tail(5).round(2).to_dict())
    print('sample questions:', m.U.loc[g.mid.unique()[:12], 'q'].tolist())
