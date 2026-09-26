"""Summarise a scanner run (+ ladder run) into tables used by REPORT.md.

Usage: python research/arb/analyze.py [run_dir] [ladder_run_dir]
Prints markdown-ish tables and writes data/arb/summary.json.
"""
import glob, gzip, json, os, sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data', 'arb')
runs = sorted(d for d in glob.glob(f'{DATA}/run_*') if not d.endswith(('_ladder', '_hot', '_inplay')))
lruns = sorted(glob.glob(f'{DATA}/run_*_ladder'))
hruns = sorted(glob.glob(f'{DATA}/run_*_hot'))
RUNS = sys.argv[1].split(',') if len(sys.argv) > 1 else runs
LRUN = sys.argv[2] if len(sys.argv) > 2 else (lruns[-1] if lruns else None)
pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_colwidth', 70)


def rj(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    op = gzip.open if path.endswith('.gz') else open
    with op(path, 'rt') as f:
        return pd.DataFrame([json.loads(l) for l in f if l.strip()])


def q(x, ps=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    x = pd.Series(x).dropna()
    return {f'p{int(p*100)}': round(float(x.quantile(p)), 4) for p in ps} | {'n': int(len(x)), 'sum': round(float(x.sum()), 2)}


S = {}


def cat(name):
    fr = []
    for i, r in enumerate(RUNS):
        d = rj(f'{r}/{name}')
        if len(d):
            d['run'] = i
            if 'sweep' in d:
                d['sweep'] = f'r{i}_' + d['sweep'].astype(str)
            fr.append(d)
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


sw = cat('sweeps.jsonl')
op = cat('opps.jsonl')
nx = cat('opps_nonexh.jsonl')
rc = cat('rechecks.jsonl')
print('RUNS', RUNS)
for kind, g in sw.groupby('kind'):
    span = sum((h.t0.max() - h.t0.min()) / 60 for _, h in g.groupby('run'))
    S[f'sweeps_{kind}'] = dict(n=len(g), span_min=round(span, 1), fetch_s_med=float(g.fetch_s.median()),
                               n_tok_med=int(g.n_tok.median()), n_books_med=int(g.n_books.median()),
                               n_req_total=int(g.n_req.sum()))
    print(kind, S[f'sweeps_{kind}'])
hours = sum((g.t0.max() - g.t0.min()) / 3600 for _, g in sw.groupby('run')) if len(sw) else 0
nr_sweeps = sorted(sw[sw.kind == 'nr'].sweep, key=lambda s: (int(s.split('_')[0][1:]), int(s.split('_')[1][2:])))
nr_idx = {s: i for i, s in enumerate(nr_sweeps)}
S['n_nr_sweeps'] = len(nr_sweeps)
print('scanner hours covered', round(hours, 2))

# ------------------------------------------------------------------ opportunity table
if len(op):
    op['cat'] = np.where(op.fee_killed, 'gross_only(fee-killed)', np.where(op['exec'], 'exec(net>0,>=5sh)', 'net>0 but <5sh'))
    op['key'] = op['type'] + '|' + op['gid'].fillna(op.get('mid', pd.Series(dtype=str)).astype(str))
    print('\n=== detections by type x category (rows = sweep-detections; uniq = distinct event/market)')
    t = op.groupby(['type', 'cat']).agg(detections=('key', 'size'), uniq=('key', 'nunique'))
    print(t)
    S['by_type'] = {f'{a}|{b}': dict(r) for (a, b), r in t.iterrows()}
    ex = op[op['exec']].copy()
    if len(ex):
        ex['nr_i'] = ex.sweep.map(nr_idx)
        g = ex.groupby('key')
        u = g.agg(type=('type', 'first'), title=('title', 'first'), exh=('exh', 'first'), N=('N', 'first'),
                  flags=('flags', 'first'), first=('nr_i', 'min'), last=('nr_i', 'max'), seen=('nr_i', 'nunique'),
                  profit_first=('profit', 'first'), profit_med=('profit', 'median'), profit_max=('profit', 'max'),
                  units_med=('units', 'median'), notional_med=('notional', 'median'), edge_med=('top_net', 'median'))
        u['span'] = u['last'] - u['first'] + 1
        u['preexisting'] = u['first'].isin([0, nr_idx.get(next((x for x in nr_sweeps if x.startswith('r1_')), ''), -1)])
        print('\n=== executable opportunities (unique)')
        print(u.sort_values('profit_med', ascending=False).to_string())
        S['exec_unique'] = u.reset_index().to_dict('records')
        S['exec_profit_dist'] = q(u.profit_med)
        S['exec_units_dist'] = q(u.units_med)
    # fee-killed NO-basket gross edges
    nb = op[(op.type.isin(['nr_buy_no', 'nr_sell_yes']))]
    if len(nb):
        print('\n=== NO-basket (sum YES bids > 1) gross edges, per unique event (median over sweeps)')
        gg = nb.groupby('gid').agg(title=('title', 'first'), N=('N', 'first'), gross=('top_gross', 'median'),
                                   net=('top_net', 'median'), seen=('sweep', 'nunique'))
        print(gg.sort_values('gross', ascending=False).head(25).to_string())
        S['nobasket_gross'] = q(gg.gross)
        S['nobasket_net'] = q(gg.net)
    bn = op[op.type.str.startswith('bin_')]
    S['binary_detections'] = int(len(bn))

# ------------------------------------------------------------------ rechecks
if len(rc):
    print('\n=== rechecks (fraction of flagged opportunities still present / executable after delay)')
    rt = rc.groupby(['type', 'delay']).agg(n=('key', 'size'), present=('present', 'mean'), exec=('exec', 'mean'),
                                           profit_ratio=('profit', lambda s: 0))
    rc['ratio'] = rc.profit / rc.orig_profit.replace(0, np.nan)
    rt['profit_ratio_med'] = rc.groupby(['type', 'delay']).ratio.median()
    rt = rt.drop(columns='profit_ratio')
    print(rt)
    S['rechecks'] = {f'{a}|{b}': dict(r) for (a, b), r in rt.iterrows()}

# ------------------------------------------------------------------ near-miss stats per negRisk event
ns = cat('nrstats.jsonl.gz')
if len(ns):
    rows = []
    for _, r in ns.iterrows():
        for x in r['rows']:
            rows.append([r['sweep']] + x)
    st = pd.DataFrame(rows, columns=['sweep', 'gid', 'N', 's_ay', 's_ayf', 'n_ay_miss', 's_by', 's_byf', 'n_by_miss',
                                     'nobasket', 'maxbid'])
    st['nb_gross'] = st.s_by - 1
    st['nb_net'] = st.s_byf - 1
    full = st[st.n_ay_miss == 0]
    st['by_gross'] = 1 - st.s_ay
    print('\n=== NO-basket edge (sum of top YES bids - 1) across all event-sweeps')
    for lab, col in [('gross', 'nb_gross'), ('net of fees', 'nb_net')]:
        v = st[col]
        d = dict(n=len(v), gt0=int((v > 0).sum()), gt_m0005=int((v > -0.005).sum()), gt_m001=int((v > -0.01).sum()),
                 max=round(float(v.max()), 4))
        print(lab, d)
        S[f'nb_{col}'] = d
    v = 1 - full.s_ayf
    S['buyyes_net_all_asks_present'] = dict(n=len(v), gt0=int((v > 0).sum()), gt_m001=int((v > -0.01).sum()))
    print('buy-all-YES (all asks present) net edge>0 event-sweeps:', S['buyyes_net_all_asks_present'])
    per_ev = st.groupby('gid').agg(nb_net_max=('nb_net', 'max'), nb_gross_max=('nb_gross', 'max'), N=('N', 'first'))
    S['events_nb_gross_pos'] = int((per_ev.nb_gross_max > 0).sum())
    S['events_nb_net_pos'] = int((per_ev.nb_net_max > 0).sum())
    S['n_events'] = int(len(per_ev))
    print('events ever gross>0:', S['events_nb_gross_pos'], 'net>0:', S['events_nb_net_pos'], 'of', len(per_ev))

# ------------------------------------------------------------------ ladder run
if LRUN:
    print('\nLADDER RUN', LRUN)
    ls = rj(f'{LRUN}/lad_sweeps.jsonl')
    lo = rj(f'{LRUN}/lad_opps.jsonl')
    lr = rj(f'{LRUN}/lad_rechecks.jsonl')
    if len(ls):
        S['sweeps_ladder'] = dict(n=len(ls), span_min=round((ls.t0.max() - ls.t0.min()) / 60, 1),
                                  fetch_s_med=float(ls.fetch_s.median()), n_tok=int(ls.n_tok.median()))
        print(S['sweeps_ladder'])
    if len(lo):
        lo['key'] = lo.easy_id.astype(str) + '>' + lo.hard_id.astype(str)
        lo['cat'] = np.where(lo.fee_killed, 'gross_only', np.where(lo['exec'], 'exec', 'net<5sh'))
        lsw = sorted(lo.sweep.unique(), key=lambda s: int(s[3:]))
        li = {s: i for i, s in enumerate(sorted(ls.sweep, key=lambda s: int(s[3:])))}
        lo['i'] = lo.sweep.map(li)
        t = lo.groupby('cat').agg(detections=('key', 'size'), uniq=('key', 'nunique'), events=('eid', 'nunique'))
        print(t)
        S['ladder_by_cat'] = {k: dict(r) for k, r in t.iterrows()}
        le = lo[lo['exec']]
        u = le.groupby('key').agg(title=('title', 'first'), easy=('easy', 'first'), hard=('hard', 'first'),
                                  kind=('kind', 'first'), first=('i', 'min'), last=('i', 'max'), seen=('i', 'nunique'),
                                  profit_med=('profit', 'median'), profit_max=('profit', 'max'),
                                  units_med=('units', 'median'), edge_med=('top_net', 'median'),
                                  end=('endDate', lambda s: s.iloc[0][1]), viol=('n_viol_pairs', 'first'),
                                  npairs=('n_pairs', 'first'))
        print(u.sort_values('profit_med', ascending=False).head(40).to_string())
        S['ladder_exec_unique'] = u.reset_index().to_dict('records')
        S['ladder_profit_dist'] = q(u.profit_med)
    if len(lr):
        lr['ratio'] = lr.profit / lr.orig_profit.replace(0, np.nan)
        rt = lr.groupby('delay').agg(n=('key', 'size'), present=('present', 'mean'), exec=('exec', 'mean'),
                                     ratio_med=('ratio', 'median'))
        print(rt)
        S['ladder_rechecks'] = {int(k): dict(r) for k, r in rt.iterrows()}

for hr in hruns:
    hp = rj(f'{hr}/hot_polls.jsonl')
    ho = rj(f'{hr}/hot_opps.jsonl')
    if len(hp):
        S['hot'] = dict(polls=len(hp), minutes=round((hp.t0.max() - hp.t0.min()) / 60, 1),
                        cadence_s=round((hp.t0.max() - hp.t0.min()) / max(1, len(hp) - 1), 2),
                        opp_rows=len(ho), exec_rows=int(ho['exec'].sum()) if len(ho) else 0,
                        nb_net_max=float(hp.nb_net_max.max()))
        print('HOT', hr, S['hot'])
        if len(ho):
            print(ho.groupby(['title', 'type']).agg(n=('sweep', 'size'), units=('units', 'max'), profit=('profit', 'max'),
                                                   exec=('exec', 'sum')))
S['hours'] = hours
with open(f'{DATA}/summary.json', 'w') as f:
    json.dump(S, f, default=str, indent=1)
