"""Event-clustered statistics for per-entry returns."""
import numpy as np, pandas as pd


def cl_stats(df, col='ret', cl='eid'):
    """Mean of df[col] with cluster-robust (by event) SE; plus concentration diagnostics."""
    n = len(df)
    if n == 0:
        return dict(n=0)
    r = df[col].to_numpy(float)
    m = r.mean()
    g = pd.Series(r - m).groupby(df[cl].to_numpy()).sum()
    G = len(g)
    se = np.sqrt((g ** 2).sum() * G / max(G - 1, 1)) / n if G > 1 else np.nan
    ev = df.groupby(cl)[col].sum().sort_values(ascending=False)
    tot = ev.sum()
    top1 = ev.index[0]
    rest = df[df[cl] != top1][col]
    ser_n = df.series.nunique() if 'series' in df else np.nan
    ser_top = (df.groupby('series')[col].sum().max() / tot) if ('series' in df and tot > 0) else np.nan
    return dict(n=n, mkts=df.mid.nunique(), evs=G, series=ser_n, top_series_share=ser_top,
                mean=m, se=se, t=m / se if se and se > 0 else np.nan,
                hit=df.win.mean(), px=df.px.mean(), pnl_sh=df.pnl.mean(),
                top1_share=ev.iloc[0] / tot if tot > 0 else np.nan,
                top3_share=ev.iloc[:3].sum() / tot if tot > 0 else np.nan,
                mean_ex_top1=rest.mean() if len(rest) else np.nan)


def fmt(d, keys=('n', 'mkts', 'evs', 'px', 'hit', 'mean', 't', 'pnl_sh', 'top3_share', 'mean_ex_top1')):
    return {k: (round(d[k], 3) if isinstance(d.get(k), float) else d.get(k)) for k in keys}
