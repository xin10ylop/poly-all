"""Steps 2+3: candidate set -> daily PnL series + v2 user-stats -> consistency metrics.

Candidate superset = wallets on >=2 leaderboard periods (any category)
                     + top-60 all-time in weather / mentions / culture / tech / finance
                     + top-30 month in weather / mentions / culture.
We fetch everything (cheap: 2 requests per wallet) and rank afterwards.

Outputs (data/wallets/):
  pnl_series.csv        long format wallet,t,p (cumulative PnL, daily)
  user_stats.csv        flattened v2/user-stats
  wallet_metrics.csv    one row per wallet with consistency metrics + leaderboard flags
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import polylib as pl  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wallets")
CACHE = os.path.join(D, "cache")
os.makedirs(CACHE, exist_ok=True)
NOW = time.time()


def cached(name, fn):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    try:
        v = fn()
    except Exception as e:  # noqa: BLE001
        return {"_err": str(e)}
    with open(path, "w") as f:
        json.dump(v, f)
    return v


def fetch(w):
    s = cached(f"pnl_{w}.json", lambda: pl.get(
        "https://user-pnl-api.polymarket.com/user-pnl",
        {"user_address": w, "interval": "all", "fidelity": "1d"}))
    u = cached(f"stats_{w}.json", lambda: pl.get(f"{pl.DATA}/v2/user-stats", {"user": w}))
    time.sleep(0.05)
    return w, s, u


def candidates(lb):
    g = lb.groupby("wallet")
    nper = g.period.nunique()
    multi = set(nper[nper >= 2].index)
    top_small = set(lb[(lb.period == "all") & lb.category.isin(
        ["weather", "mentions", "culture", "tech", "finance"]) & (lb["rank"] <= 60)].wallet)
    top_month = set(lb[(lb.period == "month") & lb.category.isin(
        ["weather", "mentions", "culture"]) & (lb["rank"] <= 30)].wallet)
    return sorted(multi | top_small | top_month)


def series_metrics(pts):
    if not isinstance(pts, list) or len(pts) < 3:
        return {}
    s = pd.Series([p["p"] for p in pts], index=pd.to_datetime([p["t"] for p in pts], unit="s"))
    s = s[~s.index.duplicated(keep="last")]
    # resample to calendar days (last value), forward fill
    s = s.resample("1D").last().ffill()
    d = s.diff().dropna()
    active = d[d.abs() > 1.0]
    if len(active) == 0:
        return {}
    first = active.index[0]
    d = d[d.index >= first]  # calendar days since first activity
    total = s.iloc[-1]
    peak = s.cummax()
    dd = (s - peak)
    mdd = -dd.min()
    std = d.std()
    sharpe = d.mean() / std * np.sqrt(365) if std > 0 else np.nan
    d90 = d[d.index >= d.index[-1] - pd.Timedelta(days=90)]
    sharpe90 = d90.mean() / d90.std() * np.sqrt(365) if d90.std() > 0 else np.nan
    gains = d[d > 0].sort_values(ascending=False)
    gain_total = gains.sum()
    # weekly consistency
    wk = s.resample("W").last().diff().dropna()
    wk = wk[wk.index >= first]
    mo = s.resample("MS").last().diff().dropna()
    return {
        "series_total": total,
        "first_day": first.date().isoformat(),
        "span_days": int((d.index[-1] - first).days) + 1,
        "active_days": int(len(active)),
        "active_frac": len(active) / max(1, len(d)),
        "sharpe": sharpe,
        "sharpe_90d": sharpe90,
        "pct_pos_days": float((active > 0).mean()),
        "pct_pos_weeks": float((wk > 0).mean()) if len(wk) else np.nan,
        "n_weeks": int(len(wk)),
        "pct_pos_months": float((mo > 0).mean()) if len(mo) else np.nan,
        "max_dd": mdd,
        "max_dd_frac_peak": mdd / peak.max() if peak.max() > 0 else np.nan,
        "max_day_gain": gains.iloc[0] if len(gains) else 0.0,
        "lump_1d": gains.iloc[0] / total if total > 0 and len(gains) else np.nan,
        "lump_top5d": gains.iloc[:5].sum() / gain_total if gain_total > 0 else np.nan,
        "worst_day": d.min(),
        "pnl_7d": s.iloc[-1] - s.asof(s.index[-1] - pd.Timedelta(days=7)),
        "pnl_30d": s.iloc[-1] - s.asof(s.index[-1] - pd.Timedelta(days=30)),
        "pnl_90d": s.iloc[-1] - s.asof(s.index[-1] - pd.Timedelta(days=90)),
        "mean_daily": d.mean(),
    }


def flat_stats(u):
    if not isinstance(u, dict) or not isinstance(u.get("data"), dict):
        return {}
    x = u["data"]
    a = x.get("all_time_pnl") or {}
    out = {"n_markets": x.get("trades"), "biggest_win": x.get("biggest_win"),
           "join_date": pd.to_datetime(x.get("join_date"), unit="s").date().isoformat()
           if x.get("join_date") else None}
    for k in ["realized_pnl", "unrealized_pnl", "economic_pnl", "trade_pnl", "fees_paid",
              "maker_rebate", "taker_rebate", "reward_income", "yield_income", "volume",
              "volume_usdc", "trade_count"]:
        out[k] = a.get(k)
    return out


def main():
    lb = pd.read_csv(os.path.join(D, "leaderboards.csv"))
    cands = candidates(lb)
    print(len(cands), "candidates")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(fetch, cands))

    ser_rows, stat_rows, met_rows = [], [], []
    for w, s, u in res:
        if isinstance(s, list):
            ser_rows += [{"wallet": w, "t": p["t"], "p": p["p"]} for p in s]
        st = flat_stats(u)
        stat_rows.append({"wallet": w, **st})
        met_rows.append({"wallet": w, **series_metrics(s), **st})
    pd.DataFrame(ser_rows).to_csv(os.path.join(D, "pnl_series.csv"), index=False)
    pd.DataFrame(stat_rows).to_csv(os.path.join(D, "user_stats.csv"), index=False)

    m = pd.DataFrame(met_rows)
    # leaderboard context
    lb["tag"] = lb.period + ":" + lb.category + "#" + lb["rank"].astype(str)
    g = lb.groupby("wallet")
    ctx = pd.DataFrame({
        "name": g.name.first(),
        "lb_periods": g.period.apply(lambda x: ",".join(sorted(set(x)))),
        "lb_cats": g.category.apply(lambda x: ",".join(sorted(set(x) - {"overall"}))),
        "lb_best": g.apply(lambda x: ";".join(x.sort_values("rank").tag.head(4))),
        "lb_max_roi": g.apply(lambda x: (x.pnl / x.vol.replace(0, np.nan)).max()),
    })
    m = m.merge(ctx, left_on="wallet", right_index=True, how="left")
    m["roi_vol"] = m.economic_pnl / m.volume.replace(0, np.nan)
    m["lump_bigwin"] = m.biggest_win / m.series_total.where(m.series_total > 0)
    m["pnl_per_market"] = m.economic_pnl / m.n_markets.replace(0, np.nan)
    m["avg_trade_usdc"] = m.volume_usdc / m.trade_count.replace(0, np.nan)
    m["fees_frac_vol"] = -m.fees_paid / m.volume.replace(0, np.nan)
    m.to_csv(os.path.join(D, "wallet_metrics.csv"), index=False)
    print(m.describe().T.to_string())


if __name__ == "__main__":
    main()
