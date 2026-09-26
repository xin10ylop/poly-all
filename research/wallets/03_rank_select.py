"""Rank wallets by PnL consistency and pick a category-diverse deep-dive list.

Inputs : data/wallets/wallet_metrics.csv, leaderboards.csv
Outputs: data/wallets/ranked_consistent.csv   (filtered + scored universe)
         data/wallets/deep_dive.csv           (wallets for trade-level analysis)
"""
import os

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wallets")

# primary leaderboard category = the non-overall category where the wallet ranks best
PRIMARY_ORDER = ["weather", "mentions", "culture", "tech", "finance", "economics",
                 "crypto", "sports", "politics"]


def primary_cat(lb):
    x = lb[lb.category != "overall"].copy()
    x["w"] = x.period.map({"all": 0, "month": 1, "week": 2})
    x = x.sort_values(["wallet", "rank", "w"])
    return x.groupby("wallet").category.first()


def main():
    m = pd.read_csv(os.path.join(D, "wallet_metrics.csv"))
    lb = pd.read_csv(os.path.join(D, "leaderboards.csv"))
    m["primary_cat"] = m.wallet.map(primary_cat(lb)).fillna("overall")
    m["roi_usdc"] = m.economic_pnl / m.volume_usdc.replace(0, np.nan)

    f = m[(m.series_total > 5000) & (m.span_days >= 60) & (m.active_days >= 40)
          & (m.pnl_90d > 0) & (m.pnl_30d > -0.02 * m.series_total)
          & (m.lump_bigwin < 0.3) & (m.lump_1d < 0.25)].copy()
    comps = [f.sharpe.clip(upper=15).rank(pct=True),
             f.sharpe_90d.clip(upper=15).rank(pct=True),
             f.pct_pos_weeks.rank(pct=True),
             (-f.lump_top5d).rank(pct=True),
             (-f.lump_bigwin).rank(pct=True),
             np.log(f.active_days).rank(pct=True)]
    f["consistency"] = sum(comps) / len(comps)
    f["small_cap_ok"] = (f.avg_trade_usdc < 400) & (f.volume_usdc < 6e7)
    f = f.sort_values("consistency", ascending=False)
    f["cons_rank"] = np.arange(1, len(f) + 1)
    f.to_csv(os.path.join(D, "ranked_consistent.csv"), index=False)
    print(len(m), "screened ->", len(f), "pass consistency filters;",
          int(f.small_cap_ok.sum()), "small-cap-ok")

    # deep-dive: top-N per primary category among small-cap-ok, plus top overall
    sc = f[f.small_cap_ok]
    per = {"weather": 12, "mentions": 7, "culture": 5, "tech": 4, "finance": 3,
           "economics": 2, "crypto": 7, "sports": 5, "politics": 3, "overall": 2}
    pick = []
    for cat, n in per.items():
        pick += list(sc[sc.primary_cat == cat].wallet.head(n))
    pick += list(sc.wallet.head(25))
    # reference names: famous category leaders even if they fail filters
    ref_names = ["gopfan2", "ColdMath", "Platykurtic", "Bilberry", "HondaCivic", "jjavi",
                 "opopv.", "semi", "Dropper"]
    pick += list(m[m.name.isin(ref_names)].wallet)
    pick = list(dict.fromkeys(pick))
    dd = m[m.wallet.isin(pick)].copy()
    dd = dd.merge(f[["wallet", "consistency", "cons_rank"]], on="wallet", how="left")
    dd = dd.sort_values("consistency", ascending=False)
    dd.to_csv(os.path.join(D, "deep_dive.csv"), index=False)
    print(len(dd), "deep-dive wallets")
    print(dd[["name", "primary_cat", "cons_rank", "series_total", "sharpe", "avg_trade_usdc"]]
          .round(2).to_string())


if __name__ == "__main__":
    main()
