"""Emit markdown tables used in REPORT.md (printed to stdout and saved to data/wallets/tables.md)."""
import os

import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wallets")
R = pd.read_csv(os.path.join(D, "ranked_consistent.csv"))
P = pd.read_csv(os.path.join(D, "wallet_profiles.csv"))
C = pd.read_csv(os.path.join(D, "capital.csv"))


def k(x):
    if pd.isna(x):
        return "–"
    return f"{x / 1e6:.2f}M" if abs(x) >= 1e6 else f"{x / 1e3:.0f}k" if abs(x) >= 1e4 else f"{x / 1e3:.1f}k"


def short(n, w):
    n = str(n)
    return (w[:10] + "…") if n.startswith("0x") else n[:20]


out = []
out.append("### A. Top 40 by consistency score (all 1,170 screened wallets; filters in Method)\n")
out.append("| # | name | wallet | lb cat | PnL | 30d | 90d | Sharpe | Sh 90d | +wk | big-win/PnL | active d | avg fill $ | vol $ | maker reb | rewards |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for _, r in R.head(40).iterrows():
    out.append(f"| {r.cons_rank} | {short(r['name'], r.wallet)} | `{r.wallet}` | {r.primary_cat} | {k(r.series_total)} | "
               f"{k(r.pnl_30d)} | {k(r.pnl_90d)} | {r.sharpe:.1f} | {r.sharpe_90d:.1f} | {r.pct_pos_weeks:.0%} | "
               f"{r.lump_bigwin:.2f} | {r.active_days:.0f} | {r.avg_trade_usdc:.0f} | {k(r.volume_usdc)} | "
               f"{k(r.maker_rebate)} | {k(r.reward_income)} |")

out.append("\n### B. Deep-dive fill statistics (last ≤3,000 fills per wallet)\n")
out.append("| name | cons # | main market type | taker % | BUY$ at >0.9 | median fill $ | fills/day | hold-to-res ret/$ | "
           "mkts won % | ret on turnover | capital p90 $ | open pos $ |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
P = P.merge(C[["wallet", "cap_p90", "win_rate", "ret_on_turnover", "n_mkts"]], on="wallet", how="left")
for _, r in P.sort_values("cons_rank").iterrows():
    hi = r["buy_0.90-0.97"] + r["buy_>0.97"]
    out.append(f"| {short(r['name'], r.wallet)} | {int(r.cons_rank) if pd.notna(r.cons_rank) else '–'} | "
               f"{r.mix.split(',')[0] if isinstance(r.mix, str) else ''} | {r.taker_share_usdc:.0%} | {hi:.0%} | "
               f"{r.med_fill_usdc:.1f} | {r.fills_per_day:,.0f} | "
               f"{(f'{r.hold_ret:+.3f}' if pd.notna(r.hold_ret) else '–')} | "
               f"{(f'{r.win_rate:.0%} (n={int(r.n_mkts)})' if pd.notna(r.win_rate) else '–')} | "
               f"{(f'{r.ret_on_turnover:+.1%}' if pd.notna(r.ret_on_turnover) else '–')} | {k(r.cap_p90)} | "
               f"{k(r.open_value)} |")
txt = "\n".join(out)
with open(os.path.join(D, "tables.md"), "w") as f:
    f.write(txt)
print(txt)
