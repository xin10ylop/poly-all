"""Step 4b: characterize the strategy of each deep-dive wallet from its recent fills.

Inputs : data/wallets/{trades_all,trades_taker,activity,markets_meta,open_value,deep_dive}.csv
Outputs: data/wallets/wallet_profiles.csv        (one row per wallet)
         data/wallets/wallet_bucket_edge.csv     (hold-to-resolution edge by entry-price bucket)
         data/wallets/profiles.txt               (human-readable dump)
"""
import json
import os
import re

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wallets")

SPORT_RE = re.compile(
    r"^(mlb|nba|nfl|nhl|wnba|cfb|cbb|ncaa[a-z]*|epl|lal|sea|bun|fl1|ucl|uel|uecl|mls|atp|wta|"
    r"ufc|cs2|dota2|lol|val|fifwc|ptc|bra|arg|mex|jpn|kor|chn|aus|ned|por|tur|bel|sco|"
    r"itsb|efl|elc|lpga|pga|f1|nascar|ipl|cric[a-z]*|kbo|npb|afl|nrl|rugby|boxing|mma|"
    r"tennis|golf|euroleague|cl|el|wc|cwc|copa|conmebol|concacaf|lck|lpl|r6|ow)\b")
SPORT_RE2 = re.compile(r"^[a-z0-9]{2,8}-[a-z0-9]{2,6}-[a-z0-9]{2,6}-\d{4}-\d{2}-\d{2}")
SPORT_KW = ["world-cup", "nba-champion", "super-bowl", "stanley-cup", "world-series", "ballon",
            "premier-league", "champions-league", "grand-slam", "open-winner", "masters",
            "wimbledon", "us-open", "tour-de-france", "mvp", "heisman", "playoffs", "vs-",
            "-win-on-", "spread", "-total-", "lpga", "grand-prix"]


def mtype(slug, ev):
    ev = ev if isinstance(ev, str) else ""
    slug = slug if isinstance(slug, str) else ""
    s = (ev or slug).lower()
    s2 = slug.lower()
    if "highest-temperature" in s:
        return "weather_high"
    if "lowest-temperature" in s:
        return "weather_low"
    if any(k in s for k in ["temperature", "precipitation", "rainfall", "snowfall", "hurricane",
                            "-rain-", "tornado", "heat-index"]):
        return "weather_other"
    m = re.search(r"(btc|eth|sol|xrp|doge|bnb|hype)-updown-(\d+)m", s)
    if m:
        return f"crypto_updown_{m.group(2)}m"
    if "up-or-down" in s and any(k in s for k in ["bitcoin", "ethereum", "solana", "xrp", "doge"]):
        return "crypto_updown_1h+"
    if re.search(r"(-say-|what-will-.*-say|mention|during-.*(call|briefing|speech|interview)|"
                 r"-said-|say-.*-during)", s2) or ("what-will-" in s and "-say" in s):
        return "mentions"
    if SPORT_RE.match(s) or SPORT_RE2.match(s) or any(k in s for k in SPORT_KW):
        return "sports"
    if any(k in s for k in ["bitcoin", "ethereum", "solana", "btc", "eth-", "xrp", "crypto",
                            "microstrategy", "doge", "what-price-will", "memecoin", "pump-fun",
                            "fdv", "token", "airdrop", "hyperliquid"]):
        return "crypto_price"
    if any(k in s for k in ["-ai-", "ai-model", "openai", "grok", "gemini", "claude", "anthropic",
                            "gpt", "llm", "free-app", "paid-app", "app-store", "gpu", "deepseek",
                            "which-company-has", "text-arena", "lmarena", "apple", "nvidia",
                            "google", "tesla", "spacex", "starship", "tech"]):
        return "tech_ai"
    if any(k in s for k in ["mrbeast", "views", "spotify", "song", "billboard", "box-office",
                            "rotten", "tweets", "elon-musk-of", "kai-and-speed", "kanye", "album",
                            "netflix", "youtube", "tiktok", "grammy", "oscar", "emmy", "movie",
                            "artist", "taylor-swift", "celebrity", "wordle", "streamer", "time-person",
                            "person-of-the-year", "eurovision", "animals", "mobs", "twitch",
                            "kick", "followers", "subscribers", "pope", "bachelor", "survivor"]):
        return "culture"
    if any(k in s for k in ["fed-", "fed-decision", "cpi", "inflation", "gdp", "recession",
                            "unemployment", "jobs", "interest-rate", "rate-cut", "sp-500", "spx",
                            "s-p-500", "nasdaq", "dow", "close-above", "earnings", "stock", "ipo",
                            "gamestop", "tariff", "treasury", "gold", "oil", "xau", "xag", "silver",
                            "valuation", "market-cap", "largest-company", "revenue", "powell",
                            "yield", "fomc", "payroll"]):
        return "finance_econ"
    if any(k in s for k in ["election", "president", "trump", "iran", "russia", "ukraine",
                            "israel", "china", "prime-minister", "governor", "senate", "house",
                            "parliament", "ceasefire", "xi-jinping", "zelensk", "putin", "party",
                            "mayor", "nominee", "primary", "cabinet", "minister", "vance",
                            "musk", "war", "nato", "gaza", "strike", "invade", "capture",
                            "leader", "venezuela", "maduro", "kharg", "hormuz", "tariffs",
                            "impeach", "pardon", "resign", "out-as", "deport", "attend",
                            "meet", "visit", "sanction", "deal", "shutdown", "supreme-court"]):
        return "politics_geo"
    return "other"


BUCKETS = [0, 0.1, 0.5, 0.9, 0.97, 1.0001]
BLABELS = ["<0.10", "0.10-0.50", "0.50-0.90", "0.90-0.97", ">0.97"]


def pct(x):
    return f"{100 * x:.0f}%" if pd.notna(x) else "na"


def main():
    A = pd.read_csv(os.path.join(D, "trades_all.csv"))
    T = pd.read_csv(os.path.join(D, "trades_taker.csv"))
    ACT = pd.read_csv(os.path.join(D, "activity.csv"))
    M = pd.read_csv(os.path.join(D, "markets_meta.csv"))
    OV = pd.read_csv(os.path.join(D, "open_value.csv")).set_index("wallet").open_value
    DD = pd.read_csv(os.path.join(D, "deep_dive.csv")).set_index("wallet")

    A = A.drop_duplicates(["wallet", "transactionHash", "asset", "side", "price", "size", "timestamp"])
    A["usdc"] = A.price * A["size"]
    A["mtype"] = [mtype(s, e) for s, e in zip(A.slug, A.eventSlug)]
    A["ts"] = pd.to_datetime(A.timestamp, unit="s", utc=True)

    # --- taker/maker label: fill's tx hash appears in the wallet's taker-only list
    tk = set(zip(T.wallet, T.transactionHash))
    A["is_taker"] = [(w, h) in tk for w, h in zip(A.wallet, A.transactionHash)]
    tmin = T.groupby("wallet").timestamp.min()
    A["in_tk_window"] = A.timestamp >= A.wallet.map(tmin).fillna(0)

    # --- market metadata join
    M["closed_ts"] = pd.to_datetime(M.closedTime, utc=True, errors="coerce", format="mixed")
    M["end_ts"] = pd.to_datetime(M.endDate, utc=True, errors="coerce", format="mixed")
    M["game_ts"] = pd.to_datetime(M.gameStartTime, utc=True, errors="coerce", format="mixed")

    def winner(r):
        try:
            outs, prices = json.loads(r.outcomes), [float(x) for x in json.loads(r.outcomePrices)]
        except Exception:  # noqa: BLE001
            return None
        if r.closed is not True and str(r.closed).lower() != "true":
            return None
        if max(prices) < 0.99:
            return None
        return outs[int(np.argmax(prices))]

    M["winner"] = M.apply(winner, axis=1)
    A = A.merge(M[["conditionId", "closed_ts", "end_ts", "game_ts", "winner", "negRisk",
                   "feesEnabled", "rewardsMaxSpread"]], on="conditionId", how="left")
    A["h_to_close"] = (A.closed_ts - A.ts).dt.total_seconds() / 3600
    A["h_to_end"] = (A.end_ts - A.ts).dt.total_seconds() / 3600
    A["h_from_game"] = (A.ts - A.game_ts).dt.total_seconds() / 3600  # >0 => after start
    A["won"] = np.where(A.winner.isna(), np.nan, (A.outcome == A.winner).astype(float))
    A["bucket"] = pd.cut(A.price, BUCKETS, labels=BLABELS, right=False)

    # activity types
    act_counts = ACT.groupby(["wallet", "type"]).size().unstack(fill_value=0)
    act_span = ACT.groupby("wallet").timestamp.agg(lambda x: (x.max() - x.min()) / 86400)

    rows, edge_rows, lines = [], [], []
    for w, g in A.groupby("wallet"):
        info = DD.loc[w]
        r = {"wallet": w, "name": info["name"], "cons_rank": info.get("cons_rank"),
             "primary_cat": info.get("primary_cat")}
        span_d = max((g.timestamp.max() - g.timestamp.min()) / 86400, 1 / 24)
        r["window_days"] = span_d
        r["n_fills"] = len(g)
        r["fills_per_day"] = len(g) / span_d
        r["usdc_per_day"] = g.usdc.sum() / span_d
        r["n_markets"] = g.conditionId.nunique()
        r["markets_per_day"] = r["n_markets"] / span_d
        r["fills_per_market"] = len(g) / r["n_markets"]
        mix = g.groupby("mtype").usdc.sum().sort_values(ascending=False) / g.usdc.sum()
        r["mix"] = ", ".join(f"{k} {v:.0%}" for k, v in mix.head(4).items() if v >= 0.03)
        r["top_type"] = mix.index[0]
        buys = g[g.side == "BUY"]
        r["buy_share_usdc"] = buys.usdc.sum() / g.usdc.sum()
        r["buy_share_n"] = len(buys) / len(g)
        bq = buys.groupby("bucket", observed=False).usdc.sum() / max(buys.usdc.sum(), 1e-9)
        for b in BLABELS:
            r[f"buy_{b}"] = bq.get(b, 0)
        r["buy_vwap"] = (buys.price * buys.usdc).sum() / max(buys.usdc.sum(), 1e-9)
        r["buy_med_price"] = buys.price.median()
        yn = buys[buys.outcome.isin(["Yes", "No"])]
        r["buy_no_share"] = (yn.outcome == "No").mean() if len(yn) else np.nan
        r["med_fill_usdc"] = g.usdc.median()
        r["p90_fill_usdc"] = g.usdc.quantile(0.9)
        gw = g[g.in_tk_window]
        r["taker_share_usdc"] = gw[gw.is_taker].usdc.sum() / max(gw.usdc.sum(), 1e-9)
        r["taker_share_buy"] = (gw[(gw.side == "BUY") & gw.is_taker].usdc.sum()
                                / max(gw[gw.side == "BUY"].usdc.sum(), 1e-9))
        # both outcomes bought in same market
        per_m = buys.groupby("conditionId").outcome.nunique()
        r["both_sides_mkts"] = (per_m >= 2).mean() if len(per_m) else np.nan
        # sold later (exit before resolution) vs hold
        sold_assets = set(g[g.side == "SELL"].asset)
        ba = buys.groupby("asset").usdc.sum()
        r["bought_then_sold_share"] = ba[ba.index.isin(sold_assets)].sum() / max(ba.sum(), 1e-9)
        # hold time: first buy -> first sell after it, per asset
        fb = buys.groupby("asset").timestamp.min()
        sl = g[g.side == "SELL"].groupby("asset").timestamp.max()
        both = fb.index.intersection(sl.index)
        ht = ((sl[both] - fb[both]) / 3600)
        r["med_hold_h_sold"] = ht[ht > 0].median() if len(ht) else np.nan
        # timing vs close
        bt = buys.dropna(subset=["h_to_close"])
        r["meta_cov"] = len(bt) / max(len(buys), 1)
        r["buy_med_h_to_close"] = bt.h_to_close.median() if len(bt) else np.nan
        r["buy_share_last6h"] = ((bt.h_to_close < 6) * bt.usdc).sum() / max(bt.usdc.sum(), 1e-9) if len(bt) else np.nan
        r["buy_share_last24h"] = ((bt.h_to_close < 24) * bt.usdc).sum() / max(bt.usdc.sum(), 1e-9) if len(bt) else np.nan
        # weather: relative to local observation day (gameStartTime = local 00:00)
        wb = buys[buys.mtype.str.startswith("weather") & buys.h_from_game.notna()]
        if len(wb) > 20:
            hfg = wb.h_from_game
            r["wx_med_h_from_localday_start"] = hfg.median()
            r["wx_share_before_day"] = ((hfg < 0) * wb.usdc).sum() / wb.usdc.sum()
            r["wx_share_during_day"] = (((hfg >= 0) & (hfg < 24)) * wb.usdc).sum() / wb.usdc.sum()
            r["wx_share_after_day"] = ((hfg >= 24) * wb.usdc).sum() / wb.usdc.sum()
            r["wx_share_local_afternoon"] = (((hfg >= 12) & (hfg < 24)) * wb.usdc).sum() / wb.usdc.sum()
        sb = buys[(buys.mtype == "sports") & buys.h_from_game.notna()]
        if len(sb) > 20:
            r["sp_share_inplay"] = ((sb.h_from_game > 0) * sb.usdc).sum() / sb.usdc.sum()
            r["sp_med_h_from_start"] = sb.h_from_game.median()
        # hold-to-resolution edge of buys (selection skill), overall and by bucket
        rb = buys.dropna(subset=["won"])
        r["res_cov"] = len(rb) / max(len(buys), 1)
        if len(rb) >= 20:
            r["hit_rate"] = (rb.won * rb.usdc).sum() / rb.usdc.sum()
            r["avg_price_res"] = (rb.price * rb.usdc).sum() / rb.usdc.sum()
            # return per $ if held: sum(size*won - usdc)/sum(usdc)
            r["hold_ret"] = ((rb["size"] * rb.won).sum() - rb.usdc.sum()) / rb.usdc.sum()
            for b, gb in rb.groupby("bucket", observed=True):
                if len(gb) >= 5:
                    edge_rows.append({"wallet": w, "name": info["name"], "bucket": b, "n": len(gb),
                                      "usdc": gb.usdc.sum(),
                                      "avg_price": (gb.price * gb.usdc).sum() / gb.usdc.sum(),
                                      "hit_rate_usdc": (gb.won * gb.usdc).sum() / gb.usdc.sum(),
                                      "hit_rate_n": gb.won.mean(),
                                      "hold_ret": ((gb["size"] * gb.won).sum() - gb.usdc.sum()) / gb.usdc.sum()})
        # activity
        ac = act_counts.loc[w] if w in act_counts.index else pd.Series(dtype=float)
        asp = max(act_span.get(w, 1), 1 / 24)
        for t in ["TRADE", "REDEEM", "MERGE", "SPLIT", "REWARD", "CONVERSION", "MAKER_REBATE"]:
            r[f"act_{t.lower()}"] = int(ac.get(t, 0))
        r["act_span_days"] = asp
        # capital proxies
        r["open_value"] = OV.get(w)
        daily_buy = buys.set_index("ts").usdc.resample("1D").sum()
        daily_buy = daily_buy[daily_buy > 0]
        r["med_daily_buy_usdc"] = daily_buy.median() if len(daily_buy) else np.nan
        for k in ["series_total", "pnl_30d", "pnl_90d", "sharpe", "sharpe_90d", "pct_pos_days",
                  "pct_pos_weeks", "lump_1d", "lump_bigwin", "max_dd", "active_days", "span_days",
                  "n_markets", "trade_count", "volume_usdc", "avg_trade_usdc", "fees_paid",
                  "maker_rebate", "reward_income", "taker_rebate", "economic_pnl"]:
            r[f"s_{k}"] = info.get(k)
        rows.append(r)

    P = pd.DataFrame(rows).sort_values("cons_rank")
    P.to_csv(os.path.join(D, "wallet_profiles.csv"), index=False)
    E = pd.DataFrame(edge_rows)
    E.to_csv(os.path.join(D, "wallet_bucket_edge.csv"), index=False)

    for _, r in P.iterrows():
        lines.append(f"=== #{r.cons_rank} {r['name']} {r.wallet} [{r.primary_cat}]")
        lines.append(f"  PnL {r.s_series_total:,.0f} | 30d {r.s_pnl_30d:,.0f} | 90d {r.s_pnl_90d:,.0f} | "
                     f"Sharpe {r.s_sharpe:.1f} (90d {r.s_sharpe_90d:.1f}) | +days {pct(r.s_pct_pos_days)} "
                     f"+wks {pct(r.s_pct_pos_weeks)} | lump1d {r.s_lump_1d:.2f} bigwin {r.s_lump_bigwin:.2f} | "
                     f"active {r.s_active_days:.0f}/{r.s_span_days:.0f}d")
        lines.append(f"  all-time: mkts {r.s_n_markets:,.0f} fills {r.s_trade_count:,.0f} vol$ {r.s_volume_usdc:,.0f} "
                     f"avgfill$ {r.s_avg_trade_usdc:.1f} fees {r.s_fees_paid:,.0f} makerReb {r.s_maker_rebate:,.0f} "
                     f"rewards {r.s_reward_income:,.0f} open$ {r.open_value if pd.notna(r.open_value) else float('nan'):,.0f}")
        lines.append(f"  window {r.window_days:.1f}d: {r.n_fills} fills ({r.fills_per_day:,.0f}/d), {r.n_markets} mkts "
                     f"({r.markets_per_day:.1f}/d), ${r.usdc_per_day:,.0f}/d, med fill ${r.med_fill_usdc:.1f} p90 ${r.p90_fill_usdc:.0f}")
        lines.append(f"  mix: {r.mix}")
        lines.append(f"  BUY {pct(r.buy_share_usdc)} of $ | buy $ by price: " +
                     " ".join(f"{b}:{pct(r['buy_' + b])}" for b in BLABELS) +
                     f" | vwap {r.buy_vwap:.2f} | NO-share {pct(r.buy_no_share)}")
        lines.append(f"  taker share {pct(r.taker_share_usdc)} (buys {pct(r.taker_share_buy)}) | both-sides mkts "
                     f"{pct(r.both_sides_mkts)} | bought-then-sold {pct(r.bought_then_sold_share)} "
                     f"med hold(sold) {r.med_hold_h_sold:.1f}h")
        lines.append(f"  timing: med h to close {r.buy_med_h_to_close:.1f} | last6h {pct(r.buy_share_last6h)} "
                     f"last24h {pct(r.buy_share_last24h)} (meta cov {pct(r.meta_cov)})")
        if pd.notna(r.get("wx_med_h_from_localday_start")):
            lines.append(f"  weather: med h from local-day start {r.wx_med_h_from_localday_start:.1f} | before day "
                         f"{pct(r.wx_share_before_day)} during {pct(r.wx_share_during_day)} (afternoon "
                         f"{pct(r.wx_share_local_afternoon)}) after {pct(r.wx_share_after_day)}")
        if pd.notna(r.get("sp_share_inplay")):
            lines.append(f"  sports: in-play share {pct(r.sp_share_inplay)} | med h from start {r.sp_med_h_from_start:.1f}")
        if pd.notna(r.get("hit_rate")):
            lines.append(f"  hold-to-res (cov {pct(r.res_cov)}): hit {pct(r.hit_rate)} at avg price "
                         f"{r.avg_price_res:.2f} -> ret/$ {r.hold_ret:+.3f}")
            e = E[E.wallet == r.wallet]
            lines.append("    " + " | ".join(f"{x.bucket}: n{x.n} p{x.avg_price:.2f} hit{x.hit_rate_usdc:.2f} "
                                              f"ret{x.hold_ret:+.2f}" for x in e.itertuples()))
        lines.append(f"  activity(last {r.act_span_days:.1f}d): trade {r.act_trade} redeem {r.act_redeem} merge "
                     f"{r.act_merge} split {r.act_split} reward {r.act_reward} conv {r.act_conversion} "
                     f"| med daily buy ${r.med_daily_buy_usdc:,.0f}")
    with open(os.path.join(D, "profiles.txt"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print("\nmtype 'other' share of $:", A[A.mtype == "other"].usdc.sum() / A.usdc.sum())


if __name__ == "__main__":
    main()
