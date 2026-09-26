"""Step 4c: targeted pattern checks on the deep-dive fills.

 1. Weather: minute-of-hour of high-price NO buys (METAR cadence?), buy->sell round trips,
    local-hour distribution per wallet.
 2. Crypto up/down: seconds into the 5m/15m window at which each wallet buys, and price.
 3. Top event families per wallet (what exactly they trade).
 4. Mentions: timing of YES buys vs market close for in-event snipers.
Output: data/wallets/patterns.txt
"""
import os
import re

import numpy as np
import pandas as pd

D = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wallets")
out = []


def p(*a):
    s = " ".join(str(x) for x in a)
    out.append(s)
    print(s)


A = pd.read_csv(os.path.join(D, "trades_all.csv"))
T = pd.read_csv(os.path.join(D, "trades_taker.csv"))
M = pd.read_csv(os.path.join(D, "markets_meta.csv"))
P = pd.read_csv(os.path.join(D, "wallet_profiles.csv"))
names = P.set_index("wallet")["name"].to_dict()
A = A.drop_duplicates(["wallet", "transactionHash", "asset", "side", "price", "size", "timestamp"])
A["usdc"] = A.price * A["size"]
A["ts"] = pd.to_datetime(A.timestamp, unit="s", utc=True)
tk = set(zip(T.wallet, T.transactionHash))
A["is_taker"] = [(w, h) in tk for w, h in zip(A.wallet, A.transactionHash)]
M["game_ts"] = pd.to_datetime(M.gameStartTime, utc=True, errors="coerce", format="mixed")
M["closed_ts"] = pd.to_datetime(M.closedTime, utc=True, errors="coerce", format="mixed")
A = A.merge(M[["conditionId", "game_ts", "closed_ts"]], on="conditionId", how="left")

# ---------------------------------------------------------------- 1. weather
p("=" * 100)
p("1. WEATHER: high-price (>=0.9) BUY fills — minute-of-hour (UTC) histogram, local hour, round trips")
wx = A[A.slug.fillna("").str.contains("temperature")]
for w, g in wx.groupby("wallet"):
    if len(g) < 200:
        continue
    b = g[(g.side == "BUY")]
    hi = b[b.price >= 0.9]
    tkb = hi[hi.is_taker]
    mins = tkb.ts.dt.minute
    bins = pd.cut(mins, [0, 5, 15, 19, 26, 45, 49, 56, 60], right=False,
                  labels=["00-04", "05-14", "15-18", "19-25", "26-44", "45-48", "49-55", "56-59"])
    hist = (bins.value_counts(normalize=True).reindex(bins.cat.categories).fillna(0) * 100).round(0)
    loc_h = ((b.ts - b.game_ts).dt.total_seconds() / 3600).dropna()
    lh = pd.cut(loc_h, [-48, 0, 6, 10, 13, 16, 20, 24, 72], right=False).value_counts(normalize=True).sort_index()
    # round trips: buy then sell same asset
    sells = g[g.side == "SELL"].groupby("asset").agg(sell_t=("timestamp", "min"), sell_p=("price", "mean"))
    fb = b.groupby("asset").agg(buy_t=("timestamp", "min"), buy_p=("price", "mean"), usdc=("usdc", "sum"))
    rt = fb.join(sells, how="inner")
    rt = rt[rt.sell_t > rt.buy_t]
    p(f"-- {names.get(w, w)[:28]:28s} buys>=0.9: {len(hi):5d} ({len(hi) / max(len(b), 1):.0%} of buy fills), taker {len(tkb)}")
    if len(tkb) >= 30:
        p("   taker minute-of-hour %: " + " ".join(f"{k}:{v:.0f}" for k, v in hist.items()))
    p("   buy fills by local hour-of-observation-day: " +
      " ".join(f"{str(k)}:{v:.0%}" for k, v in lh.items() if v > 0.005))
    if len(rt) >= 10:
        p(f"   round trips: {len(rt)} assets, med hold {((rt.sell_t - rt.buy_t) / 60).median():.0f} min, "
          f"med buy {rt.buy_p.median():.3f} -> med sell {rt.sell_p.median():.3f}")

# ---------------------------------------------------------------- 2. crypto up/down
p("=" * 100)
p("2. CRYPTO UP/DOWN: seconds into window at BUY (window start parsed from slug)")
cu = A[A.slug.fillna("").str.contains(r"-updown-(5|15)m-\d+", regex=True)].copy()
ext = cu.slug.str.extract(r"-updown-(\d+)m-(\d+)")
cu["win_m"] = ext[0].astype(float)
cu["win_start"] = ext[1].astype(float)
cu["sec_in"] = cu.timestamp - cu.win_start
cu["frac_in"] = cu.sec_in / (cu.win_m * 60)
for w, g in cu.groupby("wallet"):
    if len(g) < 300:
        continue
    b = g[g.side == "BUY"]
    for wm, gb in b.groupby("win_m"):
        if len(gb) < 100:
            continue
        fr = pd.cut(gb.frac_in, [-10, 0, 0.25, 0.5, 0.75, 0.9, 1.0, 10],
                    labels=["pre", "0-25%", "25-50%", "50-75%", "75-90%", "90-100%", "after"])
        dist = fr.value_counts(normalize=True).reindex(fr.cat.categories).fillna(0)
        byb = gb.groupby(fr, observed=True).price.median()
        p(f"-- {names.get(w, w)[:22]:22s} {int(wm):>2}m n={len(gb):5d} taker={gb.is_taker.mean():.0%} "
          f"| share by window pos: " + " ".join(f"{k}:{v:.0%}" for k, v in dist.items() if v > 0.005) +
          " | med price by pos: " + " ".join(f"{k}:{v:.2f}" for k, v in byb.items()))
    # both-sides within same window: sum of avg prices
    per = b.groupby(["conditionId", "outcome"]).apply(
        lambda x: pd.Series({"vw": (x.price * x["size"]).sum() / x["size"].sum(), "sz": x["size"].sum()}))
    per = per.unstack("outcome")
    if ("vw", "Up") in per.columns and ("vw", "Down") in per.columns:
        bs = per.dropna()
        if len(bs):
            s = bs[("vw", "Up")] + bs[("vw", "Down")]
            matched = np.minimum(bs[("sz", "Up")], bs[("sz", "Down")]).sum()
            tot = (bs[("sz", "Up")] + bs[("sz", "Down")]).sum()
            p(f"   both-sides windows: {len(bs)} of {len(per)}; median Up+Down vwap sum {s.median():.3f}; "
              f"share<1: {(s < 1).mean():.0%}; hedged share of size {2 * matched / tot:.0%}")

# ---------------------------------------------------------------- 3. event families
p("=" * 100)
p("3. TOP EVENT FAMILIES per wallet (by USDC, digits/dates stripped)")


def fam(s):
    s = s if isinstance(s, str) else ""
    s = re.sub(r"\d{4}-\d{2}-\d{2}", "", s)
    s = re.sub(r"(january|february|march|april|may|june|july|august|september|october|november|december)", "", s)
    s = re.sub(r"\d+", "#", s)
    return "-".join([t for t in s.split("-") if t][:6])


A["fam"] = A.eventSlug.map(fam)
for w in P.wallet:
    g = A[A.wallet == w]
    f = g.groupby("fam").usdc.sum().sort_values(ascending=False)
    f = f / f.sum()
    p(f"-- {names.get(w, w)[:28]:28s}: " + "; ".join(f"{k} {v:.0%}" for k, v in f.head(6).items()))

# ---------------------------------------------------------------- 4. mentions timing
p("=" * 100)
p("4. MENTIONS: YES/NO buys by hours-before-close and price")
mm = A[A.slug.fillna("").str.contains(r"-say-|mention", regex=True)].copy()
mm["h_to_close"] = (mm.closed_ts - mm.ts).dt.total_seconds() / 3600
for w, g in mm.groupby("wallet"):
    b = g[(g.side == "BUY")].dropna(subset=["h_to_close"])
    if len(b) < 50:
        continue
    for oc, gb in b.groupby("outcome"):
        if len(gb) < 20:
            continue
        hb = pd.cut(gb.h_to_close, [-1, 0.5, 2, 6, 24, 72, 1e5])
        dist = gb.groupby(hb, observed=True).usdc.sum() / gb.usdc.sum()
        pr = gb.groupby(hb, observed=True).price.median()
        p(f"-- {names.get(w, w)[:22]:22s} {oc:3s} n={len(gb):4d} taker={gb.is_taker.mean():.0%}: " +
          " ".join(f"{k}:{v:.0%}@{pr[k]:.2f}" for k, v in dist.items()))

with open(os.path.join(D, "patterns.txt"), "w") as f:
    f.write("\n".join(out))
