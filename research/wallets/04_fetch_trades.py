"""Step 4a: fetch recent fills + activity for deep-dive wallets, and gamma metadata
for a sample of their markets.

NOTE: data-api /trades defaults to takerOnly=true. We fetch both
  - all fills   (takerOnly=false)  -> what the wallet actually did
  - taker fills (takerOnly=true)   -> used to label each fill taker vs maker

Outputs (data/wallets/):
  trades_all.csv, trades_taker.csv, activity.csv, markets_meta.csv
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import polylib as pl  # noqa: E402

D = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wallets")
CACHE = os.path.join(D, "cache")
MAXN = 3000
KEEP = ["proxyWallet", "timestamp", "conditionId", "type", "side", "price", "size",
        "usdcSize", "outcome", "outcomeIndex", "title", "slug", "eventSlug",
        "transactionHash", "asset"]


def cached(name, fn):
    path = os.path.join(CACHE, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    v = fn()
    with open(path, "w") as f:
        json.dump(v, f)
    return v


def paged(url, params, maxn=MAXN):
    out = []
    for off in range(0, maxn, 500):
        try:
            b = pl.get(url, dict(params, limit=500, offset=off))
        except Exception as e:  # noqa: BLE001
            print("ERR", url, params.get("user"), off, e)
            break
        out += [{k: r.get(k) for k in KEEP} for r in b]
        if len(b) < 500:
            break
        time.sleep(0.1)
    return out


def fetch_wallet(w):
    a = cached(f"tr_all_{w}.json", lambda: paged(f"{pl.DATA}/trades", {"user": w, "takerOnly": "false"}))
    t = cached(f"tr_tk_{w}.json", lambda: paged(f"{pl.DATA}/trades", {"user": w, "takerOnly": "true"}))
    act = cached(f"act_{w}.json", lambda: paged(f"{pl.DATA}/activity", {"user": w}, maxn=2000))
    return w, a, t, act


def fetch_meta(cids, closed=None):
    fields = ["conditionId", "slug", "question", "endDate", "closedTime", "startDate",
              "negRisk", "feesEnabled", "orderMinSize", "rewardsMinSize", "rewardsMaxSpread",
              "volumeNum", "liquidityNum", "gameStartTime", "createdAt", "outcomes",
              "outcomePrices", "umaResolutionStatus", "closed", "sportsMarketType"]
    out = []
    for i in range(0, len(cids), 40):
        chunk = cids[i:i + 40]
        try:
            q = [("condition_ids", c) for c in chunk] + [("limit", 100)]
            if closed is not None:
                q.append(("closed", closed))
            b = pl.get(f"{pl.GAMMA}/markets", q)
        except Exception as e:  # noqa: BLE001
            print("meta ERR", e)
            continue
        for m in b:
            r = {k: m.get(k) for k in fields}
            ev = (m.get("events") or [{}])[0]
            r["event_slug"] = ev.get("slug")
            r["event_endDate"] = ev.get("endDate")
            r["eventDate"] = ev.get("eventDate")
            r["series"] = ((ev.get("series") or [{}])[0]).get("slug")
            out.append(r)
        time.sleep(0.1)
    return out


def main():
    dd = pd.read_csv(os.path.join(D, "deep_dive.csv"))
    wallets = list(dd.wallet)
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(fetch_wallet, wallets))
    A, T, ACT = [], [], []
    for w, a, t, act in res:
        for r in a:
            r["wallet"] = w
        for r in t:
            r["wallet"] = w
        for r in act:
            r["wallet"] = w
        A += a
        T += t
        ACT += act
        print(w[:10], len(a), len(t), len(act))
    A, T, ACT = pd.DataFrame(A), pd.DataFrame(T), pd.DataFrame(ACT)
    A.to_csv(os.path.join(D, "trades_all.csv"), index=False)
    T.to_csv(os.path.join(D, "trades_taker.csv"), index=False)
    ACT.to_csv(os.path.join(D, "activity.csv"), index=False)

    # sample markets: per wallet the top-80 by USDC plus 80 random others
    A["usdc"] = A.price * A.size
    cids = set()
    for w, g in A.groupby("wallet"):
        agg = g.groupby("conditionId").usdc.sum().sort_values(ascending=False)
        cids |= set(agg.index[:80])
        rest = agg.index[80:]
        if len(rest):
            cids |= set(pd.Series(rest).sample(min(80, len(rest)), random_state=0))
    cids = sorted(cids)
    print(len(cids), "markets to look up")
    # gamma /markets excludes closed markets unless closed=true -> query both
    meta = cached("markets_meta_v2.json", lambda: fetch_meta(cids))
    have = {m["conditionId"] for m in meta}
    miss = [c for c in cids if c not in have]
    meta += cached("markets_meta_closed_v2.json", lambda: fetch_meta(miss, closed="true"))
    pd.DataFrame(meta).drop_duplicates("conditionId").to_csv(
        os.path.join(D, "markets_meta.csv"), index=False)
    print(len(meta), "market meta rows")
    # open-position value snapshot (capital-in-use proxy)
    vals = []
    for w in wallets:
        v = cached(f"value_{w}.json", lambda: pl.get(f"{pl.DATA}/value", {"user": w}))
        vals.append({"wallet": w, "open_value": (v[0].get("value") if v else None)})
    pd.DataFrame(vals).to_csv(os.path.join(D, "open_value.csv"), index=False)


if __name__ == "__main__":
    main()
