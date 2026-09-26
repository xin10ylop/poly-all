"""Step 1: pull Polymarket leaderboards (PNL-ordered) for period x category.

Output: data/wallets/leaderboards.csv  (one row per period/category/rank)
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import polylib as pl  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "wallets")
os.makedirs(OUT, exist_ok=True)

PERIODS = ["week", "month", "all"]
CATS = ["overall", "politics", "sports", "crypto", "weather", "culture",
        "mentions", "economics", "tech", "finance"]
TOP = 200  # per combo


def pull(period, cat):
    rows = []
    for off in range(0, TOP, 50):
        p = {"timePeriod": period, "orderBy": "PNL", "limit": 50, "offset": off}
        if cat != "overall":
            p["category"] = cat
        try:
            d = pl.get(f"{pl.DATA}/v1/leaderboard", p)
        except Exception as e:  # noqa: BLE001
            print("ERR", period, cat, off, e)
            break
        if not d:
            break
        for r in d:
            rows.append({"period": period, "category": cat, "rank": int(r["rank"]),
                         "wallet": r["proxyWallet"].lower(), "name": r.get("userName"),
                         "vol": r.get("vol"), "pnl": r.get("pnl")})
        if len(d) < 50:
            break
        time.sleep(0.1)
    return rows


def main():
    combos = [(p, c) for p in PERIODS for c in CATS]
    with ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda pc: pull(*pc), combos))
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(os.path.join(OUT, "leaderboards.csv"), index=False)
    print(df.groupby(["period", "category"]).size().unstack())
    print(len(df), "rows,", df.wallet.nunique(), "unique wallets")


if __name__ == "__main__":
    main()
