"""Thin helpers around Polymarket public APIs (gamma, CLOB, data-api, lb-api)."""
import json
import os
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
LB = "https://lb-api.polymarket.com"

_s = requests.Session()
_s.mount("https://", HTTPAdapter(pool_maxsize=64, max_retries=Retry(
    total=5, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"])))
_s.headers["User-Agent"] = "poly-research/0.1"


def get(url, params=None, timeout=30):
    r = _s.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def post(url, body, headers=None, timeout=60):
    r = _s.post(url, json=body, headers=headers or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def jl(x):
    """gamma returns JSON-encoded lists as strings."""
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return x
    return x


def gamma_paginate(path, params, page=500, max_items=None, sleep=0.0):
    out, offset = [], 0
    while True:
        p = dict(params, limit=page, offset=offset)
        batch = get(f"{GAMMA}/{path}", p)
        if not batch:
            break
        out.extend(batch)
        offset += len(batch)
        if max_items and len(out) >= max_items:
            break
        if len(batch) < page:
            break
        time.sleep(sleep)
    return out


def prices_history(token_id, interval="max", fidelity=60, start_ts=None, end_ts=None):
    p = {"market": token_id, "fidelity": fidelity}
    if start_ts is not None:
        p["startTs"] = int(start_ts)
        p["endTs"] = int(end_ts)
    else:
        p["interval"] = interval
    return get(f"{CLOB}/prices-history", p).get("history", [])


def book(token_id):
    return get(f"{CLOB}/book", {"token_id": token_id})


def books(token_ids):
    return post(f"{CLOB}/books", [{"token_id": t} for t in token_ids])


def trades(market=None, user=None, limit=500, offset=0, taker_only=None, **kw):
    p = {"limit": limit, "offset": offset}
    if market:
        p["market"] = market
    if user:
        p["user"] = user
    if taker_only is not None:
        p["takerOnly"] = str(taker_only).lower()
    p.update(kw)
    return get(f"{DATA}/trades", p)


def load_env():
    path = os.path.join(os.path.dirname(__file__), "..", ".env.sh")
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                k, v = line[7:].split("=", 1)
                os.environ.setdefault(k, v)


def gamma_keyset(path, params, limit=100, max_items=None, sleep=0.0, key=None):
    """Keyset pagination for /markets/keyset and /events/keyset."""
    key = key or path.split("/")[0]
    out, cursor = [], None
    while True:
        p = dict(params, limit=limit)
        if cursor:
            p["after_cursor"] = cursor
        d = get(f"{GAMMA}/{path}", p)
        batch = d.get(key, [])
        out.extend(batch)
        cursor = d.get("next_cursor")
        if not batch or not cursor or (max_items and len(out) >= max_items):
            break
        time.sleep(sleep)
    return out
