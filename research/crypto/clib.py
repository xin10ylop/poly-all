"""Crypto threshold-market research helpers: market parsing, spot data (Binance mirror),
Deribit vol surface, and fair-value models (digital / range / one-touch)."""
import datetime as dt
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import get, post, jl, books, prices_history, trades, gamma_keyset, gamma_paginate  # noqa

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA = os.path.join(ROOT, 'data', 'crypto')
os.makedirs(DATA, exist_ok=True)

BV = 'https://data-api.binance.vision/api/v3'      # Binance market-data mirror (not geo-blocked)
DERIBIT = 'https://www.deribit.com/api/v2/public'
YEAR_S = 365.0 * 86400

UNDER = {'bitcoin': 'BTCUSDT', 'btc': 'BTCUSDT', 'ethereum': 'ETHUSDT', 'eth': 'ETHUSDT',
         'solana': 'SOLUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT'}
# Deribit (currency for summary call, instrument prefix, index name)
DERIBIT_MAP = {'BTCUSDT': ('BTC', 'BTC', 'btc_usd'), 'ETHUSDT': ('ETH', 'ETH', 'eth_usd'),
               'SOLUSDT': ('USDC', 'SOL_USDC', 'sol_usdc'), 'XRPUSDT': ('USDC', 'XRP_USDC', 'xrp_usdc')}
DVOL = {'BTCUSDT': 'BTC', 'ETHUSDT': 'ETH'}

FEE_RATE = 0.07


def taker_fee(p):
    """Polymarket crypto taker fee per share."""
    return FEE_RATE * p * (1 - p)


# ---------------------------------------------------------------- parsing
MONTHS = {m: i for i, m in enumerate(['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                                      'september', 'october', 'november', 'december'], 1)}
SLUG_RE = [
    ('above_hourly', re.compile(r'^(bitcoin|ethereum|solana|xrp)-above-on-([a-z]+)-(\d+)-(\d{4})-(\d+)(am|pm)-et$')),
    ('above_daily', re.compile(r'^(bitcoin|ethereum|solana|xrp)-above-on-([a-z]+)-(\d+)(?:-(\d{4}))?$')),
    ('range_daily', re.compile(r'^(bitcoin|ethereum|solana|xrp)-price-on-([a-z]+)-(\d+)(?:-(\d{4}))?$')),
    ('hit_daily', re.compile(r'^what-price-will-(bitcoin|ethereum|solana|xrp)-hit-on-([a-z]+)-(\d+)(?:-(\d{4}))?$')),
    ('hit_weekly', re.compile(r'^what-price-will-(bitcoin|ethereum|solana|xrp)-hit-([a-z]+)-(\d+)-(\d+)(?:-(\d{4}))?$')),
    ('hit_weekly2', re.compile(r'^what-price-will-(bitcoin|ethereum|solana|xrp)-hit-([a-z]+)-(\d+)-([a-z]+)-(\d+)(?:-(\d{4}))?$')),
    ('hit_monthly', re.compile(r'^what-price-will-(bitcoin|ethereum|solana|xrp)-hit-in-([a-z]+)(?:-(\d{4}))?$')),
    ('hit_yearly', re.compile(r'^what-price-will-(bitcoin|ethereum|solana|xrp)-hit-(?:before|in)-(\d{4})$')),
    ('hit_when', re.compile(r'^when-will-(bitcoin|ethereum|solana|xrp)-hit-([0-9pk]+)$')),
]


def num(s):
    s = s.replace(',', '').replace('$', '').strip()
    mult = 1
    if s.lower().endswith('k'):
        mult, s = 1000, s[:-1]
    return float(s) * mult


def classify(slug):
    for kind, rx in SLUG_RE:
        m = rx.match(slug)
        if m:
            return kind, m
    return None, None


def ts(x):
    return pd.Timestamp(x).tz_convert('UTC') if pd.Timestamp(x).tzinfo else pd.Timestamp(x).tz_localize('UTC')


def et_midnight_utc(date):
    """00:00 America/New_York on `date` expressed in UTC."""
    return pd.Timestamp(date).tz_localize('America/New_York').tz_convert('UTC')


def parse_market(ev, m):
    """Return a dict describing one market's payoff, or None if not a price-threshold market we model."""
    kind, mm = classify(ev['slug'])
    if kind is None:
        return None
    und = UNDER[mm.group(1)]
    title = (m.get('groupItemTitle') or '').strip()
    q = m.get('question') or ''
    desc = m.get('description') or ''
    end = ts(m.get('endDate') or ev.get('endDate'))
    created = ts(m.get('createdAt') or ev.get('createdAt'))
    toks = jl(m.get('clobTokenIds')) or [None, None]
    outs = jl(m.get('outcomes')) or ['Yes', 'No']
    r = dict(event=ev['slug'], kind=kind, und=und, mid_id=m['id'], cid=m.get('conditionId'), q=q, title=title,
             yes=toks[0], no=toks[1] if len(toks) > 1 else None, outcomes=outs, end=end, created=created,
             tick=float(m.get('orderPriceMinTickSize') or 0.01), closed=bool(m.get('closed')),
             accepting=m.get('acceptingOrders'), vol=float(m.get('volume') or 0), negRisk=m.get('negRisk'),
             rewMin=m.get('rewardsMinSize'), rewSpread=m.get('rewardsMaxSpread'),
             outcomePrices=jl(m.get('outcomePrices')))
    try:
        if kind in ('above_daily', 'above_hourly'):
            r.update(payoff='digital', K=num(title), lo=None, hi=None, T=end)
            if 'higher than' not in desc and 'above' not in q.lower():
                return None
        elif kind == 'range_daily':
            if title.startswith('<'):
                r.update(payoff='range', lo=None, hi=num(title[1:]))
            elif title.startswith('>'):
                r.update(payoff='range', lo=num(title[1:]), hi=None)
            else:
                a, b = title.split('-')
                r.update(payoff='range', lo=num(a), hi=num(b))
            r.update(K=None, T=end)
        elif kind.startswith('hit_') and kind != 'hit_when':
            if title.startswith('↑'):
                r.update(payoff='touch_up', K=num(title[1:]))
            elif title.startswith('↓'):
                r.update(payoff='touch_down', K=num(title[1:]))
            else:
                return None
            # window start: ET midnight of first day, or market creation if description says so
            if kind == 'hit_daily':
                y = int(mm.group(4) or end.year)
                ws = et_midnight_utc(dt.date(y, MONTHS[mm.group(2)], int(mm.group(3))))
            elif kind == 'hit_weekly':
                y = int(mm.group(5) or end.year)
                ws = et_midnight_utc(dt.date(y, MONTHS[mm.group(2)], int(mm.group(3))))
            elif kind == 'hit_weekly2':
                y = int(mm.group(6) or end.year)
                ws = et_midnight_utc(dt.date(y, MONTHS[mm.group(2)], int(mm.group(3))))
            elif kind == 'hit_monthly':
                y = int(mm.group(3) or end.year)
                ws = et_midnight_utc(dt.date(y, MONTHS[mm.group(2)], 1))
            else:
                ws = created
            if 'creation of this market' in desc:
                ws = max(ws, created)
            r.update(T=end, win_start=ws)
        elif kind == 'hit_when':
            r.update(payoff='touch_up', K=num(mm.group(2).replace('pt', '.')), T=end, win_start=created)
            if 'dip' in q.lower():
                r['payoff'] = 'touch_down'
            # "when will X hit" markets count from creation of the event (series); conservative: event creation
            r['win_start'] = ts(ev.get('createdAt') or m.get('createdAt'))
        else:
            return None
    except Exception:
        return None
    if (r.get('K') is not None and r['K'] <= 0):
        return None
    return r


# ---------------------------------------------------------------- spot data (Binance mirror)
def klines(symbol, interval, start_ms, end_ms, limit=1000):
    out = []
    cur = int(start_ms)
    step = {'1m': 60e3, '5m': 300e3, '15m': 900e3, '1h': 3600e3, '1d': 86400e3}[interval]
    while cur < end_ms:
        d = get(f'{BV}/klines', {'symbol': symbol, 'interval': interval, 'startTime': cur, 'endTime': int(end_ms),
                                 'limit': limit})
        if not d:
            break
        out.extend(d)
        nxt = int(d[-1][0] + step)
        if nxt <= cur or len(d) < limit:
            break
        cur = nxt
    if not out:
        return pd.DataFrame(columns=['t', 'o', 'h', 'l', 'c', 'v'])
    df = pd.DataFrame([[r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in out],
                      columns=['t', 'o', 'h', 'l', 'c', 'v']).drop_duplicates('t')
    df['t'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    return df.set_index('t')


def spot_now(symbols):
    d = get(f'{BV}/ticker/price', {'symbols': '[' + ','.join(f'"{s}"' for s in symbols) + ']'})
    return {x['symbol']: float(x['price']) for x in d}


def running_extremes(symbol, start, now=None):
    """(max high, min low) of Binance 1m candles from `start` to now (uses 1h candles for the bulk)."""
    now = now or pd.Timestamp.now(tz='UTC')
    start = ts(start)
    if start >= now:
        return np.nan, np.nan
    h0 = start.ceil('h')
    parts = []
    if h0 > start:
        parts.append(klines(symbol, '1m', start.value // 10**6, min(h0, now).value // 10**6))
    if h0 < now:
        parts.append(klines(symbol, '1h', h0.value // 10**6, now.value // 10**6))
    df = pd.concat([p for p in parts if len(p)])
    return df['h'].max(), df['l'].min()


def realized_vol(symbol, days=7, interval='5m', end=None):
    end = end or pd.Timestamp.now(tz='UTC')
    k = klines(symbol, interval, (end - pd.Timedelta(days=days)).value // 10**6, end.value // 10**6)
    r = np.log(k['c']).diff().dropna()
    per = {'1m': 525600, '5m': 105120, '15m': 35040, '1h': 8760}[interval]
    return float(r.std() * math.sqrt(per))


# ---------------------------------------------------------------- Deribit surface
MON3 = {m: i for i, m in enumerate(['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 1)}


def parse_deribit_exp(s):
    m = re.match(r'(\d{1,2})([A-Z]{3})(\d{2})', s)
    return pd.Timestamp(dt.datetime(2000 + int(m.group(3)), MON3[m.group(2)], int(m.group(1)), 8, 0), tz='UTC')


class Surface:
    """Deribit mark-IV surface for one underlying. Smile per expiry in standardized moneyness
    x = ln(K/F)/sqrt(tau); total-variance interpolation in time at fixed x."""

    def __init__(self, und, rows, index_price, now):
        self.und, self.now, self.index = und, now, index_price
        pref = DERIBIT_MAP[und][1] + '-'
        by = {}
        for r in rows:
            n = r['instrument_name']
            if not n.startswith(pref) or r.get('mark_iv') is None:
                continue
            _, e, k, cp = n.split('-')
            K = float(k.replace('d', '.'))
            exp = parse_deribit_exp(e)
            tau = (exp - now).total_seconds() / YEAR_S
            if tau <= 0.5 / 365 / 24:
                continue
            F = r['underlying_price']
            by.setdefault(exp, {'tau': tau, 'F': F, 'pts': {}})
            # keep OTM side (both sides share the mark surface anyway)
            if (cp == 'C' and K >= F) or (cp == 'P' and K < F) or K not in by[exp]['pts']:
                by[exp]['pts'][K] = r['mark_iv'] / 100.0
        self.exps = []
        for exp in sorted(by):
            d = by[exp]
            Ks = np.array(sorted(d['pts']))
            if len(Ks) < 5:
                continue
            iv = np.array([d['pts'][k] for k in Ks])
            x = np.log(Ks / d['F']) / math.sqrt(d['tau'])
            atm = float(np.interp(0.0, x, iv))
            self.exps.append(dict(exp=exp, tau=d['tau'], F=d['F'], x=x, iv=iv, atm=atm))
        self.taus = np.array([e['tau'] for e in self.exps])

    def fwd_ratio(self, tau):
        """forward / index for maturity tau (interpolated basis)."""
        rs = np.array([e['F'] / self.index for e in self.exps])
        return float(np.interp(tau, self.taus, rs, left=rs[0], right=rs[-1]))

    def vol(self, K, tau, F):
        x = math.log(K / F) / math.sqrt(tau)
        E = self.exps
        if tau <= E[0]['tau']:
            return float(np.interp(x, E[0]['x'], E[0]['iv']))
        if tau >= E[-1]['tau']:
            return float(np.interp(x, E[-1]['x'], E[-1]['iv']))
        i = int(np.searchsorted(self.taus, tau))
        a, b = E[i - 1], E[i]
        wa = np.interp(x, a['x'], a['iv']) ** 2 * a['tau']
        wb = np.interp(x, b['x'], b['iv']) ** 2 * b['tau']
        w = wa + (wb - wa) * (tau - a['tau']) / (b['tau'] - a['tau'])
        return float(math.sqrt(max(w, 1e-12) / tau))

    def atm(self, tau, F):
        return self.vol(F, tau, F)


def deribit_surfaces(unds=('BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'), now=None):
    now = now or pd.Timestamp.now(tz='UTC')
    cache = {}
    out = {}
    for u in unds:
        ccy, pref, idx = DERIBIT_MAP[u]
        if ccy not in cache:
            cache[ccy] = get(f'{DERIBIT}/get_book_summary_by_currency', {'currency': ccy, 'kind': 'option'})['result']
        ip = get(f'{DERIBIT}/get_index_price', {'index_name': idx})['result']['index_price']
        out[u] = Surface(u, cache[ccy], ip, now)
    return out


def dvol_history(ccy, start, end, resolution=3600):
    """Deribit DVOL candles (close) -> Series in vol units (0.45 = 45%)."""
    out = []
    s, e = int(ts(start).value // 10**6), int(ts(end).value // 10**6)
    cur_end = e
    while True:
        d = get(f'{DERIBIT}/get_volatility_index_data', {'currency': ccy, 'start_timestamp': s,
                                                           'end_timestamp': cur_end, 'resolution': resolution})['result']
        rows = d.get('data') or []
        out.extend(rows)
        cont = d.get('continuation')
        if not cont or not rows or cont <= s:
            break
        cur_end = cont
    if not out:
        return pd.Series(dtype=float)
    df = pd.DataFrame(out, columns=['t', 'o', 'h', 'l', 'c']).drop_duplicates('t').sort_values('t')
    return pd.Series(df['c'].values / 100.0, index=pd.to_datetime(df['t'], unit='ms', utc=True))


# ---------------------------------------------------------------- pricing models
def bs_call(F, K, sig, tau):
    if tau <= 0 or sig <= 0:
        return max(F - K, 0.0)
    s = sig * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * s * s) / s
    return F * norm.cdf(d1) - K * norm.cdf(d1 - s)


def p_above_flat(S, K, sig, tau, F=None):
    """P(S_T > K) lognormal, risk-neutral drift to forward F."""
    F = F or S
    if tau <= 0:
        return float(S > K)
    s = sig * math.sqrt(tau)
    return float(norm.cdf((math.log(F / K) - 0.5 * s * s) / s))


def p_above_smile(K, tau, F, volfn):
    """-dC/dK with smile-dependent vol (includes the skew term)."""
    if tau <= 0:
        return float(F > K)
    sig = volfn(K)
    h = K * max(1e-4, 0.02 * sig * math.sqrt(tau))
    c1 = bs_call(F, K - h, volfn(K - h), tau)
    c2 = bs_call(F, K + h, volfn(K + h), tau)
    return float(min(1.0, max(0.0, -(c2 - c1) / (2 * h))))


def p_touch(S, H, sig, tau, mu=0.0):
    """Continuous-monitoring one-touch probability for GBM with drift mu (dS/S = mu dt + sig dW)."""
    if tau <= 0 or sig <= 0:
        return float(S >= H) if H >= S else float(S <= H)
    nu = mu - 0.5 * sig * sig
    st = sig * math.sqrt(tau)
    b = math.log(H / S)
    if H >= S:
        return float(norm.cdf((-b + nu * tau) / st) + math.exp(2 * nu * b / sig ** 2) * norm.cdf((-b - nu * tau) / st))
    return float(norm.cdf((b - nu * tau) / st) + math.exp(2 * nu * b / sig ** 2) * norm.cdf((b + nu * tau) / st))


def model_prob(r, S, now, surf=None, rv=None, hi_so_far=None, lo_so_far=None):
    """Fair YES probability under (a) Deribit smile ('iv'), (b) ATM Deribit vol flat ('atm'),
    (c) realized vol ('rv'). Returns dict."""
    tau = max((r['T'] - now).total_seconds(), 0) / YEAR_S
    out = {'tau_h': tau * 365 * 24}
    F = S * (surf.fwd_ratio(tau) if surf else 1.0)
    mu = math.log(F / S) / tau if tau > 0 else 0.0
    vols = {}
    if surf is not None and tau > 0:
        vols['iv'] = lambda K: surf.vol(K, tau, F)
        atm = surf.atm(tau, F)
        vols['atm'] = lambda K, a=atm: a
        out['atm_iv'] = atm
    if rv:
        vols['rv'] = lambda K, a=rv: a
        out['rv'] = rv
    p = r['payoff']
    r = dict(r)
    for key in ('lo', 'hi', 'K'):
        if r.get(key) is not None and isinstance(r[key], float) and math.isnan(r[key]):
            r[key] = None
    for name, vf in vols.items():
        if p == 'digital':
            val = p_above_smile(r['K'], tau, F, vf) if name == 'iv' else p_above_flat(S, r['K'], vf(r['K']), tau, F)
            out['sig_' + name] = vf(r['K'])
        elif p == 'range':
            def D(K):
                if K is None:
                    return None
                return p_above_smile(K, tau, F, vf) if name == 'iv' else p_above_flat(S, K, vf(K), tau, F)
            lo = 1.0 if r['lo'] is None else D(r['lo'])
            hi = 0.0 if r['hi'] is None else D(r['hi'])
            val = lo - hi
            kk = r['lo'] if r['lo'] is not None else r['hi']
            out['sig_' + name] = vf(kk)
        elif p in ('touch_up', 'touch_down'):
            H = r['K']
            if p == 'touch_up' and hi_so_far is not None and hi_so_far >= H:
                val = 1.0
            elif p == 'touch_down' and lo_so_far is not None and lo_so_far <= H:
                val = 1.0
            elif (p == 'touch_up' and S >= H) or (p == 'touch_down' and S <= H):
                val = 1.0
            else:
                sg = vf(H)
                val = p_touch(S, H, sg, tau, mu)
                out['sig_' + name] = sg
        else:
            continue
        out['p_' + name] = float(min(1, max(0, val)))
    return out


def book_top(b):
    """best bid/ask (+size) from a CLOB /books entry."""
    bids = [(float(x['price']), float(x['size'])) for x in b.get('bids', [])]
    asks = [(float(x['price']), float(x['size'])) for x in b.get('asks', [])]
    bb = max(bids) if bids else (np.nan, 0.0)
    ba = min(asks) if asks else (np.nan, 0.0)
    return dict(bid=bb[0], bid_sz=bb[1], ask=ba[0], ask_sz=ba[1],
                bid_depth5=sum(s for p, s in bids if not np.isnan(bb[0]) and p >= bb[0] - 0.05),
                ask_depth5=sum(s for p, s in asks if not np.isnan(ba[0]) and p <= ba[0] + 0.05))
