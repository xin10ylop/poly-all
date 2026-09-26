"""Minimal client for the OpenRouter Decisions API (TypeSafe Jev) with disk cache + cost tracking."""
import hashlib, json, os, sqlite3, threading, time
import requests
from polylib import load_env

URL = 'https://openrouter.ai/api/alpha/decisions'
MODEL = os.environ.get('JEV_MODEL', 'typesafe/jev-1.13')   # pinned (docs: pin when tuning thresholds)
_db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'jev_cache.sqlite')
_lock = threading.Lock()
_conn = None
spent = {'usd': 0.0, 'calls': 0, 'cached': 0}


def _db():
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(_db_path), exist_ok=True)
        _conn = sqlite3.connect(_db_path, check_same_thread=False)
        _conn.execute('create table if not exists c (k text primary key, v text)')
    return _conn


def decide(state, questions, model=MODEL, retries=5, budget_usd=None):
    """Return the `answers` dict. Cached on (model, state, questions)."""
    load_env()
    key = hashlib.sha256(json.dumps([model, state, questions], sort_keys=True).encode()).hexdigest()
    with _lock:
        row = _db().execute('select v from c where k=?', (key,)).fetchone()
    if row:
        spent['cached'] += 1
        return json.loads(row[0])
    if budget_usd is not None and spent['usd'] >= budget_usd:
        raise RuntimeError('jev budget exhausted')
    body = {'model': model, 'state': state, 'questions': questions}
    hdr = {'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY'], 'Content-Type': 'application/json',
           'X-Title': 'poly-research'}
    for a in range(retries):
        try:
            r = requests.post(URL, json=body, headers=hdr, timeout=60)
            if r.status_code in (402, 429, 500, 502, 503):
                time.sleep(float(r.headers.get('Retry-After', 2 ** a)))
                continue
            r.raise_for_status()
            d = r.json()
            ans = d['answers']
            with _lock:
                spent['usd'] += float((d.get('usage') or {}).get('cost') or 0)
                spent['calls'] += 1
                _db().execute('insert or replace into c values (?,?)', (key, json.dumps(ans)))
                _db().commit()
            return ans
        except requests.RequestException:
            time.sleep(2 ** a)
    raise RuntimeError('jev request failed')
