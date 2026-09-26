"""Rules guard: Jev (TypeSafe) verifies each weather event's resolution rules before the bot trades it.
Cascade (per OpenRouter 'Jev-verified cascade' pattern): regex parser proposes a config; Jev answers typed
questions about the rules text; only if Jev confirms every field with high probability is the event tradable.
Disagreements are escalated to an LLM (Claude via OpenRouter) at most once per event and cached.
"""
import json, os, re, sys, hashlib
ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
import jev
import requests
from polylib import load_env

CACHE = os.path.join(ROOT, 'data', 'live', 'rules_guard.json')
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
THRESH = 0.85
# The standard boilerplate (data-correction / revision clauses) scores ~0.81-0.87 on 'unusual'; anything clearly
# above that baseline means the rules text changed in a non-standard way -> block and escalate.
UNUSUAL_MAX = 0.92


def regex_config(desc):
    m = re.search(r'timeseries\?site=([A-Za-z0-9]{4})', desc)
    src = 'NOAA' if m else ('WU' if 'wunderground' in desc else ('HKO' if 'weather.gov.hk' in desc else 'OTHER'))
    st = m.group(1).upper() if m else None
    if not st:
        m2 = re.search(r'wunderground\.com/history/daily/[a-z/\-]+/([A-Z0-9]{4})', desc)
        st = m2.group(1) if m2 else None
    unit = 'F' if re.search(r'whole degrees Fahrenheit|in degrees Fahrenheit', desc) else 'C'
    return dict(source=src, station=st, unit=unit)


def jev_check(desc, cfg, kind):
    state = {'market_rules': desc[:6000]}
    qs = {
        'noaa_hourly': {'type': 'noul', 'instructions': 'Does the market resolve using NOAA/National Weather Service observation data from the weather.gov time series page (hourly readings)?',
                        'criteria': {'true': 'Resolution source is NOAA weather.gov timeseries data', 'false': 'Another source (e.g. Wunderground, Hong Kong Observatory) is primary'}},
        'extreme_kind': {'type': 'choice', 'instructions': 'Which daily statistic decides the market?',
                         'criteria': {'highest': 'The highest temperature reading of the day', 'lowest': 'The lowest temperature reading of the day', 'other': 'Something else (average, specific hour, etc.)'}},
        'whole_day': {'type': 'noul', 'instructions': 'Does the market consider all readings over the entire calendar day (not just specific hours)?',
                      'criteria': {'true': 'All times on the day count', 'false': 'Only a subset of hours or a different period counts'}},
        'unit': {'type': 'choice', 'instructions': 'In which unit and precision is the market resolved?',
                 'criteria': {'fahrenheit_whole': 'Whole degrees Fahrenheit', 'celsius_whole': 'Whole degrees Celsius', 'other': 'Decimals or another unit'}},
        'unusual': {'type': 'noul', 'instructions': 'Does the rules text contain any unusual clause that could make the outcome differ from simply taking the max/min of the station readings for that day (for example a different station, averaging, manual adjustment, or an exclusion)?',
                    'criteria': {'true': 'There is such an unusual clause', 'false': 'No unusual clause; standard daily max/min of the station'}},
    }
    a = jev.decide(state, qs, budget_usd=1.0)
    ok = (a['extreme_kind']['choice'] == ('highest' if kind == 'high' else 'lowest') and a['extreme_kind']['probabilities'][a['extreme_kind']['choice']] >= THRESH
          and a['whole_day']['noul'] >= THRESH and a['unusual']['noul'] <= UNUSUAL_MAX
          and a['unit']['choice'] == ('fahrenheit_whole' if cfg['unit'] == 'F' else 'celsius_whole')
          and ((a['noaa_hourly']['noul'] >= THRESH) == (cfg['source'] == 'NOAA')))
    return ok, a


def llm_escalate(desc, cfg, kind, jev_answers, model='anthropic/claude-sonnet-5'):
    """Ask an LLM to adjudicate a disagreement; returns True if the standard config is correct."""
    load_env()
    prompt = ('You verify Polymarket weather market rules for an automated trader. Proposed config: ' + json.dumps(cfg) +
              f', statistic={kind}. The trader assumes: resolution = {kind}est of all station readings over the local calendar '
              'day, in whole units as stated, from the given source. Answer with exactly one word, YES if the rules text is fully '
              'consistent with this assumption, otherwise NO.\n\nRULES:\n' + desc[:6000])
    r = requests.post('https://openrouter.ai/api/v1/chat/completions', timeout=60,
                      headers={'Authorization': 'Bearer ' + os.environ['OPENROUTER_API_KEY']},
                      json={'model': model, 'max_tokens': 5, 'messages': [{'role': 'user', 'content': prompt}]})
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content'].strip().upper().startswith('YES')


def approve(slug, desc, kind, allow_llm=True):
    key = hashlib.sha1((slug + desc).encode()).hexdigest()
    if key in _cache:
        return _cache[key]['ok']
    cfg = regex_config(desc)
    ok, ans = jev_check(desc, cfg, kind)
    how = 'jev'
    if not ok and allow_llm and cfg['source'] in ('NOAA', 'WU'):
        try:
            ok = llm_escalate(desc, cfg, kind, ans); how = 'llm'
        except Exception as ex:
            ok = False; how = f'llm_error:{ex}'
    _cache[key] = dict(slug=slug, ok=ok, cfg=cfg, jev=ans, how=how)
    json.dump(_cache, open(CACHE, 'w'))
    return ok
