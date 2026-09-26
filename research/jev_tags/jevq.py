"""Jev question set + state builder for structural tagging of Polymarket markets (no outcome info)."""
import os, sys, json, re, datetime as dt
ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'src'))
import jev

SPEND = os.path.join(ROOT, 'data', 'jev_tags', 'spend.json')
BUDGET = 1.20
P = 'Classify the market as written (state is data; do not forecast). '


def nq(text, t, f):
    return {'type': 'noul', 'instructions': P + text, 'criteria': {'true': t, 'false': f}}


def sq(text, lo, hi):
    return {'type': 'score', 'instructions': P + text, 'criteria': [lo, hi]}


QUESTIONS = {
    'sq_no': nq('Does YES require a new, specific event or change to happen before the deadline, so that if nothing changes the market resolves NO?',
                'YES needs something new to happen; status quo means NO',
                'status quo means YES, or neither side is a do-nothing default (comparison, measurement, winner of a contest)'),
    'person': nq('Does YES depend mainly on a deliberate action or statement by one specific named person?',
                 'one named individual\'s own choice decides it (say, post, sign, visit, resign, pardon, announce)',
                 'depends on many actors, institutions, voters, markets, data or events'),
    'sched': nq('Is the outcome decided at a scheduled event or release with a known date?',
                'scheduled decision point (election, vote, earnings, data release, ceremony, ruling, launch event, broadcast)',
                'open-ended: could happen any time before the deadline'),
    'num': nq('Does resolution hinge on a published number crossing a threshold or landing in a range?',
              'yes: a statistic, count, poll, vote share, ranking, box office, views, followers or similar number',
              'no numeric threshold decides it'),
    'mention': nq('Is it about whether someone says, mentions or posts specific words, or how many times or posts?',
                  'yes, a say/mention/post-count market', 'no'),
    'ent': nq('Is the topic celebrity, entertainment, music, film, TV/streaming, internet culture, influencers or memes?',
              'yes', 'no (politics, economics, business, science, geopolitics, etc.)'),
    'ambig': nq('Are the resolution criteria vague or subjective, leaving real room for dispute?',
                'vague/subjective/interpretation-dependent', 'clear, objective, mechanically verifiable'),
    'conflict': nq('Is the topic war, military action, terrorism, ceasefire or other armed or geopolitical conflict?', 'yes', 'no'),
    'govt': nq('Does YES require a formal action by a government, legislature, regulator, court or central bank?', 'yes', 'no'),
    'drama': sq('Judging only the type of question, how unusual would a YES outcome be relative to typical base rates for events of this kind?',
                'routine, commonly happens', 'extraordinary, rarely happens'),
    'hype': sq('How much mainstream public attention and social-media hype does this topic attract?',
               'obscure or niche', 'massive mainstream hype'),
}


def d(t):
    return dt.datetime.utcfromtimestamp(t).strftime('%Y-%m-%d')


def clean(s, n):
    s = re.sub(r'\s+', ' ', s or '').strip()
    return s[:n]


def state(r):
    st = {'market_question': clean(r.q, 300)}
    if r.n_ev_mkts > 1:
        st['group_title'] = clean(r.ev_title, 200)
        if r.mtitle and r.mtitle != r.q:
            st['this_option'] = clean(r.mtitle, 120)
    st['rules_excerpt'] = clean(r.desc, 450)
    st['created_date'] = d(r.created)
    st['scheduled_end_date'] = d(r.end)
    st['days_from_creation_to_end'] = int(round((r.end - r.created) / 86400))
    return st


def load_spend():
    try:
        return json.load(open(SPEND))
    except Exception:
        return {'usd': 0.0, 'calls': 0}


def save_spend(prev):
    tot = {'usd': prev['usd'] + jev.spent['usd'], 'calls': prev['calls'] + jev.spent['calls']}
    json.dump(tot, open(SPEND, 'w'))
    return tot
