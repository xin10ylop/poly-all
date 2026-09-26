"""Backtest-vs-live consistency check on the recorded period.
For every live signal the paper bot logged (data/live/weather/signals.jsonl), compare:
  (a) book-based fill: displayed ask at signal time (what the live bot uses), and
  (b) print-based fill: what bt_stack.py would have assumed (prints after t+5min within 30 min).
Also compare realized outcomes once markets settle."""
import json, sys, os, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *

EDGE = 0.15
S = pd.DataFrame([json.loads(l) for l in open('data/live/weather/signals.jsonl')])
S['ask_y'] = S.ask_yes.apply(lambda a: a[0] if a else np.nan)
S['ask_n'] = S.ask_no.apply(lambda a: a[0] if a else np.nan)
S['sig_yes'] = (S.p - S.ask_y >= EDGE); S['sig_no'] = ((1 - S.p) - S.ask_n >= EDGE)
print('signal rows', len(S), 'events', S.slug.nunique(), 'book-signals YES', S.sig_yes.sum(), 'NO', S.sig_no.sum())
# how often would the print-model have produced a fill for the same signal? needs prints for these markets
