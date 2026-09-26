"""Live/backtest parity: recompute the stacked probability for logged live signals using the archival pipeline
(IEM METAR archive + data-api prints) at the same timestamps, and compare with what the live bot computed."""
import json, sys, os, pickle, time
import numpy as np, pandas as pd, requests, io
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from polylib import *
from stations import STATIONS
from features_live import features_at
from train_nowcast import prep, FEATS, KMAX
from bt_nowcast import bucket_probs
from stack import SF
from wlib import parse_bucket

models = {k: pickle.load(open(f'data/nowcast_{k}.pkl', 'rb')) for k in ('high', 'low')}
stack = pickle.load(open('data/stack_model_prod.pkl', 'rb'))
S = pd.DataFrame([json.loads(l) for l in open('data/live/weather/signals.jsonl')])
S['city'] = S.slug.str.replace(r'-on-.*', '', regex=True).str.replace('highest-temperature-in-', '').str.replace('lowest-temperature-in-', '')
S['kind'] = np.where(S.slug.str.startswith('highest'), 'high', 'low')
samp = S.drop_duplicates(['slug', 'ts']).sample(min(25, S.drop_duplicates(['slug', 'ts']).shape[0]), random_state=1)
iem = {}


def iem_metar(icao, unit):
    if icao not in iem:
        now = time.gmtime()
        url = ('https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=%s&data=tmpc&year1=2026&month1=9&day1=24'
               '&year2=2026&month2=9&day2=27&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3&report_type=4') % icao
        for a in range(6):
            txt = requests.get(url, timeout=120).text
            if txt.startswith('station'):
                break
            time.sleep(15 * (a + 1))
        df = pd.read_csv(io.StringIO(txt), na_values=['M']).dropna(subset=['tmpc'])
        df['ts'] = (pd.to_datetime(df['valid'], utc=True) - pd.Timestamp('1970-01-01', tz='UTC')) // pd.Timedelta(seconds=1)
        df['tf'] = df.tmpc * 9 / 5 + 32 if unit == 'F' else df.tmpc
        df['t'] = np.round(df.tf + 1e-9).astype(int)
        iem[icao] = df[['ts', 't', 'tf']].sort_values('ts').reset_index(drop=True)
        time.sleep(3)
    return iem[icao]


out = []
for r in samp.itertuples():
    icao, unit, tz = STATIONS[r.city]
    ev = get(f'{GAMMA}/events', {'slug': r.slug})[0]
    bks = sorted([(parse_bucket(m.get('groupItemTitle')), m['conditionId']) for m in ev['markets'] if parse_bucket(m.get('groupItemTitle'))], key=lambda x: x[0][0])
    f = features_at(iem_metar(icao, unit), r.city, r.kind, r.ts)
    if f is None:
        continue
    P = models[r.kind].predict_proba(prep(pd.DataFrame([f]))[FEATS])
    run = f['run'] if r.kind == 'high' else -f['run']
    Q = bucket_probs(P, np.array([run]), [b[0] for b in bks], r.kind)[0]
    live = S[(S.slug == r.slug) & (S.ts == r.ts)].set_index('bucket')
    for j, (rng, cid) in enumerate(bks):
        title = [m.get('groupItemTitle') for m in ev['markets'] if m['conditionId'] == cid][0]
        if title not in live.index:
            continue
        L = live.loc[title]
        out.append(dict(slug=r.slug[:40], bucket=title, run_live=L['run'], run_arch=f['run'], q_live=L['q'], q_arch=Q[j]))
D = pd.DataFrame(out)
pd.set_option('display.width', 200)
print(D.round(3).head(30).to_string())
print('rows', len(D), '| run mismatches', int((D.run_live != D.run_arch).sum()), '| max |q_live - q_arch| %.4f' % (D.q_live - D.q_arch).abs().max())
