#!/bin/sh
# Forward out-of-sample replay of the frozen production model on days after its training cutoff.
# Same print-based fill rules as the backtest; covers every city/hour regardless of live-bot uptime.
# usage: research/weather/forward_replay.sh FROM_DATE [TO_DATE_EXCLUSIVE]
set -e
cd "$(dirname "$0")/../.."
FROM=${1:-2026-09-26}; TO=${2:-$(date -u +%F)}
python3 research/weather/fetch_weather_events.py "$FROM" "$TO"
python3 research/weather/fetch_weather_trades2.py "$FROM"
python3 research/weather/fetch_metar.py "$TO" refetch
[ -d data/nowcast_train ] || cp -r data/nowcast data/nowcast_train   # keep the shards the prod model was built from
rm -rf data/nowcast && mkdir data/nowcast
python3 research/weather/nowcast.py high && python3 research/weather/nowcast.py low
DATE_MIN="$FROM" DS_OUT=data/stack_fwd.parquet python3 -c "import sys; sys.path.insert(0,'research/weather'); import stack; stack.build()"
python3 - <<'PY'
import pickle, pandas as pd, sys
sys.path.insert(0, 'research/weather')
from stack import SF
df = pd.read_parquet('data/stack_fwd.parquet')
df = df[df.ref.notna()].assign(kindH=lambda d: (d.kind == 'high').astype(int))
clf = pickle.load(open('data/stack_model_prod.pkl', 'rb'))
df['p'] = clf.predict_proba(df[SF])[:, 1]
df.to_parquet('data/stack_fwd_test.parquet')
print('forward rows', len(df), 'dates', sorted(df.date.unique()))
PY
for cap in 50 3; do
  TEST=data/stack_fwd_test.parquet EDGE=0.15 MAXUSD=$cap TAG=fwd_cap$cap python3 research/weather/bt_stack.py > data/bt_fwd_cap$cap.log
done
cat data/bt_fwd_cap50.log
