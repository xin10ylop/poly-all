#!/bin/sh
# Rolling walk-forward: retrain every ~2 weeks, trade the next window with print-based fills.
cd /home/user/poly-all
for pair in "2026-08-15 2026-09-01" "2026-09-01 2026-09-15" "2026-09-15 2026-09-27"; do
  set -- $pair
  SPLIT=$1 TEST_END=$2 python3 research/weather/stack.py > data/roll_fit_$1.log 2>&1
  TEST=data/stack_test_$1.parquet EDGE=0.15 TAG=roll_$1 python3 research/weather/bt_stack.py > data/roll_bt_$1.log 2>&1
done
echo done > data/roll_done.flag
