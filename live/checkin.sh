#!/bin/sh
# Periodic check-in: make sure recorder + bot are running, then print the live paper-trading summary.
cd "$(dirname "$0")/.."
live/ensure_running.sh
date -u +"%F %T UTC"
tail -3 data/live/weather/bot.log
python3 live/report.py
