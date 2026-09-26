#!/bin/sh
# Start the book recorder and weather paper bot if they are not running (PID files in data/live/).
cd "$(dirname "$0")/.."
mkdir -p data/live/weather
alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
if ! alive data/live/recorder.pid; then
  nohup python3 live/book_recorder.py weather 60 >> data/rec_weather.log 2>&1 &
  echo $! > data/live/recorder.pid; echo "started recorder $(cat data/live/recorder.pid)"
fi
if ! alive data/live/weather/bot.pid; then
  nohup python3 live/weather_bot.py >> data/live/weather/bot.log 2>&1 &
  echo $! > data/live/weather/bot.pid; echo "started bot $(cat data/live/weather/bot.pid)"
fi
# optional minute-level sniper add-on (~$20-30/day backtest); off by default, enable with SNIPER=1
if [ "$SNIPER" = "1" ]; then
  mkdir -p data/live/sniper
  if ! alive data/live/sniper/sniper.pid; then
    nohup python3 live/sniper.py >> data/live/sniper/sniper.log 2>&1 &
    echo $! > data/live/sniper/sniper.pid; echo "started sniper $(cat data/live/sniper/sniper.pid)"
  fi
fi
