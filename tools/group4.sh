#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while tasklist //FI "PID eq 5388" 2>/dev/null | grep -qi python; do sleep 10; done
echo "group 3 end $(date +%T) (observed by waiter)" >> out/driver.log
ts=$(date +%s); echo "group 4 start $(date +%T): train-00046..49" >> out/driver.log
python -u run.py --input finevideo/data --output out/resume_4.jsonl --workers 8 \
  --files train-00046-of-01357.parquet train-00047-of-01357.parquet train-00048-of-01357.parquet train-00049-of-01357.parquet \
  --skip-from $(ls out/*.jsonl | grep -v "resume_4.jsonl") >> out/run_resume_4.log 2>&1
echo "group 4 end $(date +%T) exit=$? ($(( $(date +%s)-ts ))s)" >> out/driver.log
echo "driver2 done $(date +%T)" >> out/driver.log
