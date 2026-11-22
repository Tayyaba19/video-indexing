#!/usr/bin/env bash
# Process completed parquet files in chunks until all 50 are done. Logs timing per chunk.
cd "$(dirname "$0")/.."
touch out/done.txt
T0=$(date +%s); echo "driver start $(date +%T)" >> out/driver.log
i=0
while true; do
  todo=$(comm -23 <(ls finevideo/data | grep -E 'train-000[0-4][0-9]-of-01357\.parquet$' | sort) <(sort out/done.txt))
  ndone=$(wc -l < out/done.txt)
  if [ -z "$todo" ]; then
    [ "$ndone" -ge 50 ] && break
    sleep 30; continue
  fi
  i=$((i+1)); ts=$(date +%s)
  echo "chunk $i start $(date +%T): $(echo $todo | wc -w) files" >> out/driver.log
  python run.py --input finevideo/data --output out/chunk_$i.jsonl --files $todo 2>&1 | grep "^\[run\]" >> out/driver.log
  echo "$todo" | tr ' ' '\n' >> out/done.txt
  echo "chunk $i end $(date +%T) ($(( $(date +%s)-ts ))s)" >> out/driver.log
done
echo "driver done $(date +%T), total $(( $(date +%s)-T0 ))s" >> out/driver.log
