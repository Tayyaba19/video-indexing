#!/usr/bin/env bash
# Resume: run only files that still have unfinished videos, 6 files per run.py process (memory is released between groups).
cd "$(dirname "$0")/.."
echo "driver2 start $(date +%T)" >> out/driver.log
T0=$(date +%s)
todo=$(python - <<'PY' | tr -d '\r'
import glob, os, pyarrow.parquet as pq, run
outs = sorted(glob.glob("out/*.jsonl"))
done = run.load_complete_videos(outs)
for f in sorted(glob.glob("finevideo/data/*.parquet")):
    name = os.path.basename(f)
    n = pq.ParquetFile(f).metadata.num_rows
    if sum(f"{name}:{r}" in done for r in range(n)) < n:
        print(name)
PY
)
echo "files with unfinished videos: $(echo $todo | wc -w)" >> out/driver.log
set -- $todo
g=0
while [ $# -gt 0 ]; do
  grp="${@:1:6}"; shift $(( $# < 6 ? $# : 6 )); g=$((g+1)); ts=$(date +%s)
  echo "group $g start $(date +%T): $grp" >> out/driver.log
  python -u run.py --input finevideo/data --output out/resume_$g.jsonl --workers 8 --files $grp \
     --skip-from $(ls out/*.jsonl | grep -v "resume_$g.jsonl") >> out/run_resume_$g.log 2>&1
  echo "group $g end $(date +%T) exit=$? ($(( $(date +%s)-ts ))s)" >> out/driver.log
done
echo "driver2 done $(date +%T), total $(( $(date +%s)-T0 ))s" >> out/driver.log
