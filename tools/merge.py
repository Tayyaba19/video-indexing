"""Merge out/chunk_*.jsonl + out/resume_*.jsonl into results.jsonl(.gz), de-duplicating re-processed videos
(the latest file wins), and validate coverage against the parquet files."""
import glob
import gzip
import json
import os
import re
import sys

import pyarrow.parquet as pq


def order(p):
    kind, n = re.match(r".*(chunk|resume)_(\d+)\.jsonl$", p).groups()
    return (kind == "resume", int(n))


files = sorted(glob.glob("out/chunk_*.jsonl") + glob.glob("out/resume_*.jsonl"), key=order)
by_video = {}  # video -> lines from the latest file that has it
for p in files:
    cur = {}
    with open(p, encoding="utf-8") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except ValueError:  # half-written last line of a killed run
                continue
            cur.setdefault(r["video"], []).append(r)
    by_video.update(cur)
    print(f"{p}: {len(cur)} videos", file=sys.stderr)

# validate
expected = {}
for f in sorted(glob.glob("finevideo/data/*.parquet")):
    name = os.path.basename(f)
    for r in range(pq.ParquetFile(f).metadata.num_rows):
        expected[f"{name}:{r}"] = True
missing = [v for v in expected if v not in by_video]
bad, n_lines, n_err = [], 0, 0
for v, rows in by_video.items():
    secs = [r["second"] for r in rows if "second" in r]
    errs = [r for r in rows if "error" in r]
    n_err += len(errs)
    if secs != list(range(len(secs))) or len(errs) > 1 or (not secs and not errs):
        bad.append(v)
    n_lines += len(rows)
print(f"videos {len(by_video)}/{len(expected)}  missing {len(missing)}  malformed {len(bad)}  "
      f"lines {n_lines}  error lines {n_err}  hours {sum(1 for rs in by_video.values() for r in rs if 'second' in r)/3600:.1f}",
      file=sys.stderr)
if missing:
    print("missing:", missing[:20], file=sys.stderr)
if bad:
    print("malformed:", bad[:20], file=sys.stderr)

if "--write" in sys.argv:
    with open("results.jsonl", "w", encoding="utf-8") as f, gzip.open("results.jsonl.gz", "wt", encoding="utf-8") as g:
        for v in sorted(by_video, key=lambda s: (s.split(":")[0], int(s.split(":")[1]))):
            for r in by_video[v]:
                line = json.dumps(r) + "\n"
                f.write(line)
                g.write(line)
    print("wrote results.jsonl and results.jsonl.gz", file=sys.stderr)
