# AI chat history — guide

The `.jsonl` files in this folder are the complete, unedited Claude Code session logs for this project, copied from
`~/.claude/projects/`. Only secrets would be replaced with `[REMOVED]`. This page is a map of the main
engineering moments in them, in order, so they are easier to follow.

| # | What happened | Outcome |
|---|---|---|
| 1 | Dataset access (gated repo), first shard download | resolved by accepting the dataset terms |
| 2 | First TensorRT benchmark on every frame (`tools/bench_trt.py`) | 151 FPS end-to-end, projected 24.7 h for 125 h — GPU-bound |
| 3 | PyTorch baseline (`tools/bench_pt.py`) | ~66–70 FPS on every frame (~57 h projected) |
| 4 | Read the brief: grading compares against every-frame results with thresholds, so frame sampling is allowed | sampling became the main speed lever |
| 5 | Checked Ultralytics 8.4 internals | found the default NMS IoU is now 0.5, the brief needs 0.7 → set explicitly; replicated the letterbox exactly |
| 6 | Built `run.py`: parallel PyAV decode workers + batched inference + per-second aggregation per the grading rules | |
| 7 | Sampling experiment (12 videos, first 60 s) | 3–5 frames/s looked safe — later shown to be optimistic |
| 8 | Decode-only benchmark | decode was not the bottleneck (~140 s of video/s) |
| 9 | Full 50-file run, processing files as they downloaded | |
| 10 | Memory leak (pyarrow pool), process stopped by low-memory guard | diagnosed, fixed (system pool, memory-mapped parquet, process per 6 files), resumed without losing finished videos |
| 11 | Duplicate run writing the same file after a restart | caught from the process count, killed both, re-ran once cleanly |
| 12 | Full run finished: 1,650 videos, 128.3 h, 3 h 2 min | merged + validated (`tools/merge.py`) |
| 13 | Independent check on 12 fresh random videos | 3 frames/s failed the people thresholds in crowded scenes |
| 14 | Isolated the cause | TensorRT FP16 640×640 drifts from the reference even on every frame; sampling was not the main issue |
| 15 | Found a flaw in the reference itself | Ultralytics silently drops the rect letterbox for mixed-shape batches → batch by shape |
| 16 | `prepare.py` test | batch-32 engine build ran out of memory on 6 GB → engine made optional, with batch fallback |
| 17 | Final default: PyTorch backend at 8 frames/s | F1 0.992, people exact 0.966, ±1 1.000 on the independent sample |

Suggestions from the AI that were wrong or rejected (all visible in the logs): keyframe-only decoding (too few
frames per second for this task), 3–4 frames/s sampling (failed on crowded scenes), and TensorRT as the default
backend (fast but not accurate enough).
