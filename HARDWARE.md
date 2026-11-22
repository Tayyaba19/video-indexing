# Hardware used for the full 50-file run

My own laptop.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB (driver 592.82) |
| CPU | AMD Ryzen 7 7445HS, 6 cores / 12 threads |
| RAM | 16 GB (15.2 GB usable) |
| Disk | Samsung MZVL8512HFLU NVMe SSD, 512 GB |
| OS | Windows 11 Home |
| Software | Python 3.11, PyTorch 2.8.0+cu126, TensorRT 10.16, Ultralytics 8.4.216, PyAV 18.1.1 |

## How long the full run took

The run processed all 50 files (1,650 videos, 128.3 hours of video, 461,757 output lines, 0 error lines).
Total time from start to finish: **3 h 2 min** (19:32:02 to 22:34:34 on 22 Nov 2026), including the restart
described below.

Details, so the numbers are honest:

- The run started while the dataset was still downloading (the download finished at ~20:20), so the first
  part shared the CPU, disk and network with the download.
- Files 0–12 were processed at 4 sampled frames/s (58.4 min for 34.3 h of video, 141 model frames/s). I then
  changed the default to 3 frames/s (which also passes the accuracy check, see README) to finish before the
  deadline; files 13–49 were processed at 3 frames/s.
- At 21:24 the laptop was running low on RAM (the main process grew to ~4.5 GB because of pyarrow's memory pool
  holding the 0.5 GB parquet buffers). I stopped the run and resumed it with a fix (system memory pool,
  memory-mapped parquet, smaller queues, one process per 6 files). Videos that were already complete were
  skipped; the ~10 videos that were in progress were processed again. `results.jsonl.gz` is the merged,
  de-duplicated output (one result per video).
- This full run used the TensorRT FP16 backend. After it finished, an independent check (README, "Correctness
  check") showed TensorRT's people counts drift from the reference in crowded scenes, so the final `run.py`
  default is the PyTorch backend at 8 frames/s. The results file was not regenerated with the final default
  (it would take ~10+ hours on this laptop).
- Throughput on this laptop: ~34 seconds of video per second at 4 frames/s and ~48 at 3 frames/s. The GPU was
  the bottleneck (decoding alone runs at ~140 seconds of video per second with 10 worker processes).
