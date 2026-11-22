# Video indexing at speed — YOLO11l

For each second of each video in a folder of FineVideo Parquet files, `run.py` records which of
person / car / dog / laptop / cell phone are present and how many people are visible, and writes `results.jsonl`.

## Install

Tested with Python 3.11 and an NVIDIA driver that supports CUDA 12.

```bash
sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0   # needed by opencv-python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Preparation command (run once, untimed, needs internet)

```bash
python prepare.py
```

Downloads the official `yolo11l.pt` (all the default run needs). It also tries to build an optional TensorRT FP16
engine for `--backend tensorrt`; if that build fails, the default run is not affected.

## Run

```bash
python run.py --input <folder of .parquet files> --output results.jsonl
```

No other flags are needed. Optional flags (development only):

| flag | meaning |
|---|---|
| `--sample-fps K` | frames per second the model runs on (default 8; `0` = every frame) |
| `--backend pytorch\|tensorrt` | default `pytorch` (Ultralytics `predict`, FP32 — same preprocessing as the reference); `tensorrt` = FP16 engine, faster but less accurate (see below) |
| `--workers N` | decode processes (default: CPU threads − 2, max 24) |
| `--batch N` | batch size for the pytorch backend (default 32) |
| `--reference` | same as `--backend pytorch`; with `--sample-fps 0` it is the slow every-frame reference used for checking |
| `--dump-frames F` | write per-frame class counts, used by `eval_sampling.py` |

## How it works

1. A reader thread walks the Parquet files row by row (`mp4` column only, memory-mapped) and queues
   `(file:row, bytes)`.
2. Decode worker processes (PyAV) decode every frame of a video and compute each frame's timestamp relative to the
   first decoded frame (exact `pts × time_base`). They keep the first frame of each `1/K`-second slot (K = 8 frames
   per second). Only kept frames are converted to BGR and rotated if the stream has a rotation tag.
3. The main process batches the kept frames **by shape** and runs Ultralytics `predict` with `yolo11l.pt`
   (imgsz 640, conf 0.4, IoU 0.7, class-aware NMS, classes person/car/dog/laptop/cell phone). Batching by shape
   matters: Ultralytics only uses its minimal-padding letterbox when all images in a batch have the same shape, so
   each frame is processed exactly as it would be on its own.
4. When a video finishes, each second is aggregated with the grading rule: an object is present if found in at
   least half of that second's sampled frames, and the people count is the median per-frame count rounded down.
   Every second up to the one holding the last frame gets a line. A damaged video gets an error line after the
   lines for the seconds that decoded.

Note: Ultralytics 8.4 changed the default predict NMS IoU to 0.5; the brief requires 0.7, so it is set explicitly.

## Correctness check

`eval_sampling.py` implements the three pass thresholds from the brief. The reference is
`run.py --reference --sample-fps 0` (Ultralytics `predict` with `yolo11l.pt`, FP32, on every frame).

Independent check: 12 videos picked at random from 12 different files (1,417 seconds, first 150 s of each), not
used for any tuning. Every-frame reference vs. each pipeline:

| pipeline | frames/s | object F1 (≥0.95) | people exact (≥0.85) | people ±1 (≥0.95) | bootstrap P(fail) |
|---|---|---|---|---|---|
| PyTorch (default) | 3 | 0.977 | 0.915 | 0.991 | 2.9% |
| PyTorch (default) | 5 | 0.987 | 0.944 | 0.998 | 0% |
| **PyTorch (default)** | **8** | **0.992** | **0.966** | **1.000** | **0%** |
| TensorRT FP16 640×640 | 8 | 0.980 | 0.913 | 0.978 | 8.7% |
| TensorRT FP16 640×640 | every frame | 0.988 | 0.917 | 0.972 | 21.5% |

TensorRT was 2× faster, but in crowded scenes its people counts differ from the reference even on every frame
(one video with ~5 people per frame: 0.46 exact vs 1.00). The engine uses a fixed 640×640 padded input in FP16,
while the reference uses a 640×384 input in FP32. That is why the default is PyTorch.

```bash
python run.py --input <dir> --output ref.jsonl --reference --sample-fps 0 --max-seconds 150 --only-videos <file:row ...>
python run.py --input <dir> --output out.jsonl --max-seconds 150 --only-videos <file:row ...>
python eval_sampling.py --ref-results ref.jsonl --results out.jsonl
```

Speed on the development laptop (RTX 4050 Laptop, every frame): PyTorch FP32 ~70 frames/s, TensorRT FP16
~142 frames/s.

## About `results.jsonl.gz`

The submitted results for the 50 files were produced **before** the independent check, with the TensorRT backend
at 4 frames/s (files 0–12) and 3 frames/s (files 13–49) — see HARDWARE.md. On the independent sample those results
score object F1 0.969, people exact 0.875, people ±1 0.965 (passing overall, but the 3 frames/s part alone is
below the people thresholds). There was not enough time before the deadline to re-run 128 hours of video on the
laptop with the final default.

## Repository layout

| path | what |
|---|---|
| `run.py` | the script |
| `prepare.py` | preparation command |
| `eval_sampling.py` | correctness check against the grading thresholds |
| `results.jsonl.gz` | results for all 50 files |
| `HARDWARE.md`, `WRITEUP.md` | hardware/timing and write-up answers |
| `tools/` | development scripts: first benchmarks (`bench_trt.py`, `bench_pt.py`), decode benchmark, sampling simulation, the drivers used for the full run, and the merge/validation script |
| `ai_chats/` | full AI chat history + a guide to it |
