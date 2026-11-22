"""Benchmark YOLO11l in plain PyTorch on one FineVideo parquet file, for comparison with bench_trt.py.

Same decode pipeline, videos, batch size and classes as bench_trt.py; only the inference backend differs.
Use --half for FP16 autocast-free half precision (Ultralytics converts the model to .half()).

Usage:
  python bench_pt.py --parquet finevideo/data/train-00000-of-01357.parquet --weights yolo11l.pt --batch 8 --max-videos 5
  python bench_pt.py --parquet finevideo/data/train-00000-of-01357.parquet --weights yolo11l.pt --batch 8 --max-videos 5 --half
"""
import argparse
import queue
import threading
import time

import numpy as np
import torch
from ultralytics import YOLO

from bench_trt import CLASSES, producer


def run_batch(model, frames, imgsz, half, stats):
    results = model.predict(frames, imgsz=imgsz, conf=0.4, classes=CLASSES, half=half, device=0, verbose=False)
    n = len(frames)
    sp = results[0].speed  # ms per image, averaged over the batch (Ultralytics syncs CUDA in its timers)
    stats["pre_ms"] += sp["preprocess"] * n
    stats["inf_ms"] += sp["inference"] * n
    stats["post_ms"] += sp["postprocess"] * n
    stats["infer_frames"] += n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--weights", default="yolo11l.pt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-videos", type=int, default=5)
    ap.add_argument("--half", action="store_true", help="FP16 inference")
    args = ap.parse_args()

    precision = "FP16" if args.half else "FP32"
    print(f"Loading {args.weights} for PyTorch {precision} inference on {torch.cuda.get_device_name(0)}...")
    model = YOLO(args.weights, task="detect")
    dummy = [np.zeros((720, 1280, 3), np.uint8)] * args.batch
    for _ in range(3):  # warm-up (also fuses Conv+BN on first call)
        model.predict(dummy, imgsz=args.imgsz, half=args.half, device=0, verbose=False)

    stats = dict(videos=0, frames=0, decode_wall=0.0, pre_ms=0.0, inf_ms=0.0, post_ms=0.0, infer_frames=0)
    q = queue.Queue(maxsize=args.batch * 8)
    t = threading.Thread(target=producer, args=(args.parquet, args.max_videos, q, stats), daemon=True)

    t_start = time.perf_counter()
    t.start()
    buf = []
    while True:
        img = q.get()
        if img is None:
            break
        buf.append(img)
        if len(buf) == args.batch:
            run_batch(model, buf, args.imgsz, args.half, stats)
            buf = []
    if buf:  # PyTorch handles a short last batch, no padding needed
        run_batch(model, buf, args.imgsz, args.half, stats)
    wall = time.perf_counter() - t_start

    n = stats["infer_frames"]
    gpu_s = (stats["pre_ms"] + stats["inf_ms"] + stats["post_ms"]) / 1000
    print(f"\n===== RESULTS (PyTorch {precision}) =====")
    print(f"videos: {stats['videos']}   frames: {n}")
    print(f"per-frame ms  preprocess {stats['pre_ms']/n:.2f} | PyTorch {stats['inf_ms']/n:.2f} | NMS {stats['post_ms']/n:.2f}")
    print(f"PyTorch-only FPS         : {n / (stats['inf_ms']/1000):8.1f}")
    print(f"model FPS (pre+inf+nms)  : {n / gpu_s:8.1f}")
    print(f"decode FPS (overlapped)  : {n / stats['decode_wall']:8.1f}")
    print(f"END-TO-END FPS           : {n / wall:8.1f}   ({wall:.1f}s wall)")
    est_frames = 125 * 3600 * 30
    print(f"projected time for 125h @30fps, every frame: {est_frames / (n / wall) / 3600:.1f} h")


if __name__ == "__main__":
    main()
