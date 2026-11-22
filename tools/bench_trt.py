"""Benchmark YOLO11l TensorRT throughput on one FineVideo parquet file.

Reports three numbers:
  * decode FPS     - how fast PyAV can decode frames (CPU side)
  * inference FPS  - preprocess + TensorRT inference + NMS, from Ultralytics' own timers
  * end-to-end FPS - decode thread feeding the GPU, wall clock

Usage:
  python bench_trt.py --parquet finevideo/data/train-00000-of-01357.parquet --engine yolo11l.engine --batch 8 --max-videos 5
"""
import argparse
import io
import queue
import threading
import time

import av
import numpy as np
import pyarrow.parquet as pq
from ultralytics import YOLO

CLASSES = [0, 2, 16, 63, 67]  # person, car, dog, laptop, cell phone (COCO ids)


def iter_videos(path, max_videos):
    pf = pq.ParquetFile(path)
    row = 0
    for batch in pf.iter_batches(batch_size=1, columns=["mp4"]):
        for mp4 in batch.column("mp4").to_pylist():
            if row >= max_videos:
                return
            yield row, mp4
            row += 1


def decode_frames(mp4_bytes):
    with av.open(io.BytesIO(mp4_bytes)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            img = frame.to_ndarray(format="bgr24")
            rot = getattr(frame, "rotation", 0) or 0  # PyAV >= 14 exposes display-matrix rotation
            if rot:
                img = np.ascontiguousarray(np.rot90(img, k=(rot // 90) % 4))
            yield img


def producer(path, max_videos, q, stats):
    t0 = time.perf_counter()
    for row, mp4 in iter_videos(path, max_videos):
        n = 0
        try:
            for img in decode_frames(mp4):
                q.put(img)
                n += 1
        except Exception as e:  # damaged video
            print(f"  row {row}: decode error {e!r}")
        stats["videos"] += 1
        stats["frames"] += n
        print(f"  row {row}: {n} frames decoded")
    stats["decode_wall"] = time.perf_counter() - t0
    q.put(None)


def run_batch(model, frames, batch, imgsz, stats):
    n_real = len(frames)
    if n_real < batch:  # static-batch engine: pad with the last frame
        frames = frames + [frames[-1]] * (batch - n_real)
    results = model.predict(frames, imgsz=imgsz, conf=0.4, classes=CLASSES, verbose=False)
    sp = results[0].speed  # ms per image, averaged over the batch
    stats["pre_ms"] += sp["preprocess"] * batch
    stats["inf_ms"] += sp["inference"] * batch
    stats["post_ms"] += sp["postprocess"] * batch
    stats["infer_frames"] += n_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--engine", default="yolo11l.engine")
    ap.add_argument("--batch", type=int, default=8, help="must match the batch the engine was exported with")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-videos", type=int, default=5)
    args = ap.parse_args()

    model = YOLO(args.engine, task="detect")
    dummy = [np.zeros((720, 1280, 3), np.uint8)] * args.batch
    for _ in range(3):  # warm-up
        model.predict(dummy, imgsz=args.imgsz, verbose=False)

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
            run_batch(model, buf, args.batch, args.imgsz, stats)
            buf = []
    if buf:
        run_batch(model, buf, args.batch, args.imgsz, stats)
    wall = time.perf_counter() - t_start

    n = stats["infer_frames"]
    gpu_s = (stats["pre_ms"] + stats["inf_ms"] + stats["post_ms"]) / 1000
    print("\n===== RESULTS =====")
    print(f"videos: {stats['videos']}   frames: {n}")
    print(f"per-frame ms  preprocess {stats['pre_ms']/n:.2f} | TensorRT {stats['inf_ms']/n:.2f} | NMS {stats['post_ms']/n:.2f}")
    print(f"TensorRT-only FPS        : {n / (stats['inf_ms']/1000):8.1f}")
    print(f"model FPS (pre+trt+nms)  : {n / gpu_s:8.1f}")
    print(f"decode FPS (overlapped)  : {n / stats['decode_wall']:8.1f}")
    print(f"END-TO-END FPS           : {n / wall:8.1f}   ({wall:.1f}s wall)")
    # 125 h of footage at ~30 fps ~= 13.5M frames
    est_frames = 125 * 3600 * 30
    print(f"projected time for 125h @30fps, every frame: {est_frames / (n / wall) / 3600:.1f} h")


if __name__ == "__main__":
    main()
