"""Video indexing: per-second object presence + people count with YOLO11l.

    python run.py --input <folder of .parquet files> --output results.jsonl

Pipeline
  reader thread  : walks the parquet files row by row, hands (video_id, mp4 bytes) to a task queue
  decode workers : (processes) PyAV-decode every frame, keep K (default 8) frames per second chosen by timestamp,
                   rotate if the stream has a rotation tag
  main process   : batches frames by shape and runs Ultralytics `predict` with yolo11l.pt (FP32, conf 0.4,
                   IoU 0.7, class-aware NMS) -- the same preprocessing as the reference -- counts the five classes
                   per frame, and when a video is finished aggregates per second (object present if seen in
                   >= half of the sampled frames, people = median count rounded down) and writes the JSON lines.

--backend tensorrt instead letterboxes to 640x640 in the workers and runs a FP16 TensorRT engine (2x faster, but
its people counts drift from the reference in crowded scenes -- see README).
"""
import argparse
import glob
import io
import json
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections import defaultdict

import numpy as np

CLASSES = [0, 2, 16, 63, 67]
NAMES = ["person", "car", "dog", "laptop", "cell phone"]
IMGSZ = 640
CONF = 0.4
IOU = 0.7
CHUNK = 16  # frames per message from a decode worker


# ----------------------------------------------------------------------------- decode worker
def letterbox(img, new=IMGSZ):
    """Same result as ultralytics.data.augment.LetterBox(new_shape=(640, 640), auto=False)(image=img)."""
    import cv2

    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nw, nh = max(round(w * r), 1), max(round(h * r), 1)
    dw, dh = (new - nw) / 2, (new - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def decode_worker(task_q, out_q, sample_fps, do_letterbox, max_seconds, dec_threads):
    import av
    import cv2

    cv2.setNumThreads(1)
    av.logging.set_level(av.logging.PANIC)
    while True:
        task = task_q.get()
        if task is None:
            out_q.put(("exit",))
            return
        vid, data = task
        frames, secs, times = [], [], []
        n_secs, n_dec = 0, 0
        try:
            if not data:
                raise ValueError("empty mp4 bytes")
            with av.open(io.BytesIO(data)) as container:
                if not container.streams.video:
                    raise ValueError("no video stream")
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                stream.thread_count = dec_threads
                tb = stream.time_base
                rate = float(stream.average_rate or stream.guessed_rate or 30)
                pts0 = t0 = None
                last_slot = -1
                for frame in container.decode(stream):
                    # timestamp relative to the first decoded frame; exact rational arithmetic when pts is known
                    if frame.pts is not None and tb is not None:
                        if pts0 is None:
                            pts0 = frame.pts
                        num, den = (frame.pts - pts0) * tb.numerator, tb.denominator
                        sec = max(num // den, 0)
                        slot = (num * sample_fps) // den if sample_fps else n_dec
                        rel = num / den
                    else:
                        t = frame.time if frame.time is not None else n_dec / rate
                        if t0 is None:
                            t0 = t
                        rel = t - t0
                        sec = max(math.floor(rel), 0)
                        slot = math.floor(rel * sample_fps) if sample_fps else n_dec
                    n_dec += 1
                    if max_seconds and sec >= max_seconds:
                        break
                    n_secs = max(n_secs, sec + 1)
                    if slot <= last_slot:  # not the first frame of a new sampling slot -> skip conversion
                        continue
                    last_slot = slot
                    img = frame.to_ndarray(format="bgr24")
                    rot = getattr(frame, "rotation", 0) or 0
                    if rot:
                        img = np.ascontiguousarray(np.rot90(img, k=(rot // 90) % 4))
                    if do_letterbox:
                        img = letterbox(img)
                    frames.append(img)
                    secs.append(sec)
                    times.append(rel)
                    if len(frames) == CHUNK:
                        out_q.put(("frames", vid, frames, secs, times))
                        frames, secs, times = [], [], []
            if frames:
                out_q.put(("frames", vid, frames, secs, times))
            if n_dec == 0:
                out_q.put(("error", vid, "no frames decoded", 0))
            else:
                out_q.put(("done", vid, n_secs))
        except Exception as e:  # damaged / unusual video: keep complete seconds, add an error line
            if frames:
                out_q.put(("frames", vid, frames, secs, times))
            out_q.put(("error", vid, f"decode failed: {type(e).__name__}: {e}"[:200], max(n_secs - 1, 0)))


# ----------------------------------------------------------------------------- reader
def reader(files, task_q, out_q, n_workers, max_videos, skip=frozenset(), only=None):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # the default (mimalloc) pool kept growing over many 0.5 GB files; the system pool returns memory
    pa.set_memory_pool(pa.system_memory_pool())
    count = 0
    try:
        for path in files:
            name = os.path.basename(path)
            row = 0
            try:
                pf = pq.ParquetFile(path, memory_map=True)
                for batch in pf.iter_batches(batch_size=1, columns=["mp4"], use_threads=False):
                    for mp4 in batch.column("mp4").to_pylist():
                        if max_videos and count >= max_videos:
                            return
                        if f"{name}:{row}" in skip or (only and f"{name}:{row}" not in only):
                            row += 1
                            continue
                        task_q.put((f"{name}:{row}", mp4))
                        row += 1
                        count += 1
            except Exception as e:  # unreadable parquet file / row
                out_q.put(("error", f"{name}:{row}", f"could not read parquet: {type(e).__name__}: {e}"[:200], 0))
    finally:
        for _ in range(n_workers):
            task_q.put(None)


# ----------------------------------------------------------------------------- aggregation
def aggregate(vid, n_secs, per_sec):
    """per_sec: {second: [counts(5,) per sampled frame]} -> list of output dicts."""
    out = []
    for s in range(n_secs):
        rows = per_sec.get(s)
        if not rows:
            out.append({"video": vid, "second": s, "objects": [], "people": 0})
            continue
        c = np.asarray(rows)
        n = len(c)
        present = ((c > 0).sum(0) * 2 >= n)
        objs = [NAMES[i] for i in range(5) if present[i]]
        people = int(math.floor(float(np.median(c[:, 0]))))
        out.append({"video": vid, "second": s, "objects": objs, "people": people})
    return out


# ----------------------------------------------------------------------------- inference backends
class EngineBackend:
    """TensorRT engine through Ultralytics AutoBackend, with our own batching + Ultralytics NMS."""

    def __init__(self, weights):
        import torch
        from ultralytics.nn.autobackend import AutoBackend
        from ultralytics.utils import nms

        self.torch, self.nms = torch, nms
        self.dev = torch.device("cuda:0")
        self.model = AutoBackend(weights, device=self.dev, fp16=True, verbose=False)
        self.model.eval()
        self.batch = engine_batch(weights)
        self.buf = torch.zeros((self.batch, IMGSZ, IMGSZ, 3), dtype=torch.uint8).pin_memory()
        self.cls = torch.tensor(CLASSES, device=self.dev)
        self.end2end = getattr(self.model, "end2end", False)

    def infer(self, frames):
        """frames: list (<= batch) of 640x640x3 BGR uint8 -> np.ndarray (n, 5) counts."""
        torch = self.torch
        n = len(frames)
        for i, f in enumerate(frames):
            self.buf[i].copy_(torch.from_numpy(f))
        if n < self.batch:
            self.buf[n:].zero_()
        with torch.inference_mode():
            x = self.buf.to(self.dev, non_blocking=True).permute(0, 3, 1, 2).contiguous().flip(1)
            x = (x.half() if self.model.fp16 else x.float()).div_(255)
            preds = self.model(x)
            dets = self.nms.non_max_suppression(
                preds, CONF, IOU, CLASSES, False, max_det=300, nc=0, end2end=self.end2end
            )[:n]
            counts = torch.stack([(d[:, 5:6] == self.cls).sum(0) for d in dets])
        return counts.cpu().numpy()


class ReferenceBackend:
    """Plain Ultralytics predict on raw frames: what the graders' reference does."""

    def __init__(self, weights, batch):
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.batch = batch

    def infer(self, frames):
        # Ultralytics only uses the minimal-padding ("rect") letterbox when every image in a batch has the same
        # shape; a mixed batch silently falls back to 640x640. Batch by shape so each frame is treated exactly as
        # when its video is processed on its own.
        out = np.zeros((len(frames), 5), np.int64)
        groups = defaultdict(list)
        for i, f in enumerate(frames):
            groups[f.shape].append(i)
        for idxs in groups.values():
            res = self.model.predict(
                [frames[i] for i in idxs], imgsz=IMGSZ, conf=CONF, iou=IOU, classes=CLASSES, device=0, verbose=False
            )
            for i, r in zip(idxs, res):
                for c in r.boxes.cls.int().tolist():
                    out[i, CLASSES.index(c)] += 1
        return out


def load_complete_videos(paths):
    """Videos whose lines are fully present in earlier outputs. A video's lines are written together, so only the
    last video of a file that was killed mid-write can be incomplete: drop it (and any half-written last line)."""
    done = set()
    for p in paths:
        last = None
        with open(p, encoding="utf-8") as f:
            for ln in f:
                try:
                    vid = json.loads(ln)["video"]
                except (ValueError, KeyError):
                    continue
                if vid != last and last is not None:
                    done.add(last)
                last = vid
    return frozenset(done)


def engine_batch(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(4), "little")
        meta = json.loads(f.read(n).decode("utf-8"))
    return int(meta["batch"])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of .parquet files")
    ap.add_argument("--output", required=True)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--weights", default=None, help="default: yolo11l.engine (yolo11l.pt with --reference)")
    ap.add_argument("--sample-fps", type=int, default=8, help="frames per second to run the model on (0 = every frame)")
    ap.add_argument("--files", nargs="+", default=None, help="only these parquet basenames from --input")
    ap.add_argument("--workers", type=int, default=0, help="decode processes (0 = auto)")
    ap.add_argument("--backend", choices=["pytorch", "tensorrt"], default="pytorch",
                    help="pytorch = Ultralytics predict (FP32, same preprocessing as the reference); tensorrt = FP16 engine")
    ap.add_argument("--reference", action="store_true", help="same as --backend pytorch (use with --sample-fps 0)")
    ap.add_argument("--batch", type=int, default=32, help="batch size for the pytorch backend")
    ap.add_argument("--dump-frames", default=None, help="write per-frame counts (for eval_sampling.py)")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--skip-from", nargs="+", default=None,
                    help="resume: skip videos already complete in these earlier output files")
    ap.add_argument("--only-videos", nargs="+", default=None, help="only these 'file:row' ids (for checks)")
    args = ap.parse_args()
    if args.backend == "pytorch":
        args.reference = True
    if args.weights is None:
        args.weights = os.path.join(here, "yolo11l.pt" if args.reference else "yolo11l.engine")

    t_start = time.perf_counter()
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if args.files:
        files = [f for f in files if os.path.basename(f) in set(args.files)]
    if not files:
        sys.exit(f"no .parquet files in {args.input}")

    ncpu = os.cpu_count() or 4
    n_workers = args.workers or max(2, min(ncpu - 2, 24))
    dec_threads = max(1, min(4, ncpu // n_workers))

    ctx = mp.get_context("spawn")
    task_q = ctx.Queue(maxsize=n_workers * 2)
    out_q = ctx.Queue(maxsize=n_workers * 3)
    workers = [
        ctx.Process(
            target=decode_worker,
            args=(task_q, out_q, args.sample_fps, not args.reference, args.max_seconds, dec_threads),
            daemon=True,
        )
        for _ in range(n_workers)
    ]
    for w in workers:
        w.start()
    skip = load_complete_videos(args.skip_from) if args.skip_from else frozenset()
    if skip:
        print(f"[run] resuming: skipping {len(skip)} videos already complete", file=sys.stderr)
    threading.Thread(
        target=reader,
        args=(files, task_q, out_q, n_workers, args.max_videos, skip, set(args.only_videos) if args.only_videos else None),
        daemon=True,
    ).start()

    if args.reference:
        backend = ReferenceBackend(args.weights, args.batch)
    else:
        if not os.path.exists(args.weights):
            sys.exit(f"{args.weights} not found - run `python prepare.py` first")
        backend = EngineBackend(args.weights)
    B = backend.batch
    print(
        f"[run] {len(files)} files, {n_workers} decode workers x {dec_threads} threads, batch {B}, "
        f"sample_fps {args.sample_fps or 'all'}, backend {'pytorch-reference' if args.reference else 'tensorrt'}",
        file=sys.stderr,
    )

    # prefetch thread: unpickling from the multiprocessing queue overlaps with GPU work
    local_q = queue.Queue(maxsize=n_workers * 3)

    def pump():
        exits = 0
        while exits < n_workers:
            m = out_q.get()
            if m[0] == "exit":
                exits += 1
            else:
                local_q.put(m)
        local_q.put(None)

    threading.Thread(target=pump, daemon=True).start()

    out_f = open(args.output, "w", encoding="utf-8")
    dump_f = open(args.dump_frames, "w", encoding="utf-8") if args.dump_frames else None
    per_video = defaultdict(lambda: defaultdict(list))  # vid -> sec -> [counts]
    dumps = defaultdict(lambda: {"sec": [], "t": [], "counts": []})
    pend_frames, pend_meta = [], []  # frames waiting for a full batch, meta = (vid, sec, t)
    stats = dict(videos=0, errors=0, frames=0, seconds=0)
    last_log = time.perf_counter()

    def run_batch(frames, meta):
        counts = backend.infer(frames)
        for (vid, sec, t), c in zip(meta, counts):
            per_video[vid][sec].append(c)
            if dump_f:
                d = dumps[vid]
                d["sec"].append(sec)
                d["t"].append(t)
                d["counts"].append(c.tolist())
        stats["frames"] += len(frames)

    def flush():
        nonlocal pend_frames, pend_meta
        while pend_frames:
            run_batch(pend_frames[:B], pend_meta[:B])
            pend_frames, pend_meta = pend_frames[B:], pend_meta[B:]

    def finish(vid, n_secs, error=None):
        lines = aggregate(vid, n_secs, per_video.pop(vid, {}))
        for ln in lines:
            out_f.write(json.dumps(ln) + "\n")
        if error:
            out_f.write(json.dumps({"video": vid, "error": error}) + "\n")
            stats["errors"] += 1
        if dump_f:
            d = dumps.pop(vid, {"sec": [], "t": [], "counts": []})
            dump_f.write(json.dumps({"video": vid, "n_secs": n_secs, "error": error, **d}) + "\n")
        stats["videos"] += 1
        stats["seconds"] += len(lines)

    while True:
        m = local_q.get()
        if m is None:
            break
        kind = m[0]
        if kind == "frames":
            _, vid, frames, secs, times = m
            pend_frames.extend(frames)
            pend_meta.extend((vid, s, t) for s, t in zip(secs, times))
            while len(pend_frames) >= B:
                run_batch(pend_frames[:B], pend_meta[:B])
                pend_frames, pend_meta = pend_frames[B:], pend_meta[B:]
        else:
            flush()  # the finished video's last frames may still be pending
            if kind == "done":
                finish(m[1], m[2])
            else:  # error
                finish(m[1], m[3], error=m[2])
        now = time.perf_counter()
        if now - last_log > 30:
            el = now - t_start
            print(
                f"[run] {el/60:6.1f} min  videos {stats['videos']}  errors {stats['errors']}  "
                f"frames {stats['frames']} ({stats['frames']/el:.0f}/s)  seconds {stats['seconds']}",
                file=sys.stderr,
                flush=True,
            )
            last_log = now
    flush()
    out_f.close()
    if dump_f:
        dump_f.close()
    el = time.perf_counter() - t_start
    print(
        f"[run] done in {el:.1f}s: videos {stats['videos']}, errors {stats['errors']}, "
        f"model frames {stats['frames']} ({stats['frames']/el:.0f}/s), output seconds {stats['seconds']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
