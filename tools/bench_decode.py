"""Decode-only throughput of run.py's workers (no model): video-seconds per wall second."""
import multiprocessing as mp
import sys
import time

sys.path.insert(0, ".")
import pyarrow.parquet as pq  # noqa: E402

from run import decode_worker  # noqa: E402

if __name__ == "__main__":
    path, rows, n_workers, fps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    ctx = mp.get_context("spawn")
    tq, oq = ctx.Queue(maxsize=n_workers * 2), ctx.Queue(maxsize=n_workers * 6)
    ws = [ctx.Process(target=decode_worker, args=(tq, oq, fps, True, 0, 1), daemon=True) for _ in range(n_workers)]
    for w in ws:
        w.start()
    pf = pq.ParquetFile(path)
    data = [m for b in pf.iter_batches(batch_size=1, columns=["mp4"]) for m in b.column("mp4").to_pylist()][12:12 + rows]
    time.sleep(3)  # let workers import
    t0 = time.perf_counter()
    import threading

    def feed():
        for i, d in enumerate(data):
            tq.put((str(i), d))
        for _ in ws:
            tq.put(None)

    threading.Thread(target=feed, daemon=True).start()
    secs = frames = exits = 0
    while exits < n_workers:
        m = oq.get()
        if m[0] == "exit":
            exits += 1
        elif m[0] == "frames":
            frames += len(m[2])
        elif m[0] == "done":
            secs += m[2]
    el = time.perf_counter() - t0
    print(f"workers {n_workers} fps {fps}: {secs} video-s in {el:.1f}s = {secs/el:.1f} video-s/s, sampled frames {frames/el:.0f}/s"
          f" -> 125h decode-bound time {450000/(secs/el)/3600:.2f} h")
