"""Correctness check against the grading rules in section 5 of the brief.

Mode 1 - simulate sampling rates from per-frame dumps (run.py --dump-frames, all frames):
    python eval_sampling.py --ref ref_frames.jsonl --cand trt_frames.jsonl --rates 1 2 3 4 5 6 8 10 0

Mode 2 - compare two results.jsonl files (what the graders do):
    python eval_sampling.py --ref-results ref_results.jsonl --results results.jsonl
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

NAMES = ["person", "car", "dog", "laptop", "cell phone"]


def agg_second(counts):
    c = np.asarray(counts)
    n = len(c)
    present = (c > 0).sum(0) * 2 >= n
    return frozenset(NAMES[i] for i in range(5) if present[i]), int(math.floor(float(np.median(c[:, 0]))))


def per_second(rec, rate):
    """rec: dump line. rate 0 = every frame, else first frame of each 1/rate slot (as run.py does)."""
    by_sec = defaultdict(list)
    last_slot = -1
    for i, (s, t, c) in enumerate(zip(rec["sec"], rec["t"], rec["counts"])):
        if rate:
            slot = math.floor(t * rate + 1e-9)
            if slot <= last_slot:
                continue
            last_slot = slot
        by_sec[s].append(c)
    out = {}
    for s in range(rec["n_secs"]):
        out[s] = agg_second(by_sec[s]) if by_sec.get(s) else (frozenset(), 0)
    return out


def score(ref, pred):
    """ref/pred: {(video, sec): (objects, people)}"""
    tp = fp = fn = 0
    exact = within1 = n = 0
    for k, (ro, rp) in ref.items():
        po, pp = pred.get(k, (frozenset(), 0))
        tp += len(ro & po)
        fp += len(po - ro)
        fn += len(ro - po)
        n += 1
        exact += rp == pp
        within1 += abs(rp - pp) <= 1
    extra = len(set(pred) - set(ref))
    f1 = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 1.0
    return f1, exact / n, within1 / n, n, extra


def verdict(f1, ex, w1):
    return "PASS" if f1 >= 0.95 and ex >= 0.85 and w1 >= 0.95 else "FAIL"


def load_dump(path):
    return {r["video"]: r for r in map(json.loads, open(path, encoding="utf-8")) if r.get("sec") is not None}


def load_results(path):
    d = {}
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        if "error" in r:
            continue
        d[(r["video"], r["second"])] = (frozenset(r["objects"]), r["people"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref")
    ap.add_argument("--cand")
    ap.add_argument("--rates", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 0])
    ap.add_argument("--ref-results")
    ap.add_argument("--results")
    args = ap.parse_args()

    if args.ref_results:
        f1, ex, w1, n, extra = score(load_results(args.ref_results), load_results(args.results))
        print(f"seconds {n}  extra {extra}  objF1 {f1:.4f}  people exact {ex:.4f}  within1 {w1:.4f}  {verdict(f1, ex, w1)}")
        return

    ref_d, cand_d = load_dump(args.ref), load_dump(args.cand)
    vids = sorted(set(ref_d) & set(cand_d))
    ref = {(v, s): x for v in vids for s, x in per_second(ref_d[v], 0).items()}
    print(f"videos {len(vids)}  reference seconds {len(ref)}  (thresholds: F1>=0.95, exact>=0.85, within1>=0.95)")
    print(f"{'rate':>6} {'objF1':>8} {'exact':>8} {'within1':>8}  verdict   per-video worst exact")
    for rate in args.rates:
        pred = {(v, s): x for v in vids for s, x in per_second(cand_d[v], rate).items()}
        f1, ex, w1, n, _ = score(ref, pred)
        worst = min(
            (score({k: x for k, x in ref.items() if k[0] == v}, pred)[1], v) for v in vids
        )
        print(f"{rate or 'all':>6} {f1:8.4f} {ex:8.4f} {w1:8.4f}  {verdict(f1, ex, w1)}    {worst[0]:.3f} ({worst[1]})")


if __name__ == "__main__":
    main()
