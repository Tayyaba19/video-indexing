"""Simulate sampling rates from a TensorRT every-frame dump and score each against a reference results file."""
import random
import sys

import numpy as np

sys.path.insert(0, ".")
from eval_sampling import load_dump, load_results, per_second, score, verdict  # noqa: E402

ref = load_results(sys.argv[1])
dump = load_dump(sys.argv[2])
vids = sorted({k[0] for k in ref})
print(f"{len(vids)} videos, {len(ref)} reference seconds")
print(f"{'rate':>5} {'F1':>7} {'exact':>7} {'within1':>7}  verdict  | bootstrap P(fail any)  crowded video exact")
for rate in [3, 4, 5, 6, 8, 10, 15, 0]:
    pred = {(v, s): x for v in vids for s, x in per_second(dump[v], rate).items()}
    f1, ex, w1, n, _ = score(ref, pred)
    crowded = score({k: x for k, x in ref.items() if k[0] == "train-00013-of-01357.parquet:13"}, pred)[1]
    rng = random.Random(0)
    fails = 0
    for _ in range(1000):
        bs = [rng.choice(vids) for _ in vids]
        r2 = {(i, k[1]): ref[k] for i, v in enumerate(bs) for k in ref if k[0] == v}
        p2 = {(i, k[1]): pred.get(k, (frozenset(), 0)) for i, v in enumerate(bs) for k in ref if k[0] == v}
        a, b, c, _, _ = score(r2, p2)
        fails += a < 0.95 or b < 0.85 or c < 0.95
    print(f"{rate or 'all':>5} {f1:7.4f} {ex:7.4f} {w1:7.4f}  {verdict(f1, ex, w1)}     | {fails/1000:.3f}                 {crowded:.3f}")
