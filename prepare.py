"""One-time preparation (untimed, has internet).

Downloads the official yolo11l.pt, which is all the default run needs (`--backend pytorch`).
Also tries to build the optional TensorRT FP16 engine for `--backend tensorrt` (engines only work on the GPU and
TensorRT version they were built with, so it is built here and not committed). A failed engine build does not
affect the default run.

    python prepare.py
"""
import argparse
import os
import shutil
import sys

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[32, 16, 8], help="engine batch sizes to try, in order")
    ap.add_argument("--no-engine", action="store_true", help="only download yolo11l.pt")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    YOLO("yolo11l.pt")  # downloads the official weights if missing
    print(f"weights ready: {os.path.join(here, 'yolo11l.pt')}")
    if args.no_engine:
        return

    target = os.path.join(here, "yolo11l.engine")
    for b in args.batch:
        try:
            out = YOLO("yolo11l.pt").export(
                format="engine", imgsz=640, batch=b, quantize=16, dynamic=False, simplify=True, device=0
            )
        except Exception as e:  # e.g. not enough GPU memory to build this batch size
            print(f"optional engine build with batch {b} failed: {e!r}", file=sys.stderr)
            continue
        if os.path.abspath(out) != target:
            shutil.move(out, target)
        print(f"optional TensorRT engine ready: {target} (batch {b})")
        return
    print("optional TensorRT engine could not be built; the default --backend pytorch does not need it", file=sys.stderr)


if __name__ == "__main__":
    main()
