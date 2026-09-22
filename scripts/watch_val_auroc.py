"""Report per-epoch val/mean_AUROC from a running pretraining job.

Lightning's Rich progress bar truncates logged values in the log file, so read
the CSVLogger output instead.

    python scripts/watch_val_auroc.py            # print what has been logged so far
    python scripts/watch_val_auroc.py --follow   # keep printing new epochs as they land
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time

CSV_GLOB = "/home/jcxu/MELP/logs/melp/csv/*/metrics.csv"
LOG_GLOB = "/home/jcxu/MELP/logs/melp/pretrain_*.log"
FATAL = ("Traceback", "OutOfMemoryError", "CUDA out of memory", "Killed")


def newest(pattern):
    paths = sorted(glob.glob(pattern), key=os.path.getmtime)
    return paths[-1] if paths else None


def epochs_logged(csv_path):
    """(epoch, step, auroc) for every validation epoch written so far."""
    rows = []
    if not csv_path or not os.path.exists(csv_path):
        return rows
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            value = row.get("val/mean_AUROC", "")
            if value not in ("", None):
                rows.append((row.get("epoch", "?"), row.get("step", "?"), float(value)))
    return rows


def training_alive():
    out = subprocess.run(["pgrep", "-f", "envs/melp/bin/python main_pretrain"],
                         capture_output=True, text=True)
    return bool(out.stdout.strip())


def fatal_in_log(log_path, offset):
    """Return (new_offset, first fatal line) for text appended since offset."""
    if not log_path or not os.path.exists(log_path):
        return offset, None
    size = os.path.getsize(log_path)
    if size <= offset:
        return size, None
    with open(log_path, errors="replace") as fh:
        fh.seek(offset)
        chunk = fh.read()
    for line in chunk.splitlines():
        if any(marker in line for marker in FATAL):
            return size, line.strip()[:200]
    return size, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    csv_path, log_path = newest(CSV_GLOB), newest(LOG_GLOB)
    # Whatever is already in the file when we attach is history, not news: a relaunched
    # run writes its own metrics.csv, and until it appears we are looking at the
    # previous run's rows.
    seen = len(epochs_logged(csv_path)) if args.follow else 0
    offset = os.path.getsize(log_path) if (args.follow and log_path) else 0

    if not args.follow:
        rows = epochs_logged(csv_path)
        print(f"csv: {csv_path}")
        for epoch, step, auroc in rows:
            print(f"  epoch {epoch:>3}  step {step:>7}  val/mean_AUROC {auroc:.4f}")
        if not rows:
            print("  (no validation epoch finished yet)")
        return

    while True:
        # Re-glob every pass: when a run is relaunched a newer metrics.csv appears,
        # and latching onto the previous run's file would re-report its old epochs.
        current = newest(CSV_GLOB)
        if current != csv_path:
            if csv_path is not None:
                print(f"switching to newer run: {current}", flush=True)
            csv_path, seen = current, 0
        rows = epochs_logged(csv_path)
        for epoch, step, auroc in rows[seen:]:
            best = max(r[2] for r in rows[:rows.index((epoch, step, auroc)) + 1])
            print(f"epoch {epoch} (step {step}): val/mean_AUROC {auroc:.4f}"
                  f"{'  <- best so far' if auroc >= best else f'  (best {best:.4f})'}",
                  flush=True)
        seen = len(rows)

        offset, fatal = fatal_in_log(log_path, offset)
        if fatal:
            print(f"TRAINING ERROR: {fatal}", flush=True)
            return
        if not training_alive():
            print(f"training process is gone after {seen} validated epoch(s)", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
