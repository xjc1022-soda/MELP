"""Report per-epoch val/mean_AUROC for every pretraining run currently going.

Each run is labelled by the hparams CSVLogger writes next to its metrics, so
parallel A/B arms stay distinguishable.

    python scripts/watch_ab.py [--follow]
"""
import argparse, csv, glob, os, subprocess, time

CSV_GLOB = "/home/jcxu/MELP/logs/melp/csv/*/metrics.csv"
KEYS = ["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm",
        "icbeb", "chapman"]


def label(csv_path):
    hp = os.path.join(os.path.dirname(csv_path), "hparams.yaml")
    text_encoder = lr = None
    if os.path.exists(hp):
        for line in open(hp):
            if line.startswith("text_encoder_name:"):
                text_encoder = line.split(":", 1)[1].strip().split("/")[-1]
            elif line.startswith("lr:"):
                lr = line.split(":", 1)[1].strip()
    return f"{os.path.basename(os.path.dirname(csv_path))}:{text_encoder or '?'}"


# CSVLogger rewrites its header when new metric keys appear, so a file can hold
# 14-column headers next to 18-column rows and any header-based mapping silently
# reads training losses as AUROCs. The val/* columns always sort last and always
# arrive together, so take the trailing seven fields positionally instead.
VAL_ORDER = ["chapman", "icbeb", "mean", "ptbxl_form", "ptbxl_rhythm",
             "ptbxl_sub_class", "ptbxl_super_class"]


def epochs(csv_path):
    out = []
    try:
        rows = list(csv.reader(open(csv_path, newline="")))
    except OSError:
        return out
    for row in rows[1:]:
        tail = row[-len(VAL_ORDER):]
        if len(row) <= len(VAL_ORDER) or any(v == "" for v in tail):
            continue
        try:
            vals = [float(v) for v in tail]
        except ValueError:
            continue
        if not all(0.0 <= v <= 1.0 for v in vals):
            continue
        per = dict(zip(VAL_ORDER, vals))
        step = row[2] if len(row) > 2 and row[2] else "?"
        out.append((f"{row[0] or '?'}@{step}", per.pop("mean"),
                    {k: per[k] for k in KEYS if k in per}))
    return out


def alive(run_name=None):
    """Liveness of one named run, or of any run when no name is given.

    Per-arm matters: when one arm of an A/B dies the other keeps the global check
    true, and a dead arm would otherwise just go quiet and look like a slow one.
    """
    pattern = f"run_name {run_name}" if run_name else "envs/melp/bin/python main_pretrain"
    return bool(subprocess.run(["pgrep", "-f", pattern],
                               capture_output=True, text=True).stdout.strip())


def recent(n=2, runs=None):
    """The runs to watch: explicit names if given, else the n most recent."""
    if runs:
        return [f"/home/jcxu/MELP/logs/melp/csv/{r}/metrics.csv" for r in runs]
    named = sorted(glob.glob("/home/jcxu/MELP/logs/melp/csv/ab_*/metrics.csv"))
    return named or sorted(glob.glob(CSV_GLOB), key=os.path.getmtime)[-n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--arms", type=int, default=2)
    ap.add_argument("--only-notable", action="store_true",
                    help="emit only a new best, a drop below --floor, or a dead run; "
                         "plateau points stay silent")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="emit when mean falls below this")
    ap.add_argument("--runs", type=str, default=None,
                    help="comma-separated run names to watch (default: newest)")
    args = ap.parse_args()
    runs = args.runs.split(",") if args.runs else None

    if not args.follow:
        for path in recent(args.arms, runs):
            print(f"{label(path)}  [{os.path.basename(os.path.dirname(path))}]")
            for ep, mean, per in epochs(path):
                print(f"  ep {ep:>12}  mean {mean:.4f}   " +
                      "  ".join(f"{k.replace('ptbxl_', '')} {v:.3f}" for k, v in per.items()))
        return

    seen = {p: len(epochs(p)) for p in recent(args.arms, runs)}
    best = {p: max((m for _, m, _ in epochs(p)), default=0.0) for p in recent(args.arms, runs)}
    dead = set()
    while True:
        for path in recent(args.arms, runs):
            rows = epochs(path)
            for ep, mean, per in rows[seen.get(path, 0):]:
                tag = ""
                if mean > best.get(path, 0.0):
                    tag, best[path] = "  <- NEW BEST", mean
                elif args.only_notable and mean >= args.floor:
                    continue          # plateau: nothing worth waking anyone for
                elif mean < args.floor:
                    tag = f"  <- below floor {args.floor}"
                print(f"[{label(path)}] ep{ep}: mean {mean:.4f}   " +
                      "  ".join(f"{k.replace('ptbxl_', '')} {v:.3f}" for k, v in per.items())
                      + tag, flush=True)
            seen[path] = len(rows)
        for path in recent(args.arms, runs):
            name = os.path.basename(os.path.dirname(path))
            if not alive(name) and name not in dead:
                dead.add(name)
                print(f"ARM DOWN: {name} has no process left "
                      f"(after {len(epochs(path))} validated epoch(s))", flush=True)
        if not alive():
            print("no pretraining process left", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
