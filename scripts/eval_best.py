"""Find a run's best checkpoint by val/mean_AUROC and score it on the test splits.

    python scripts/eval_best.py <run_name> [<run_name> ...]

Prints the per-dataset test AUROC that scripts/zeroshot/test_zeroshot.py produces,
which is the protocol the paper reports, alongside the validation peak the
checkpoint was selected on.
"""
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import watch_ab as w  # noqa: E402

ROOT = "/home/jcxu/MELP"
SETS = ["ptbxl_rhythm", "ptbxl_form", "ptbxl_sub_class", "ptbxl_super_class",
        "icbeb", "chapman"]


def best_checkpoint(run):
    """(path, val_peak, step) for the checkpoint at the run's best validation point."""
    rows = w.epochs(f"{ROOT}/logs/melp/csv/{run}/metrics.csv")
    if not rows:
        return None, None, None
    tag, peak, _ = max(rows, key=lambda r: r[1])
    step = int(tag.split("@")[1])
    ckpts = glob.glob(f"{ROOT}/logs/melp/ckpts/{run}/*.ckpt")
    # Lightning names a checkpoint by global_step, one ahead of the validation step
    for want in (step + 1, step):
        for c in ckpts:
            if re.search(rf"step={want}\.ckpt$", c):
                return c, peak, step
    named = [c for c in ckpts if "last" not in c]
    return (named[0] if named else None), peak, step


def evaluate(ckpt):
    out = subprocess.run(
        ["/home/jcxu/envs/melp/bin/python", "test_zeroshot.py", "--model_name", "melp",
         "--ckpt_path", ckpt, "--batch_size", "128", "--num_workers", "8"],
        cwd=f"{ROOT}/scripts/zeroshot", capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("EVAL_GPU", "0"),
             "HF_HUB_OFFLINE": "1"})
    scores, current = {}, None
    for line in out.stdout.splitlines():
        m = re.search(r"zeroshot classification set is (\w+)", line)
        if m:
            current = m.group(1)
        m = re.match(r"^0\s+([\d.]+)\s", line.strip())
        if m and current:
            scores[current] = float(m.group(1))
            current = None
    return scores


def main():
    for run in sys.argv[1:]:
        ckpt, peak, step = best_checkpoint(run)
        if not ckpt:
            print(f"{run}: no checkpoint found")
            continue
        print(f"\n=== {run} ===")
        print(f"val peak {peak:.4f} @ step {step}   ckpt {os.path.basename(ckpt)}")
        s = evaluate(ckpt)
        if not s:
            print("  evaluation produced no scores")
            continue
        for k in SETS:
            if k in s:
                print(f"  {k:20s} {s[k]:.2f}")
        print(f"  {'AVERAGE':20s} {sum(s[k] for k in SETS if k in s) / len(s):.2f}")


if __name__ == "__main__":
    main()
