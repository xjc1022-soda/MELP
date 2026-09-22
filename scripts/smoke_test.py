"""End-to-end smoke test: every code path the paper's experiments depend on, on
a handful of records, in a few minutes, CPU or GPU.

    python scripts/smoke_test.py                    # everything that needs no checkpoint
    MELP_ECGFM_PATH=/path/to/ecgfm.ckpt python scripts/smoke_test.py

The model is randomly initialised unless you pass weights, so the metrics printed
here are chance level by construction -- what is being checked is that the pipeline
runs end to end and the shapes line up, not that the numbers are good.
"""
import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

from melp.paths import RAW_DATA_PATH, SPLIT_DIR, ROOT_PATH
from melp.datasets.finetune_datamodule import ECGDataModule
from melp.datasets.finetune_dataset import ECGDataset
from melp.datasets.pretrain_datamodule import ECGTextDataModule
from melp.models.melp_model import MELPModel
from melp.models.ssl_finetuner import SSLFineTuner

EVAL_SETS = ["ptbxl_super_class", "ptbxl_sub_class", "ptbxl_form", "ptbxl_rhythm",
             "icbeb", "chapman"]


def banner(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def build_model(device):
    return MELPModel(
        ecg_encoder_name="ecgfm",
        ecg_encoder_weight=os.environ.get("MELP_ECGFM_PATH", ""),
        text_encoder_name="ncbi/MedCPT-Query-Encoder",
    ).to(device)


def check_datasets():
    banner("1. evaluation datasets")
    ok = True
    for name in EVAL_SETS:
        try:
            dm = ECGDataModule(dataset_dir=str(RAW_DATA_PATH), dataset_name=name,
                               batch_size=4, num_workers=2, train_data_pct=1)
            ds = dm.test_dataloader().dataset
            batch = next(iter(dm.test_dataloader()))
            print(f"{name:20s} n={len(ds):6d}  classes={ds.num_classes:3d}  "
                  f"ecg={tuple(batch['ecg'].shape)}  label={tuple(batch['label'].shape)}")
        except Exception as exc:  # noqa: BLE001 - a missing dataset is a normal outcome here
            ok = False
            print(f"{name:20s} FAILED: {type(exc).__name__}: {exc}")
    return ok


def check_model(device):
    banner("2. model: forward, three-level loss, backward")
    model = build_model(device)
    print(f"MELPModel built, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    ecg = torch.rand(2, 12, 5000, device=device)
    report = ["sinus rhythm. normal ecg.",
              "atrial fibrillation with rapid ventricular response. st depression."]

    model.eval()
    with torch.no_grad():
        emb = model.encode_ecg(ecg, normalize=True, proj_contrast=True)
        feat = model.ext_ecg_emb(ecg)
        tok = model._tokenize(report)
        txt = model.get_text_emb(tok.input_ids.to(device), tok.attention_mask.to(device))
    print(f"encode_ecg   {tuple(emb.shape)}  (zero-shot / validation path)")
    print(f"ext_ecg_emb  {tuple(feat.shape)}  (linear-probe path)")
    print(f"get_text_emb {tuple(txt.shape)}  (text tower)")

    model.train()
    losses, _ = model.shared_step({"ecg": ecg, "report": report}, batch_idx=1)
    print("loss: " + "  ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    losses["loss"].backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"backward OK, {n_grad} tensors received gradients")
    return True


def check_zeroshot(device, pct):
    banner("3. zero-shot classification (scripts/zeroshot code path)")
    sys.path.insert(0, str(ROOT_PATH / "scripts/zeroshot"))
    import test_zeroshot as tz

    tz.device = device
    csv = pd.read_csv(SPLIT_DIR / "ptbxl/super_class/ptbxl_super_class_test.csv")
    ds = ECGDataset(data_path=str(RAW_DATA_PATH / "ptbxl"), csv_file=csv, split="test",
                    dataset_name="ptbxl_super_class", data_pct=pct)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4)
    print(f"ptbxl_super_class: {len(ds)} records ({pct:.0%} of test), classes={ds.labels_name}")

    model = build_model(device).eval()
    out_dir = ROOT_PATH / "logs/melp/results/smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    tz.zeroshot_eval(model, loader, str(out_dir), compute_ci=False, save_results=False)
    return True


def check_linear_probe(device):
    banner("4. linear probing (SSLFineTuner code path)")
    from lightning import Trainer

    dm = ECGDataModule(dataset_dir=str(RAW_DATA_PATH), dataset_name="ptbxl_super_class",
                       batch_size=8, num_workers=4, train_data_pct=0.01)
    num_classes = dm.train_dataloader().dataset.num_classes
    model = SSLFineTuner(backbone=build_model(device), in_features=256,
                         num_classes=num_classes, epochs=1)
    trainer = Trainer(accelerator="gpu" if device.type == "cuda" else "cpu", devices=1,
                      fast_dev_run=True, logger=False, enable_checkpointing=False)
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
    return True


def check_pretrain(device):
    """MIMIC-IV-ECG is a separate (large) download -- skipped, not failed, when absent."""
    banner("5. pretraining data + one step on real ECG-report pairs")
    if not (SPLIT_DIR / "mimic-iv-ecg/val.csv").exists():
        print("skipped: no mimic-iv-ecg splits under SPLIT_DIR")
        return None

    for source in ("processed", "raw"):
        try:
            dm = ECGTextDataModule(dataset_dir=str(RAW_DATA_PATH), dataset_list=["mimic-iv-ecg"],
                                   val_dataset_list=None, batch_size=4, num_workers=2,
                                   train_data_pct=1.0, use_rlm=True, ecg_source=source)
            batch = next(iter(dm.val_dataloader()))
            break
        except FileNotFoundError as exc:
            print(f"ecg_source={source!r} unavailable: {str(exc).splitlines()[0][:110]}")
            batch = None
    if batch is None:
        print("skipped: no MIMIC-IV-ECG waveforms in either source")
        return None

    print(f"ecg_source={source!r}  ecg={tuple(batch['ecg'].shape)}  "
          f"reports={len(batch['report'])}")
    print(f"report[0]: {batch['report'][0][:80]}")
    model = build_model(device)
    model.train()
    losses, _ = model.shared_step({k: (v.to(device) if torch.is_tensor(v) else v)
                                   for k, v in batch.items()}, batch_idx=1)
    print("loss: " + "  ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
    losses["loss"].backward()
    print("backward OK")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zeroshot_pct", type=float, default=0.08,
                        help="fraction of the PTB-XL test split to run zero-shot on")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["datasets", "model", "zeroshot", "probe", "pretrain"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  raw_data={RAW_DATA_PATH}  "
          f"ecgfm_weights={os.environ.get('MELP_ECGFM_PATH', '') or '<none, random init>'}")

    checks = [("datasets", check_datasets, ()),
              ("model", check_model, (device,)),
              ("zeroshot", check_zeroshot, (device, args.zeroshot_pct)),
              ("probe", check_linear_probe, (device,)),
              ("pretrain", check_pretrain, (device,))]
    results = {}
    for name, fn, fn_args in checks:
        if name in args.skip:
            continue
        try:
            results[name] = fn(*fn_args)
        except Exception as exc:  # noqa: BLE001 - report which stage broke, keep going
            results[name] = False
            print(f"FAILED: {type(exc).__name__}: {exc}")

    banner("summary")
    for name, ok in results.items():
        print(f"{name:12s} {'skipped' if ok is None else 'ok' if ok else 'FAILED'}")
    sys.exit(0 if all(ok is not False for ok in results.values()) else 1)


if __name__ == "__main__":
    main()
