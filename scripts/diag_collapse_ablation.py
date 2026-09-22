"""Why does the contrastive branch collapse? One arm per hypothesis, ~300 steps.

    python scripts/diag_collapse_ablation.py --arm as-is
        as-is      current code and config (control)
        zero-proj  text_decoder.text_projection zeroed, i.e. the behaviour before
                   MultimodalTransformer.init_parameters() was wired up
        low-lr     lr 5e-5 instead of 2e-4
        no-caption caption_loss_weight 0 (contrastive + UOT only)

Reports cma_loss (chance = ln(batch)) and how much the ECG/text embeddings still
vary across the batch. Collapse = cma pinned at chance and std -> 0.
"""
import argparse, math, os, time
import torch, torch.nn.functional as F
from melp.paths import RAW_DATA_PATH
from melp.datasets.pretrain_datamodule import ECGTextDataModule
from melp.models.melp_model import MELPModel

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True,
               choices=["as-is", "zero-proj", "low-lr", "no-caption", "warmup", "lr-1e-4"])
p.add_argument("--steps", type=int, default=300)
p.add_argument("--batch_size", type=int, default=64)
args = p.parse_args()

dev = torch.device("cuda")
torch.manual_seed(42)

lr = {"low-lr": 5e-5, "lr-1e-4": 1e-4}.get(args.arm, 2e-4)
WARMUP = 100 if args.arm == "warmup" else 0
caption_w = 0.0 if args.arm == "no-caption" else 2.0

model = MELPModel(
    ecg_encoder_name="ecgfm",
    ecg_encoder_weight="/disk1/fywang/ECG/ckpts/ecgfm/epoch=44-step=131040.ckpt",
    text_encoder_name="fuyingw/heart_bert",
    lr=lr, clip_loss_weight=1.0, caption_loss_weight=caption_w, local_loss_weight=0.2,
).to(dev)
if args.arm == "zero-proj":
    with torch.no_grad():
        model.text_decoder.text_projection.zero_()
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.2, betas=(0.9, 0.95))
# The shipped scheduler is CosineAnnealingWarmRestarts (no warmup); the commented-out
# CosineAnnealingWarmupRestarts ramps in. This arm reproduces the ramp.
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / WARMUP)) if WARMUP else None

dm = ECGTextDataModule(dataset_dir=str(RAW_DATA_PATH), dataset_list=["mimic-iv-ecg"],
                       val_dataset_list=None, batch_size=args.batch_size, num_workers=4,
                       train_data_pct=1.0, use_rlm=True, ecg_source="processed")
loader = dm.train_dataloader()
probe = next(iter(dm.val_dataloader()))
probe = {"ecg": probe["ecg"][:8].to(dev), "report": probe["report"][:8]}


@torch.no_grad()
def spread():
    model.eval()
    e = model.encode_ecg(probe["ecg"], normalize=True, proj_contrast=True)
    tok = model._tokenize(probe["report"])
    t = model.encode_text(tok.input_ids.to(dev), tok.attention_mask.to(dev))
    model.train()
    off = lambda x: (F.normalize(x, dim=-1) @ F.normalize(x, dim=-1).T)[
        ~torch.eye(len(x), dtype=bool, device=x.device)]
    return e.std(0).mean().item(), off(e).mean().item(), t.std(0).mean().item(), off(t).mean().item()


print(f"arm={args.arm}  lr={lr}  warmup={WARMUP}  caption_w={caption_w}  chance(cma)={math.log(args.batch_size):.4f}",
      flush=True)
print(f"{'step':>5} {'cma':>8} {'caption':>8} {'uot':>7} | {'ecg_std':>9} {'ecg_cos':>8} "
      f"{'txt_std':>9} {'txt_cos':>8}", flush=True)
t0 = time.time()
step = 0
for batch in loader:
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        losses, _ = model.shared_step(batch, batch_idx=step + 1)
    losses["loss"].backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    if sched is not None:
        sched.step()
    if step % 25 == 0 or step == args.steps - 1:
        es, ec, ts, tc = spread()
        print(f"{step:>5} {losses['cma_loss'].item():8.4f} {losses['caption_loss'].item():8.4f} "
              f"{losses['uot_loss'].item():7.4f} | {es:9.6f} {ec:8.4f} {ts:9.6f} {tc:8.4f}",
              flush=True)
    step += 1
    if step >= args.steps:
        break
print(f"done in {time.time() - t0:.0f}s", flush=True)
