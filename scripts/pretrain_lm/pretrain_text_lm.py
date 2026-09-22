"""Stage 1: the cardiology language model.

The paper (Appendix A.2) starts from MedCPT's query encoder and continues training
it with a masked-language-modelling objective on cardiology text; the corpus is
published at fuyingw/medcpt-cardiology-chunked. No hyperparameters are given for
this stage, so the masking ratio and schedule below are BERT defaults.

    python scripts/pretrain_lm/pretrain_text_lm.py --output_dir <dir>
"""
import argparse
import glob
import os

from datasets import load_dataset
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer, TrainingArguments)

CORPUS = "/disk1/jcxu/ECG/medcpt-cardiology-chunked/data"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", default="ncbi/MedCPT-Query-Encoder")
    p.add_argument("--corpus_dir", default=CORPUS)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=32, help="per device")
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="truncate the corpus, for smoke tests")
    p.add_argument("--max_eval_samples", type=int, default=5000,
                   help="the val split has 567k rows; scoring all of it every eval "
                        "would cost more than the training between evals")
    p.add_argument("--num_proc", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    files = {"train": sorted(glob.glob(f"{args.corpus_dir}/train-*.parquet")),
             "validation": sorted(glob.glob(f"{args.corpus_dir}/val-*.parquet"))}
    if not files["train"]:
        raise SystemExit(f"no parquet shards under {args.corpus_dir}")
    ds = load_dataset("parquet", data_files=files)
    if args.max_train_samples:
        ds["train"] = ds["train"].select(range(args.max_train_samples))
    ds["validation"] = ds["validation"].select(
        range(min(args.max_eval_samples, len(ds["validation"]))))
    print({k: len(v) for k, v in ds.items()})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="linear",
            bf16=True,
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=2000,
            save_strategy="steps",
            save_steps=2000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=8,
            seed=args.seed,
            report_to=[],
            ddp_find_unused_parameters=False,
    )

    with training_args.main_process_first(desc="tokenizing"):
        ds = ds.map(tokenize, batched=True, num_proc=args.num_proc,
                    remove_columns=["text"], desc="tokenizing")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("saved to", args.output_dir)
    print(trainer.evaluate())


if __name__ == "__main__":
    main()
