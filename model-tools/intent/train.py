"""BERT 中文意图识别微调脚本。

与 docs/意图识别模型微调与ONNX导出.md 同步演进。

用法：
    # 用配置文件（推荐）
    python train.py --config configs/base.yaml

    # 命令行参数
    python train.py \\
      --model_dir model/chinese-bert-wwm-ext \\
      --train_data data/train.tsv \\
      --val_data data/val.tsv \\
      --labels_file data/labels.json \\
      --output_dir model/intent-model \\
      --epochs 5 --batch_size 16 --lr 2e-5 --max_length 128
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report

# 允许 ``python train.py`` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.progress import fail, info, ok, step, warn


@dataclass
class TrainConfig:
    model_dir: str
    train_data: str
    val_data: str
    labels_file: str
    output_dir: str = "./model/intent-model"
    epochs: int = 5
    batch_size: int = 16
    lr: float = 2e-5
    max_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        train = data.get("train", {})
        return cls(
            model_dir=train["model_dir"],
            train_data=train["train_data"],
            val_data=train["val_data"],
            labels_file=train["labels_file"],
            output_dir=train.get("output_dir", cls.output_dir),
            epochs=train.get("epochs", cls.epochs),
            batch_size=train.get("batch_size", cls.batch_size),
            lr=train.get("lr", cls.lr),
            max_length=train.get("max_length", cls.max_length),
            warmup_ratio=train.get("warmup_ratio", cls.warmup_ratio),
            weight_decay=train.get("weight_decay", cls.weight_decay),
            seed=train.get("seed", cls.seed),
        )


def parse_args() -> tuple[argparse.Namespace, TrainConfig]:
    parser = argparse.ArgumentParser(description="BERT 意图分类微调")
    parser.add_argument("--config", type=str, help="YAML 配置文件路径")
    # 单参数模式（与 --config 互斥）
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--val_data", type=str)
    parser.add_argument("--labels_file", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.config:
        cfg = TrainConfig.from_yaml(args.config)
        # 命令行参数可覆盖配置文件
        for f in (
            "model_dir", "train_data", "val_data", "labels_file",
            "output_dir", "epochs", "batch_size", "lr", "max_length",
            "warmup_ratio", "weight_decay", "seed",
        ):
            v = getattr(args, f, None)
            if v is not None:
                setattr(cfg, f, v)
    else:
        required = ("model_dir", "train_data", "val_data", "labels_file")
        missing = [n for n in required if getattr(args, n) is None]
        if missing:
            parser.error(f"必须提供 --config 或以下参数: {', '.join('--' + m for m in missing)}")
        cfg = TrainConfig(
            model_dir=args.model_dir,
            train_data=args.train_data,
            val_data=args.val_data,
            labels_file=args.labels_file,
            output_dir=args.output_dir or TrainConfig.output_dir,
            epochs=args.epochs or TrainConfig.epochs,
            batch_size=args.batch_size or TrainConfig.batch_size,
            lr=args.lr or TrainConfig.lr,
            max_length=args.max_length or TrainConfig.max_length,
            warmup_ratio=args.warmup_ratio if args.warmup_ratio is not None else TrainConfig.warmup_ratio,
            weight_decay=args.weight_decay if args.weight_decay is not None else TrainConfig.weight_decay,
            seed=args.seed or TrainConfig.seed,
        )
    return args, cfg


def load_labels(labels_file: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """加载标签列表，构建 label2id / id2label 映射。"""
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = json.load(f)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return labels, label2id, id2label


def _read_table(path: str) -> list[tuple[str, str]]:
    """读取 TSV/CSV 文件。"""
    import csv

    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = "text" in sample.splitlines()[0].lower() if sample else False
        delimiter = "\t" if path.endswith((".tsv", ".txt")) else ","
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)
        rows = [(line[0].strip(), line[1].strip()) for line in reader if len(line) >= 2]
    if not rows:
        fail(f"数据文件 {path} 为空")
        raise SystemExit(1)
    return rows


def load_dataset(rows: list[tuple[str, str]], tokenizer, label2id: dict[str, int], max_length: int):
    from datasets import Dataset

    ds = Dataset.from_list(
        [{"text": t, "label": label2id[l]} for t, l in rows]
    )

    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_token_type_ids=True,
        )
        enc["labels"] = batch["label"]
        return enc

    return ds.map(tokenize_fn, batched=True, remove_columns=["text", "label"])


def compute_metrics(eval_pred, id2label: dict[int, str]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        labels, preds, target_names=target_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": report["macro avg"]["f1-score"],
    }


def main() -> None:
    args, cfg = parse_args()

    import torch
    from transformers import (
        BertForSequenceClassification,
        BertTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)

    # 1. 加载标签
    labels, label2id, id2label = load_labels(cfg.labels_file)
    num_labels = len(labels)
    step(f"[1/5] 标签: {labels} (共 {num_labels} 个)")

    # 2. 加载 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained(cfg.model_dir)
    model = BertForSequenceClassification.from_pretrained(
        cfg.model_dir,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    info(f"[2/5] 模型加载完成: {type(model).__name__}")

    # 3. 加载数据集
    train_rows = _read_table(cfg.train_data)
    val_rows = _read_table(cfg.val_data)
    train_dataset = load_dataset(train_rows, tokenizer, label2id, cfg.max_length)
    val_dataset = load_dataset(val_rows, tokenizer, label2id, cfg.max_length)
    info(f"[3/5] 数据集: train={len(train_dataset)}, val={len(val_dataset)}")

    # 4. 训练配置
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: compute_metrics(p, id2label),
        processing_class=tokenizer,
    )

    # 5. 训练
    step("[4/5] 开始训练...")
    train_result = trainer.train()
    info(f"      训练完成: {train_result.metrics}")

    eval_result = trainer.evaluate()
    info(f"      验证结果: {eval_result}")

    # 保存最佳模型
    step(f"[5/5] 保存模型到 {cfg.output_dir}")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # 保存标签文件到模型目录
    with open(os.path.join(cfg.output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    ok("微调完成!")
    info(f"   模型目录: {cfg.output_dir}")
    info(f"   验证准确率: {eval_result.get('eval_accuracy', 'N/A')}")
    info(f"   Macro-F1:   {eval_result.get('eval_macro_f1', 'N/A')}")


if __name__ == "__main__":
    main()
