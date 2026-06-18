# BERT 意图识别模型微调与 ONNX 导出指南

本文档介绍如何基于 `hfl/chinese-bert-wwm-ext` 预训练模型，在意图分类数据集上微调，并导出为 mica-ai-intent 模块可用的 ONNX 格式。

---

## 0. 推荐：使用 `model-tools/intent/` 一键完成

本仓库已经提供了完整的 Python 端到端工具链，位于 [`model-tools/intent/`](../model-tools/intent/README.md)。**强烈建议优先使用工具链**，本指南剩余章节可作为概念/流程参考。

```bash
# 安装依赖
pip install -r model-tools/requirements.txt
pip install -r model-tools/intent/requirements.txt

# 端到端：下载 → 微调 → 导出 ONNX
cd model-tools/intent
python download.py                              # 从 ModelScope 下载 chinese-bert-wwm-ext
python train.py --config configs/base.yaml     # 微调（数据见 data/）
python convert.py \
  --model_dir model/intent-model \
  --output_dir model/out                        # 导出 ONNX + 一致性校验
```

工具链提供的能力：

| 能力 | 工具链 | 本文档 |
|------|--------|--------|
| 数据格式 | `data/{train,val}.tsv` + `labels.json`（已附样例） | 第 4 节 |
| 微调脚本 | [`intent/train.py`](../model-tools/intent/train.py)（支持 YAML/CLI 双模式） | 第 5 节 |
| ONNX 导出 | [`intent/convert.py`](../model-tools/intent/convert.py)（带 PyTorch vs ONNX 校验，可选 INT8 量化） | 第 6 节 |
| 集成到 mica-ai | 见 [`intent/README.md`](../model-tools/intent/README.md) | 第 7 节 |

> 本文档后文**保留详细技术原理**，便于需要自定义脚本或排查问题的同学。

---

## 1. 为什么需要微调

`chinese-bert-wwm-ext` 是预训练的 Masked Language Model（MLM），**没有分类头**。直接导出 ONNX 时，分类头会被随机初始化，推理结果无语义意义。

微调的目的是：在 BERT 编码器之上训练一个线性分类头，使模型学会将中文文本映射到预定义的意图类别（如天气、音乐、聊天、新闻）。

```
微调前:  chinese-bert-wwm-ext (MLM)  →  随机分类头  →  无意义输出
微调后:  chinese-bert-wwm-ext (MLM)  →  训练分类头  →  意图分类
```

---

## 2. 整体流程

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  1. 环境准备  │ ──► │  2. 数据准备  │ ──► │  3. 微调训练  │ ──► │  4. 导出 ONNX │
│  Python 环境  │     │  CSV/JSON 数据 │     │  Trainer API  │     │  torch.onnx   │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                      │
                                                                      ▼
                                                              ┌──────────────┐
                                                              │  5. 集成测试  │
                                                              │  mica-ai 验证 │
                                                              └──────────────┘
```

---

## 3. 环境准备

### 3.1 Python 环境

```bash
# 建议 Python 3.10+
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install torch transformers datasets accelerate scikit-learn onnx onnxruntime
```

### 3.2 依赖版本要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| `torch` | 2.0+ | PyTorch 深度学习框架 |
| `transformers` | 4.30+ | Hugging Face 模型库 |
| `datasets` | 2.0+ | 数据加载与预处理 |
| `scikit-learn` | 1.0+ | 评估指标（classification_report） |
| `onnx` | 1.14+ | ONNX 模型验证 |
| `onnxruntime` | 1.15+ | ONNX 推理验证 |

### 3.3 预训练模型

下载 `hfl/chinese-bert-wwm-ext` 模型文件到本地：

```bash
# 方式一：huggingface-cli
pip install huggingface_hub
huggingface-cli download hfl/chinese-bert-wwm-ext --local-dir ./chinese-bert-wwm-ext

# 方式二：ModelScope（国内更快）
pip install modelscope
modelscope download --model AI-ModelScope/chinese-bert-wwm-ext --local_dir ./chinese-bert-wwm-ext
```

下载后目录结构：

```
chinese-bert-wwm-ext/
├── config.json
├── pytorch_model.bin       # 或 model.safetensors
├── vocab.txt               # 词表（21128 个 token）
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

---

## 4. 数据准备

### 4.1 数据格式

采用 CSV 格式，两列：`text`（用户输入）和 `label`（意图标签）。

**`intent_data/train.csv`：**

```csv
text,label
今天天气怎么样,weather
明天会下雨吗,weather
播放一首音乐,music
来首周杰伦的歌,music
你好,chat
你是谁,chat
今天有什么新闻,news
最新科技新闻,news
```

**`intent_data/val.csv`**（验证集，格式相同）：

```csv
text,label
周末天气如何,weather
放一首轻音乐,music
早上好,chat
有什么时事新闻,news
```

### 4.2 标签设计原则

| 原则 | 说明 |
|------|------|
| **互斥性** | 每条文本只属于一个意图，避免重叠 |
| **均衡性** | 各标签的样本数尽量均衡，避免长尾分布 |
| **覆盖度** | 每个标签至少 100+ 条训练样本 |
| **粒度适中** | 不宜过粗（如只有 "other"）也不宜过细（如 "play_piano" vs "play_guitar"） |

### 4.3 标签文件

创建 `intent_data/labels.json`，定义标签顺序（**必须与 ONNX 输出 logits 维度顺序一致**）：

```json
["weather", "music", "chat", "news"]
```

> ⚠️ **标签顺序至关重要**：ONNX 模型输出 `logits[i]` 对应 `labels[i]`，微调和导出时的标签顺序必须完全一致。

### 4.4 数据集来源参考

| 数据集 | 说明 | 获取方式 |
|--------|------|---------|
| [BFSI](https://github.com/sonlamta/bfsi-intent) | 银行金融意图分类 | GitHub |
| [CLINC150](https://github.com/clinc/oos-eval) | 150 个意图，多语言 | GitHub |
| 自建数据集 | 针对业务场景定制 | 人工标注 / LLM 辅助生成 |

自建数据集时，可以用 LLM 辅助生成初始语料，再人工审核：

```
Prompt 示例：
"请为意图 'weather' 生成 50 条多样化的中文用户表达，覆盖晴天、雨天、温度、
穿衣建议等子场景，每条一行，输出 CSV 格式。"
```

---

## 5. 微调训练

### 5.1 完整训练脚本

创建 `finetune_intent.py`：

```python
"""
BERT 意图分类微调脚本。

用法:
  python finetune_intent.py \
    --model_dir ./chinese-bert-wwm-ext \
    --train_data ./intent_data/train.csv \
    --val_data ./intent_data/val.csv \
    --labels_file ./intent_data/labels.json \
    --output_dir ./intent-model \
    --epochs 5 \
    --batch_size 16 \
    --lr 2e-5 \
    --max_length 128
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 意图分类微调")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="预训练模型目录 (chinese-bert-wwm-ext)")
    parser.add_argument("--train_data", type=str, required=True,
                        help="训练集 CSV 文件路径")
    parser.add_argument("--val_data", type=str, required=True,
                        help="验证集 CSV 文件路径")
    parser.add_argument("--labels_file", type=str, required=True,
                        help="标签 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, default="./intent-model",
                        help="微调后模型输出目录")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批量大小")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列长度")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="warmup 比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    return parser.parse_args()


def load_labels(labels_file):
    """加载标签列表，构建 label2id / id2label 映射。"""
    with open(labels_file, "r", encoding="utf-8") as f:
        labels = json.load(f)
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    return labels, label2id, id2label


def load_dataset(csv_path, tokenizer, label2id, max_length):
    """加载 CSV 数据并 tokenize。"""
    ds = Dataset.from_csv(csv_path)

    def tokenize_fn(examples):
        result = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_token_type_ids=True,
        )
        result["labels"] = [label2id[label] for label in examples["label"]]
        return result

    return ds.map(tokenize_fn, batched=True)


def compute_metrics(eval_pred, id2label):
    """计算评估指标。"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    target_names = [id2label[i] for i in range(len(id2label))]
    report = classification_report(
        labels, preds, target_names=target_names, output_dict=True
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": report["macro avg"]["f1-score"],
    }


def main():
    args = parse_args()

    # 1. 加载标签
    labels, label2id, id2label = load_labels(args.labels_file)
    num_labels = len(labels)
    print(f"[1/5] 标签: {labels} (共 {num_labels} 个)")

    # 2. 加载 tokenizer 和模型
    tokenizer = BertTokenizer.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(
        args.model_dir,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    print(f"[2/5] 模型加载完成: {type(model).__name__}")

    # 3. 加载数据集
    train_dataset = load_dataset(args.train_data, tokenizer, label2id, args.max_length)
    val_dataset = load_dataset(args.val_data, tokenizer, label2id, args.max_length)
    print(f"[3/5] 数据集: train={len(train_dataset)}, val={len(val_dataset)}")

    # 4. 训练配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: compute_metrics(p, id2label),
    )

    # 5. 训练
    print("[4/5] 开始训练...")
    train_result = trainer.train()
    print(f"      训练完成: {train_result.metrics}")

    # 评估
    eval_result = trainer.evaluate()
    print(f"      验证结果: {eval_result}")

    # 保存最佳模型
    print(f"[5/5] 保存模型到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存标签文件到模型目录
    with open(os.path.join(args.output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print("\n✅ 微调完成!")
    print(f"   模型目录: {args.output_dir}")
    print(f"   验证准确率: {eval_result.get('eval_accuracy', 'N/A')}")
    print(f"   Macro-F1:   {eval_result.get('eval_macro_f1', 'N/A')}")


if __name__ == "__main__":
    main()
```

### 5.2 运行微调

```bash
python finetune_intent.py \
  --model_dir ./chinese-bert-wwm-ext \
  --train_data ./intent_data/train.csv \
  --val_data ./intent_data/val.csv \
  --labels_file ./intent_data/labels.json \
  --output_dir ./intent-model \
  --epochs 5 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_length 128
```

### 5.3 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 5 | 训练轮数。BERT 微调通常 3-10 轮 |
| `--batch_size` | 16 | 批量大小。显存不足时降到 8 |
| `--lr` | 2e-5 | 学习率。BERT 微调标准范围 1e-5 ~ 5e-5 |
| `--max_length` | 128 | 最大序列长度。短文本意图分类 64-128 足够 |
| `--warmup_ratio` | 0.1 | 前 10% 步数做 warmup，稳定训练初期 |
| `--weight_decay` | 0.01 | L2 正则化，防止过拟合 |

### 5.4 训练输出

训练完成后，`intent-model/` 目录结构：

```
intent-model/
├── config.json              # 模型配置（含 num_labels、label2id）
├── pytorch_model.bin        # 微调后的模型权重
├── vocab.txt                # 词表（与预训练模型相同）
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── labels.json              # 标签顺序（ONNX 导出时使用）
└── training_args.bin        # 训练参数记录
```

---

## 6. 导出 ONNX

### 6.1 导出脚本

创建 `export_onnx.py`：

```python
"""
将微调后的 BERT 意图分类模型导出为 ONNX 格式。

用法:
  python export_onnx.py \
    --model_dir ./intent-model \
    --output_dir ./intent-onnx \
    --max_length 128 \
    --opset_version 14
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
from transformers import BertForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="导出 BERT 意图分类 ONNX 模型")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="微调后的模型目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="ONNX 模型输出目录")
    parser.add_argument("--max_length", type=int, default=128,
                        help="最大序列长度（需与微调时一致）")
    parser.add_argument("--opset_version", type=int, default=14,
                        help="ONNX opset 版本")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 加载微调模型
    print(f"[1/4] 加载微调模型: {args.model_dir}")
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    num_labels = model.config.num_labels
    print(f"      num_labels={num_labels}, max_length={args.max_length}")

    # 2. 构造 dummy 输入
    dummy_input_ids = torch.zeros(1, args.max_length, dtype=torch.long)
    dummy_attention_mask = torch.zeros(1, args.max_length, dtype=torch.long)
    dummy_token_type_ids = torch.zeros(1, args.max_length, dtype=torch.long)

    # 验证 PyTorch 前向传播
    print("[2/4] 验证 PyTorch 前向传播...")
    with torch.no_grad():
        outputs = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            token_type_ids=dummy_token_type_ids,
        )
    assert outputs.logits.shape == (1, num_labels), \
        f"logits shape 应为 (1, {num_labels})，实际为 {outputs.logits.shape}"
    print(f"      logits shape: {outputs.logits.shape} ✓")

    # 3. 导出 ONNX
    onnx_path = os.path.join(args.output_dir, "bert_intent.onnx")
    print(f"[3/4] 导出 ONNX: {onnx_path}")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids),
            onnx_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch"},
                "attention_mask": {0: "batch"},
                "token_type_ids": {0: "batch"},
                "logits": {0: "batch"},
            },
            opset_version=args.opset_version,
            do_constant_folding=True,
        )

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"      ONNX 文件大小: {file_size_mb:.1f} MB")

    # 复制 vocab.txt
    vocab_src = os.path.join(args.model_dir, "vocab.txt")
    vocab_dst = os.path.join(args.output_dir, "vocab.txt")
    shutil.copy2(vocab_src, vocab_dst)
    print(f"      复制 vocab.txt → {vocab_dst}")

    # 复制 labels.json
    labels_src = os.path.join(args.model_dir, "labels.json")
    if os.path.exists(labels_src):
        shutil.copy2(labels_src, os.path.join(args.output_dir, "labels.json"))

    # 4. 验证 ONNX 模型
    print("[4/4] 验证 ONNX 模型...")
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("      ONNX 结构验证通过 ✓")

    # 用 onnxruntime 验证推理一致性
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)

    # PyTorch 推理
    with torch.no_grad():
        pt_logits = model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            token_type_ids=dummy_token_type_ids,
        ).logits.numpy()

    # ONNX 推理
    ort_logits = sess.run(None, {
        "input_ids": dummy_input_ids.numpy(),
        "attention_mask": dummy_attention_mask.numpy(),
        "token_type_ids": dummy_token_type_ids.numpy(),
    })[0]

    max_diff = np.max(np.abs(pt_logits - ort_logits))
    print(f"      PyTorch vs ONNX 最大差异: {max_diff:.6f}")
    assert max_diff < 1e-4, f"ONNX 推理结果与 PyTorch 差异过大: {max_diff}"
    print("      推理一致性验证通过 ✓")

    print("\n✅ ONNX 导出完成!")
    print(f"   模型文件: {onnx_path}")
    print(f"   词表文件: {vocab_dst}")
    print(f"   标签文件: {os.path.join(args.output_dir, 'labels.json')}")


if __name__ == "__main__":
    main()
```

### 6.2 运行导出

```bash
python export_onnx.py \
  --model_dir ./intent-model \
  --output_dir ./intent-onnx \
  --max_length 128
```

### 6.3 导出后目录结构

```
intent-onnx/
├── bert_intent.onnx     # ONNX 模型（~390 MB）
├── vocab.txt            # 词表（21128 tokens）
└── labels.json          # 标签顺序 ["weather", "music", "chat", "news"]
```

### 6.4 ONNX 模型规格

| 项目 | 规格 |
|------|------|
| **输入 `input_ids`** | int64, shape `[batch, 128]` |
| **输入 `attention_mask`** | int64, shape `[batch, 128]` |
| **输入 `token_type_ids`** | int64, shape `[batch, 128]` |
| **输出 `logits`** | float32, shape `[batch, num_labels]` |
| **dynamic_axes** | batch 维度动态，支持批量推理 |
| **opset_version** | 14 |

---

## 7. 集成到 mica-ai

### 7.1 文件放置

将导出的文件放到项目可访问的路径：

```
your-model-dir/
├── bert_intent.onnx
├── vocab.txt
└── labels.json
```

### 7.2 Spring Boot 配置

`application.yml`：

```yaml
mica:
  ai:
    intent:
      model-path: classpath:models/bert_intent.onnx    # 或绝对路径
      vocab-path: classpath:models/vocab.txt
      max-length: 128
      labels: weather, music, chat, news                # 逗号分隔，顺序与 labels.json 一致
      intra-op-num-threads: 1
      inter-op-num-threads: 1
```

### 7.3 纯 Java 使用（无 Spring）

```java
BertIntentConfig config = BertIntentConfig.builder()
    .modelPath("/path/to/bert_intent.onnx")
    .vocabPath("/path/to/vocab.txt")
    .maxLength(128)
    .labels(List.of("weather", "music", "chat", "news"))
    .build();

try (BertIntent intent = new BertIntent(config)) {
    IntentResult result = intent.predict("今天天气怎么样");
    System.out.println("意图: " + result.intent());
    System.out.println("置信度: " + result.confidence());
    System.out.println("全部分数: " + result.allScores());
}
```

> ⚠️ **labels 顺序**：`BertIntentConfig.labels` 的顺序必须与微调时 `labels.json` 的顺序完全一致，否则意图标签会错位。

---

## 8. 验证与评估

### 8.1 运行集成测试

将 ONNX 模型放到测试候选路径后，运行集成测试：

```bash
# 在项目根目录执行
mvn test -pl mica-ai-core/mica-ai-intent -am \
  -Dtest="BertIntentIntegrationTest" \
  -DforkCount=0
```

集成测试验证项：

| 测试 | 说明 |
|------|------|
| 单条中文意图预测 | 返回非空结果，置信度在 [0, 1] |
| allScores 包含所有标签 | 输出 map 的 key 与 labels 列表一致 |
| 概率和为 1 | softmax 验证 |
| confidence = max(allScores) | 置信度等于最大概率 |
| 批量预测 | 返回等长结果列表 |
| 关闭后调用 predict | 抛出 IllegalStateException |
| 重复 close | 不抛异常 |

### 8.2 实际效果验证

微调后的模型应该能正确识别意图：

```java
intent.predict("今天天气怎么样")  →  IntentResult{intent=weather, confidence=0.95+}
intent.predict("播放一首歌")      →  IntentResult{intent=music, confidence=0.90+}
intent.predict("你好")            →  IntentResult{intent=chat, confidence=0.95+}
intent.predict("最新新闻")        →  IntentResult{intent=news, confidence=0.90+}
```

如果准确率不理想，参考下方常见问题。

---

## 9. 常见问题

### Q1：训练 loss 不下降

| 可能原因 | 解决方案 |
|---------|---------|
| 学习率太小 | 尝试 `2e-5` → `3e-5` → `5e-5` |
| 数据量太少 | 每个标签至少 100 条样本 |
| 标签不均衡 | 使用加权采样或 focal loss |
| 数据格式错误 | 检查 CSV 编码（UTF-8）、列名是否为 `text` 和 `label` |

### Q2：训练 loss 下降但验证集准确率不升

| 可能原因 | 解决方案 |
|---------|---------|
| 过拟合 | 减少 epochs（3-5 轮），增加 `weight_decay` |
| 验证集太小 | 增加验证集样本数 |
| 数据分布不一致 | 确保训练集和验证集来自同一分布 |

### Q3：ONNX 推理结果与 PyTorch 不一致

| 可能原因 | 解决方案 |
|---------|---------|
| opset 版本太低 | 使用 opset 14+ |
| 动态轴配置错误 | 检查 `dynamic_axes` 是否正确 |
| 模型未 `eval()` | 导出前确保 `model.eval()` |

### Q4：ONNX 模型太大

BERT-base 模型约 390 MB，优化方案：

| 方案 | 效果 | 说明 |
|------|------|------|
| 量化 (INT8) | ~100 MB | `onnxruntime.quantization.quantize_dynamic` |
| 蒸馏 | ~100 MB | 用小模型蒸馏，需额外训练 |
| 剪枝 | ~200 MB | 移除不重要的注意力头 |

INT8 量化示例：

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="bert_intent.onnx",
    model_output="bert_intent_int8.onnx",
    weight_type=QuantType.QUInt8,
)
# 量化后 ~110 MB，精度损失 < 1%
```

### Q5：中文分词异常

mica-ai-intent 使用自定义的 `BertTokenizer`（按字切分），与 HuggingFace tokenizer 行为一致：

- 中文：逐字切分（你 → 你，好 → 好）
- 英文/数字：连续字符作为整体（hello → hello，123 → 123）
- 特殊 token：`[CLS]=101`，`[SEP]=102`，`[PAD]=0`，`[UNK]=100`

如果 ONNX 推理结果异常，先检查 vocab.txt 是否与微调时使用的一致。

---

## 10. 完整目录结构

本仓库推荐使用 `model-tools/intent/` 工具链，目录结构如下：

```
mica-ai/
└── model-tools/
    └── intent/
        ├── configs/base.yaml           # 训练超参
        ├── data/                       # 训练数据
        │   ├── train.tsv
        │   ├── val.tsv
        │   └── labels.json
        ├── download.py                 # 脚本：下载预训练模型
        ├── train.py                    # 脚本：微调
        ├── convert.py                  # 脚本：导出 ONNX
        └── model/                      # 产物（被 .gitignore 忽略）
            ├── chinese-bert-wwm-ext/   # 预训练权重
            ├── intent-model/           # 微调后权重
            └── out/                    # 最终 ONNX 产物
                ├── bert_intent.onnx
                ├── vocab.txt
                └── labels.json
```

最终只需要 `model/out/` 目录下的三个文件即可集成到 mica-ai-intent。

> 历史上本文档使用过 `ai_test/Intent-ONNX/` 工作目录结构，已被上述结构取代。

---

## 11. 常见问题补充

### Q6：训练时报 `KeyError: 'eval_macro_f1'`

新版本 transformers（≥4.46）会改字段名。`train.py` 已经统一通过 `compute_metrics` 返回字段名，升级 transformers 即可。

### Q7：如何切换到 `chinese-roberta-wwm-ext` / `ernie-3.0` 等其它中文预训练模型？

只需要把 [`model-tools/intent/download.py`](../model-tools/intent/download.py) 顶部的 `MODEL_SCOPE_ID` 改成对应仓库，例如：

```python
MODEL_SCOPE_ID = "iic/nlp_corom_sentence-embedding_chinese-base-ecom"
```

并相应修改 `MODEL` 类为 `RobertaForSequenceClassification` / `ErnieForSequenceClassification`。
其余脚本无需改动。

### Q8：如何接入 LLM 标注的弱监督数据？

把 `data/train.tsv` 替换为 LLM 批量标注结果即可，格式保持两列：

```tsv
播放轻音乐	music
今天气温多少度	weather
```

`labels.json` 里的标签必须覆盖所有出现的标签，否则 `label2id` 转换会 KeyError。
