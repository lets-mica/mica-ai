# intent 模型工具

> 对应 mica-ai-intent（BERT 中文意图识别）。

## 模型规格

| 项目 | 内容 |
|------|------|
| 任务 | 中文意图分类（Softmax 分类头） |
| 预训练模型 | [`AI-ModelScope/chinese-bert-wwm-ext`](https://www.modelscope.cn/models/AI-ModelScope/chinese-bert-wwm-ext) |
| 原始仓库 | [hfl/chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext) |
| 词表 | 21128 tokens |
| 最大长度 | 128（短文本意图分类足够） |

> 与其它能力不同：**intent 包含完整训练流程**（download → train → convert），
> 因为 mica-ai-intent 的分类头是**业务自定义**的，必须经过微调。

## 快速使用

### 1. 准备数据

数据放在 `data/`，按以下两种格式任选其一：

**TSV**（`data/train.tsv`、`data/val.tsv`）：第一列文本，第二列标签
**CSV**：与 TSV 类似，可有表头

```tsv
今天天气怎么样	weather
明天会下雨吗	weather
播放一首音乐	music
来首周杰伦的音乐	music
你好	chat
今天有什么新闻	news
```

**标签文件** `data/labels.json`：

```json
["weather", "music", "chat", "news"]
```

> ⚠️ 标签顺序至关重要：ONNX 输出的 `logits[i]` 对应 `labels[i]`。

### 2. 下载预训练模型

```bash
pip install -r ../requirements.txt
pip install -r requirements.txt
python download.py
```

### 3. 微调

```bash
# 用默认 config
python train.py --config configs/base.yaml

# 或显式传参
python train.py \
  --model_dir model/chinese-bert-wwm-ext \
  --train_data data/train.tsv \
  --val_data data/val.tsv \
  --labels_file data/labels.json \
  --output_dir model/intent-model \
  --epochs 5 --batch_size 16 --lr 2e-5 --max_length 128
```

### 4. 导出 ONNX

```bash
python convert.py \
  --model_dir model/intent-model \
  --output_dir model/out \
  --max_length 128
```

最终产物：

```
model-tools/intent/model/out/
├── bert_intent.onnx
├── vocab.txt
└── labels.json
```

### 5. 集成到 mica-ai

```yaml
mica:
  ai:
    intent:
      model-path: <abs>/model/out/bert_intent.onnx
      vocab-path: <abs>/model/out/vocab.txt
      max-length: 128
      labels: weather,music,chat,news
```

## 量化（可选）

```bash
python convert.py --model_dir model/intent-model --output_dir model/out --quantize int8
# 产出 bert_intent_int8.onnx，体积 ~110MB，精度损失 < 1%
```
