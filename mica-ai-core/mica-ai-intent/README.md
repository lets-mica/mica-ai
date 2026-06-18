# mica-ai-intent

> BERT 中文意图识别推理的 **Java 17** 实现，基于 [hfl/chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext) 模型。

零 PyTorch / 零 Python 依赖。完整复现 BERT 中文按字分词、[CLS]/[SEP] 特殊标记、Softmax 意图分类等全链路逻辑，兼容 HuggingFace Transformers 的 vocab.txt 词表格式。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Azul Zulu 17 / Temurin 17 / Oracle 17 |
| Maven | 3.6+ | 编译 / 打包 |
| ONNX Runtime | 1.20+ | 通过 Maven 自动拉取 |

---

## 2. 模型准备

### 2.1 训练 / 导出 ONNX 模型

本模块包含推理引擎、分词器和词表加载器，需要配合已训练好的 ONNX 模型使用：

1. 下载 [hfl/chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext) 预训练模型
2. 添加分类头后导出 ONNX：

```bash
python export_bert_intent.py \
  --model_name hfl/chinese-bert-wwm-ext \
  --num_labels 4 \
  --output bert_intent.onnx \
  --max_length 128
```

3. 将 `bert_intent.onnx` 和 `vocab.txt` 放入项目中

### 2.2 模型规格

| 项目 | 规格 |
|------|------|
| 输入 | `input_ids` / `attention_mask` / `token_type_ids` → int64 [1, maxLength] |
| 输出 | `logits` → float32 [1, numLabels] |
| 词表 | HuggingFace BERT vocab.txt（兼容 `chinese-bert-wwm-ext`） |

---

## 3. 快速使用

### 3.1 Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-intent</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 3.2 基本用法

```java

import net.dreamlu.mica.ai.intent.BertIntent;
import net.dreamlu.mica.ai.intent.config.BertIntentConfig;
import net.dreamlu.mica.ai.intent.config.IntentResult;

import java.util.List;

public class Demo {
    public static void main(String[] args) throws Exception {
        // 1. 配置引擎
        BertIntentConfig config = BertIntentConfig.builder()
            .modelPath("bert_intent.onnx")
            .vocabPath("vocab.txt")
            .labels(List.of("weather", "music", "chat", "news"))
            .maxLength(128)           // 可选，默认 128
            .build();

        // 2. 创建引擎并预测
        try (BertIntent intent = new BertIntent(config)) {
            // 单条预测
            IntentResult result = intent.predict("今天天气怎么样");

            System.out.println("意图: " + result.intent());
            System.out.println("置信度: " + result.confidence());

            // 所有意图的分数
            result.allScores().forEach((label, score) ->
                System.out.printf("  %s: %.4f%n", label, score)
            );

            // 批量预测
            List<IntentResult> results = intent.predictBatch(List.of(
                "播放一首音乐",
                "帮我查一下新闻",
                "你好"
            ));
        }
    }
}
```

---

## 4. 包结构

```
net.dreamlu.mica.ai.intent
├── BertIntent.java              # 公开入口：组合分词 + 推理
├── config/                      # 配置和结果类
│   ├── BertIntentConfig.java    #   Builder 风格配置
│   └── IntentResult.java        #   推理结果（record）
├── engine/                      # 推理引擎层
│   └── BertIntentEngine.java    #   ONNX Runtime 推理 + Softmax
└── tokenizer/                   # 分词器层
    ├── BertTokenizer.java       #   BERT 中文按字切分
    └── VocabLoader.java         #   HuggingFace vocab.txt 加载
```

根包为公开 API，`config/`、`engine/` 和 `tokenizer/` 为内部实现层。

---

## 5. 核心组件

| 层 | 类 | 说明 |
|------|-----|------|
| **入口** | `BertIntent` | 组合分词 + 推理，提供 `predict` / `predictBatch` API |
| **配置** | `BertIntentConfig` | Builder 风格，链式设置模型路径、标签、线程数等 |
| **结果** | `IntentResult` | record，包含 intent、confidence、allScores |
| **推理** | `engine.BertIntentEngine` | ONNX Runtime 推理，Softmax 归一化 |
| **分词** | `tokenizer.BertTokenizer` | BERT 中文按字切分，CJK 逐字 + 英文连续 |
| **词表** | `tokenizer.VocabLoader` | HuggingFace vocab.txt 兼容加载 |

### 分词策略

- CJK 字符（中日韩统一表意文字、平假名、片假名、韩文）→ **逐字拆分**
- ASCII 字母和数字 → **连续作为整体**
- 其他字符（标点等）→ **逐字符处理**
- 自动添加 [CLS]=101 和 [SEP]=102，支持截断和 padding

### 结果数据结构

```java
record IntentResult(
    String intent,              // 最佳匹配意图
    float confidence,           // 置信度 [0, 1]
    Map<String, Float> allScores // 所有意图的分数
);
```

---

## 6. 注意事项

- **词表兼容性**：词表加载器与 HuggingFace Transformers 的 `vocab.txt` 完全兼容，可直接复用
- **分词逻辑**：手写 BERT 中文分词，无 Jieba / HanLP 等外部分词依赖
- **线程安全**：`BertIntent` 单实例不支持并发预测，多线程场景需创建多个实例
- **Softmax**：引擎内置 Softmax 归一化，输入 logits、输出概率分布

---

## 7. 致谢

- [hfl/chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext) — 哈工大讯飞联合实验室开源的中文 BERT 预训练模型
- [ONNX Runtime](https://onnxruntime.ai/) — 跨平台高性能推理引擎
