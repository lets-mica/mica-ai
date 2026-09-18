# layout 模型目录

[PP-DocLayoutV3](https://github.com/PaddlePaddle/PaddleOCR)（Apache-2.0，可商用）文档版面分析模型。PaddleX `PP-DocLayoutV3_infer` 经 paddle2onnx 2.1.0 转换 + constant folding 得到 125MB `model.onnx`，**不随仓库分发**（见下方「分发状态」）。

## 模型清单（`models/`）

| 文件 | 大小 | 说明 |
|------|------|------|
| `model.onnx` | 125 MB | PP-DocLayoutV3 版面检测（25 类 RT-DETR + 指针网络阅读顺序），3 输入 3 输出 |

- 来源：<https://github.com/PaddlePaddle/PaddleOCR>（官方 `pp_layout` / `PaddleDetection`）
- License：Apache License 2.0，**可商用** ✅
- 转换误差：V3 相对 Paddle 原版 **1.57%**（V2 为 14.8%，不可生产）

### 分发状态（已决策：暂不入库）

`model.onnx` **125 MB**（130,502,049 bytes），超出 AGENTS.md §6.2 的「单文件 <50MB」约定，
**不随仓库分发**，已在根 `.gitignore` 显式排除（`model-tools/layout/models/model.onnx`）以防误提交。

**为什么不是 Git LFS**（2026-09-18 核实平台配额后否决）：

| 平台 | 单文件限制 | 125MB 的结果 |
|------|-----------|-------------|
| GitHub | >100 MB **硬阻断整个 push**（>50 MB 仅警告） | ❌ 直接 reject |
| Gitee 社区版（个人） | 单文件 ≤50 MB；单仓库 500MB | ❌ 直接 reject |
| Gitee LFS | 免费版**无 LFS 配额**（企业版标准版起才有 1GB） | ❌ 三远端方案在 Gitee 断裂 |
| GitCode | 新建仓库默认 10 MB，需在仓库设置放宽至 100 MB | ⚠️ 需改配置 |

即：`git push` 上不去、LFS 也上不去。GitHub 官方对大文件的建议做法是走 Release 资产，
本项目暂不采用（引入外部托管依赖），先维持「不入库 + 自行获取」。

**本地获取方式**（按推荐顺序）：

1. 从项目协作者处取 `model.onnx`（当前唯一副本在开发机上，见下方 ⚠️）
2. 自行转换（Linux / WSL，Windows 的 paddle2onnx wheel 缺 `common.dll`）：
   - 从 ModelScope / AIStudio 下载 PaddleX `PP-DocLayoutV3_infer`（`model.pdmodel` + `model.pdiparams`）
   - `paddle2onnx` 2.1.0 转换 + constant folding → `model.onnx`（3 输入 3 输出，见模块 README 的 I/O 表）
3. 放好后自检：`mvn -pl mica-ai-core/mica-ai-layout -am test`（集成测试会加载真模型跑完整链路）

> ⚠️ **`model-tools/layout/models/model.onnx` 是未跟踪文件，删除即不可恢复**（无 git 历史）。
> 本地副本指纹（供同一副本的完整性校验）：
> `sha256 = 45bf71750b00739a41fc209f132eb104a4d6b5bb29483c9078164d8b87cf28ba`

> 自行转换的产物**不保证**与此指纹一致（paddle2onnx 版本 / constant folding 差异都会改变
> 二进制），但应与模块 README 的 I/O 契约一致；不一致时先跑 `scripts/probe_real_onnx.py` 比对。

**为「模型不随仓库分发」做的配套**（避免新克隆仓库踩坑）：

| 位置 | 表现 |
|------|------|
| `LayoutIntegrationTest` | 模型缺失时用 `Assumptions` **整类跳过**（不失败），并在报告里给出提示 |
| `mica-ai-example/application.yml` | `mica.ai.layout.enabled: false` + 注释；本地有模型后改 `true` |
| 根 `.gitignore` | 显式排除该文件（写在 `!model-tools/*/models/**` 之后，靠「后匹配者胜」覆盖白名单） |

## 使用

Java 端直接按路径引用（支持 `classpath:`）：

```yaml
mica:
  ai:
    layout:
      model-path: model-tools/layout/models/model.onnx
```

模型规格 / I/O 格式 / 坐标空间约定 / 后处理详见 [`mica-ai-core/mica-ai-layout/README.md`](../../mica-ai-core/mica-ai-layout/README.md)，
落地过程与实测记录见 [`docs/layout-tracking.md`](../../docs/layout-tracking.md)。

## 脚本（`scripts/`）

| 脚本 | 用途 |
|------|------|
| `calibrate_thresholds.py` | **标定**：在真实文档图上扫描 `scoreThreshold`，输出分数分布 / 阈值扫描表 / 含阅读顺序的完整明细，用于选阈值（结论见模块 README「4. 阈值标定」） |
| `probe_coord_space.py` | **核心**：固定 `image` 张量、只改 `im_shape`/`scale_factor`，定位输出坐标空间（结论：`输出坐标 = 原图坐标 / scale_factor`）；自带判定结论输出 |
| `probe_real_onnx.py` | 打印 ONNX 输入 / 输出元信息（名字、shape、dtype），换模型或怀疑导出约定变化时先跑 |
| `probe_real_image.py` / `probe_real_image2.py` / `probe_order.py` | ⚠️ 早期探查脚本，**已被 `probe_coord_space.py` 取代**，仅留作过程记录。`probe_real_image.py` 里 `cxcywh` 解码假设是**错的**（col2-5 实为 `x1y1x2y2`），勿照抄 |

`demo_doc.jpg` 是标定用的真实文档图（1654×2339，双栏版式）：`calibrate_thresholds.py` 默认读它，
也可作为集成测试的外部图输入：

```bash
sh mvnc.sh -o -pl mica-ai-core/mica-ai-layout test \
    -Dmica.ai.layout.test.image=model-tools/layout/scripts/demo_doc.jpg
```

运行前提：WSL / Linux 下已装 `onnxruntime`、`numpy`、`Pillow`、`opencv-python`；模型路径默认取同级
`../models/model.onnx`（可用环境变量 `LAYOUT_ONNX` 覆盖，`probe_coord_space.py` / `calibrate_thresholds.py` 均支持）。
demo 图缓存到 `~/.cache/mica-ai-layout/`（WSL 的 `/tmp` 是 tmpfs，不要放中间产物）。
