# mica-ai 第二轮能力计划：切片推理 + 相机运动补偿

> **启动日期**：2026-11-26
> **前置对齐**：AGENTS.md §1–§9（§6.1 商用自检、§6.2 50MB 红线、§7 标准操作、§8 验证清单）
> **与上一轮的区别**：上一轮（`docs/new-capabilities-plan.md`，2026-10-30 收口）扩了 8 个能力（EDFFIQA / 表情 / ByteTrack / NanoDet-YOLOX / PP-HumanSeg / DocAligner / PULC / Real-ESRGAN），**全部已完成**。本轮只加 **2 项通用机制**，**零新模型、零新依赖、零新增 Maven 模块、零新增 Starter**。

---

## 0. 选型依据（为什么是这两项）

评审了一份「摄像头事件识别」模型清单（10 项），按 mica-ai 的定位（Java 8 + ONNX Runtime 的**通用组件库**，不是垂直行业方案）逐条筛：

| 结论 | 项 | 理由 |
|------|----|------|
| ✅ **纳入本轮** | 切片推理（SAHI 语义） | 纯几何算法，零模型零依赖，通用价值最高 |
| ✅ **纳入本轮** | 相机运动补偿（CMC） | `mica-ai-track` 现成硬缺口：球机一动 ID 就漂 |
| ⚠️ 条件暂缓 | RF-DETR 检测族 | 等「有人带自训 DETR 权重」的真实诉求出现 |
| ⚠️ 条件暂缓 | 通用实例分割 | 是真空白但成本最高（protos + mask 解码 + 全系超 50MB） |
| ❌ 排除 | PP-YOLOE-SOD | VisDrone 垂直域权重，必须微调才有 PPE 价值，全系超限 |
| ❌ 排除 | DM-Count | 窄场景；`RoiCounterRule.getCurrentInside()` 已覆盖 80% 需求 |
| ❌ 排除 | EfficientNet-B0 二阶段 | **零改动**即可（`mica-ai-classify` 已是可插拔分类器） |
| ❌ 排除 | Grounding DINO | ORT 侧硬阻塞：`MultiScaleDeformableAttention` 非标准算子，主仓 contrib op（PR #20184）至今 WIP 未合并 |
| ❌ 排除 | SigLIP 复判层 | 双塔 + tokenizer 成本高于收益，误报复判用普通分类器即可 |
| ❌ 排除 | ORT / Optimum | 是运行时不是能力，mica-ai 已用 ORT 1.18 |

**核心判断**：mica-ai 该加的是「**通用机制**」，不是「垂直模型」。安全帽 / 反光衣 / 烟火这类由用户自训、自带权重（`model-path` 外置机制已支持），mica-ai 负责切片 + 跟踪 + 事件 + 分类门控。

---

## 1. 能力一：切片推理（扩 `mica-ai-detect`）

### 1.1 目标与边界

| 项 | 内容 |
|----|------|
| 目标 | 大图切成多片分别推理，提升小目标召回；跨片结果合并成一份原图坐标系的检测结果 |
| 挂载 | **扩 `mica-ai-detect`**，不新开 Maven 模块、不新增 Starter（现有 detect Starter 增一个配置块） |
| 新模型 | **无** |
| 新依赖 | **无**（OpenCV 已在，NMS 已在） |
| 不做 | 不做多尺度金字塔（不做 SPP/SAHI 的多切片尺寸）；不做旋转/翻转 TTA；不改变整图推理路径的行为 |

### 1.2 关键设计决定：切片边长 = 模型输入边长

现有 `DetectImageUtils.letterbox(bgr, size, padValue, interpolation)` 的反算关系是：

```
模型坐标 = 原图坐标 * ratio + (left, top)
```

`ObjectDetector.finish()` 反算回原图用 `(x - left) / ratio`。

**推论**：若切片是**正方形且边长 == 模型输入边长**（416），则 `ratio = min(416/416, 416/416) = 1.0`、`left = top = 0` → 反算退化为「加切片原点」，**且每片走的是与整图完全相同的推理契约**。

因此：

- **切片边长不做成自由参数**，恒等于 `config.getInputSize()`（模型张量边长）——模型输入尺寸固定，用户也无法选更小的切片；
- **切片网格步进** = `S - round(S * overlapRatio)`，`S = inputSize`；
- 每片的预处理走**同一个** `letterbox(slice, S, padValue, interpolation)`，`ratio` 恒为 1，**无二次缩放误差**。

> 这条推导同时排除了「边缘切片不足整块时放大」这个经典坑：边缘切片必须**补齐**到 `S×S`（见 1.4 第 2 步），否则 `letterbox` 会把它放大（`ratio > 1`），小目标优势直接归零。

### 1.3 配置面

新增 `SlicedDetectConfig`，**委托** `DetectConfig`（零字段重复）：

| 字段 | 默认 | 说明 |
|------|------|------|
| `detectConfig` | 必填 | 委托的整图检测配置（模型路径 / 族 / 阈值 / 标签…） |
| `overlapRatio` | `0.25` | 切片重叠比例，`(0, 1)`；重叠是**必需**的——没有重叠，位于切片边界上的物体会被切成两半，两半的 IoU 常常低于 NMS 阈值，会输出「半个物体」 |
| `maxSlices` | `64` | 切片数上限，超限抛异常并给出可操作提示（显式门槛，不是隐形门槛） |
| `mergeStrategy` | `IOU_MERGE` | 跨片合并策略，见下 |

```java
public enum SliceMergeStrategy {
    /** 跨片 IoU 合并（复用 Nms.suppress，类别语义与模型族一致） */
    IOU_MERGE,
    /** 不合并，保留各片原始框（调试 / 自定义后处理用） */
    KEEP_ALL
}
```

**校验**（`validate()`，抛 `MicaAiException(ErrorCode.ILLEGAL_ARGUMENT)`）：

- `detectConfig != null`
- `overlapRatio > 0 && overlapRatio < 1`
- `maxSlices > 0`
- `detectConfig.validate()` 的既有校验继续生效

### 1.4 算法流程

```
detect(Mat bgr):
  1. 若 bgr 为空 → 返回 null（与 DetectEngine 行为一致）
  2. 网格计算 SliceGrid.of(width, height, S=inputSize, step)
     - 起始点序列：0, step, 2*step, ...
     - 末列 / 末行回退：若 起点 + S > 边界，则起点 = 边界 - S（保证覆盖到边缘且不越界）
     - 若 边界 <= S → 只有 1 片（起点 0）
  3. 若 片数 > maxSlices → 抛 MicaAiException(ILLEGAL_ARGUMENT,
       "图像 WxH 在 416 切片下需要 N 片，超过上限 M；请调大 max-slices 或改用更大输入边长的模型")
  4. 对每个切片：
     a. ROI 提取；越界部分用 copyMakeBorder 补齐到 S×S（填充值按族：NanoDet 0 / YOLOX 114）
     b. engine.detect(slice) → 得到 **切片局部坐标** 的 DetectResult
     c. 每个框 + 切片原点 → 原图坐标；超出原图边界的部分裁掉
  5. 跨片合并：
     - IOU_MERGE → Nms.suppress(all, nmsThreshold, classAgnostic, maxDetections)
       （classAgnostic 语义必须与族一致：NanoDet 类别无关 / YOLOX 按类别）
     - KEEP_ALL → 直接收集
  6. 组装 new DetectResult(origW, origH, mergedBoxes)
```

### 1.5 类骨架

```
mica-ai-core/mica-ai-detect/src/main/java/net/dreamlu/mica/ai/detect/
├── core/
│   ├── DetectEngine.java            # 现有，不改
│   └── SlicedDetectEngine.java      # 新：切片编排门面
├── config/
│   ├── DetectConfig.java            # 现有，不改
│   └── SlicedDetectConfig.java      # 新：委托 DetectConfig + 切片参数
├── sliced/
│   ├── SliceGrid.java               # 新：网格计算（纯几何，可单测）
│   └── Slice.java                   # 新：值对象（x, y, size）
├── model/
│   ├── DetectBox.java               # 现有，不改
│   ├── DetectResult.java            # 现有，不改（复用公开构造器）
│   ├── SliceMergeStrategy.java      # 新：枚举
│   └── CocoLabels.java              # 现有，不改
└── postprocess/
    └── Nms.java                     # 现有，不改（跨片合并复用）
```

**`SlicedDetectEngine` 生命周期**：`DetectEngine` 已持有 ONNX session 且是 `AutoCloseable`。`SlicedDetectEngine` **只做编排、不持有模型**，依赖外部传入的 `DetectEngine`：

```java
DetectEngine engine = DetectEngine.create(detectConfig);       // 调用方持有并关闭
SlicedDetectEngine sliced = SlicedDetectEngine.create(engine, slicedConfig);
DetectResult result = sliced.detect(mat);
```

因此 `SlicedDetectEngine` **不实现 `AutoCloseable`**（无资源可释放），与 `mica-ai-track` 的 `TrackEventEngine` 同款处理。

**`SliceGrid` 边界规则**（单测重点）：

- `W <= S` → 1 片，起点 0
- `W = 1200, S = 416, step = 312` → 起点 `0, 312, 624, 784`（末列回退到 `1200-416=784`）
- 保证**所有原图像素至少被一片完整覆盖**（重叠 + 末列回退）

### 1.6 Starter 配置

`DetectProperties` 增加嵌套块 `Sliced`（YAML 前缀 `mica.ai.detect.sliced`）：

```yaml
mica:
  ai:
    detect:
      enabled: true
      model-path: model-tools/detect/models/object_detection_nanodet_2022nov.onnx
      family: NANODET_PLUS
      input-size: 416
      score-threshold: 0.35
      nms-threshold: 0.6
      max-detections: 100
      sliced:
        enabled: true
        overlap-ratio: 0.25
        max-slices: 64
        merge-strategy: IOU_MERGE
```

自动装配：`SlicedDetectEngine` Bean 依赖已有的 `DetectEngine` Bean（`@ConditionalOnBean`），不重复加载模型。

### 1.7 测试计划

| 测试 | 内容 |
|------|------|
| `SliceGridTest`（新，纯几何） | 片数计算；末列 / 末行回退到 `边界 - S`；`W <= S` 只 1 片；覆盖完整性（每像素至少被一片覆盖）；非法 `overlapRatio` 抛异常 |
| `SliceMergeStrategyTest`（新） | `IOU_MERGE` 跨片去重（同物两片 → 保留高分框）；`KEEP_ALL` 保留全部；类别语义随族切换（NanoDet 类别无关 / YOLOX 按类别） |
| `DetectIntegrationTest`（扩） | **核心验收**：合成 2000×1500 大图，内嵌 30×30 清晰目标 → 整图 letterbox 到 416 后目标仅约 8px（检不出），切片后目标 30px（检出）→ 断言切片模式召回更高、坐标落在原图范围内；空图返回 `null`；片数超 `maxSlices` 抛异常 |

> 集成测试沿用 `nu.pattern.OpenCV.loadLocally()`（`forkCount=0` 下 Java 8 不能用 `loadShared()`）。

### 1.8 验收标准

- [ ] 整图推理路径**行为零变化**（现有 `DetectIntegrationTest` 全绿，不改一行现有断言）
- [ ] 切片模式在合成小目标样本上召回高于整图模式
- [ ] 跨片同一物体不重复输出（IOU_MERGE）
- [ ] 切片数超限抛异常且消息可操作
- [ ] `mvn -pl mica-ai-core/mica-ai-detect -am test` 全绿

### 1.9 能实现的功能与 example 设计

#### 业务功能

| 场景 | 说明 |
|------|------|
| **远处小目标** | 整图 letterbox 到 416 后不足 20px 的目标（远处的人、车、烟头、落物）整图检不出，切片后按原像素推理可检出 |
| **超大画面** | 4K/8K 监控画面、扫描件、卫星图 —— 整图缩放直接丢细节 |
| **密集小目标** | 停车场货架、停车场车辆、密集人群 |
| **画面边缘目标** | letterbox 后边缘目标被缩得最小，整图模式下最容易漏 |

#### example 端点（新增 `SlicedDetectController`，前缀 `/detect`）

| 端点 | 请求 | 响应 | 演示什么 |
|------|------|------|---------|
| `POST /detect/sliced` | multipart `file` | JSON：`width` / `height` / `count` / `boxes` + `sliceCount` / `elapsedMs` | 切片模式基本用法，与 `/detect/detect` 返回结构对齐便于对比 |
| `POST /detect/sliced-compare` | multipart `file` | JSON：`whole{count,boxes,elapsedMs}` + `sliced{count,boxes,elapsedMs}` + **`onlyInSliced`** + `extraCostRatio` | **核心演示**：量化「切片多抓到了什么」与「算力代价多少倍」 |
| `POST /detect/sliced-annotate` | multipart `file` | PNG | 切片框画在原图上 |
| `POST /detect/sliced-grid` | multipart `file` | PNG | 切片网格可视化（叠加网格线 + 片序号），直观展示末列回退与边缘补齐 |

**`onlyInSliced` 的计算口径**：整图框与切片框按 IoU 双向匹配（阈值取 `nmsThreshold`），未匹配上的切片框即为「整图漏掉、切片抓到」—— 这是切片推理存在价值的**直接证据**，而不是只给两个数字让人自己看。

**`extraCostRatio`**：切片耗时 ÷ 整图耗时，让用户直观看到算力放大倍数（切片数 × 单片成本）。

---

## 2. 能力二：相机运动补偿（扩 `mica-ai-track`）

### 2.1 目标与边界

| 项 | 内容 |
|----|------|
| 目标 | 相机运动（球机云台转动）时，轨迹 ID 不再断裂，行为事件（越线 / ROI 进出 / 滞留 / 折返）不再失真 |
| 挂载 | **扩 `mica-ai-track`**，不新开 Maven 模块、不新增 Starter（现有 track Starter 不变） |
| 新模型 | **无** |
| 新依赖 | **无**（保持 `mica-ai-track` 零依赖：零 ONNX / 零 OpenCV / 零 Spring） |
| 不做 | 不做图像 GMC（ORB / ECC 全局运动估计）——需要 OpenCV，破坏零依赖定位；位姿未知的场景由调用方在业务侧估计后注入 |

**为什么不用图像 GMC**：① 要引 OpenCV，`mica-ai-track` 当前是**刻意零依赖**的（不引 `mica-ai-common`，避免把 ONNX Runtime 拖进来）；② 巡检球机是 PTZ，**有 yaw/pitch 角度时运动学补偿比图像估计更准也更省**，图像 GMC 只在位姿完全未知时才是唯一选择。

**参考**：camera motion compensation 的通用思路。[BoT-SORT](https://github.com/NirAharon/BoT-SORT)（**MIT**，非 GPL）是该方向的公开参考，但**具体做法本仓库没有照搬** —— 它的「把轨迹框往前推一帧」会把相机运动重复计入 Kalman 速度（见 §2.2），本仓库改用参考帧方案。

### 2.2 关键设计决定：匹配与 Kalman 全部活在「参考帧」

`ByteTracker.update()` 的匹配代价矩阵用的是 `d.iou(trk.getKf().predictedBBox())`（`ByteTracker.java:230`）。相机一动，轨迹框还在旧位置、检测框在新位置 → IoU 掉 → 匹配失败 → ID 断裂。

**做法**：把「相机固定坐标系」定义为**参考帧**（= 跟踪器创建那一刻的图像坐标系），跟踪器内部（Kalman 状态、匹配代价矩阵）**一律在参考帧下计算**：

1. 每帧的检测框先经 `motionModel.toReference(det)` 变换回参考帧
2. 代价矩阵用「参考帧下的检测框」与「参考帧下的 Kalman 预测框」算 IoU → **相机运动对匹配零影响**
3. Kalman 用参考帧坐标更新（`Track.onMatched(referenceBBox, currentBBox)`）
4. 轨迹点与对外框仍是**当前帧 / 图像坐标**（命中帧用实测框、未命中帧用 `toCurrent(kf.predictedBBox())`）→ ROI / 越线判定不失真
5. 轨迹点额外携带**参考帧中心** `refCx/refCy`，速度与方向优先用它算 → 得到的是目标在**世界里的真实运动**，而非「相机运动 + 目标运动」的表观运动

**为什么不用 BoT-SORT 那种「把轨迹框往前推一帧」**：Kalman 速度本身就是**图像坐标下**的速度，已经吸收了一部分相机运动；再往前推一帧等于把相机运动**重复计入**，匀速平移时反而更差。参考帧方案把相机运动从匹配与运动估计里**彻底摘出去**。

**局限（必须写进 README）**：① 旋转会让轴对齐外接矩形变大，逆变换只能还原到「原框被外接矩形包住」的程度；② 参考帧会随相机持续转动而漂移，长期巡航建议定期重锚定；③ 纯运动估计由调用方喂入，模块不内置图像 GMC。

### 2.3 接口与内置实现

```java
// net.dreamlu.mica.ai.track.motion.MotionModel
public interface MotionModel {
    /** 参考帧 → 当前帧 */
    BBox toCurrent(BBox referenceBox);

    /** 当前帧 → 参考帧 */
    BBox toReference(BBox currentBox);
}
```

**内置实现 `AffineMotionModel`**（2×3 仿射，纯几何、零依赖）：

```
x' = a*x + b*y + c
y' = d*x + e*y + f
```

- 四角点变换 → 取**轴对齐外接矩形**（旋转后框会变大，这是几何事实，不是 bug）
- 覆盖两种来源：图像 GMC 估出的仿射矩阵、PTZ 角度推出的近似仿射
- 构造：`AffineMotionModel.of(a, b, c, d, e, f)` / `identity()` / `translation(dx, dy)` / `rotation(radians)` / `rotationAround(radians, cx, cy)`
- **有状态**：`append(incremental)` 逐帧累积（`this = this ∘ incremental`），`compose(first, second)` 复合，`copy()` / `reset()` 复位
- 退化矩阵（不可逆）逆变换退化为恒等

**由谁更新变换矩阵**：`MotionModel` 实例由**调用方**持有并逐帧更新（PTZ 角度差 / GMC 估计结果），然后调用 `tracker.update(dets, timestampMs)`。**调用契约**：参考帧那一帧**不追加增量**，从第 2 帧起每帧追加一次「相对上一帧」的增量。

### 2.4 挂载点与 API 变更

| 文件 | 改动 |
|------|------|
| `motion/MotionModel.java` | 新增：接口（`toCurrent` / `toReference`） |
| `motion/AffineMotionModel.java` | 新增：仿射实现（有状态、可累积） |
| `core/ByteTracker.java` | 加 `motionModel` 字段（可为 null = 不补偿）；代价矩阵改用参考帧坐标；`framePredictedBox` 变换到当前帧；新轨迹用双框构造 |
| `model/Track.java` | 加 `framePredictedBox` / `referenceMatchedBBox` 字段；`onMatched(referenceBBox, currentBBox)` 重载；速度优先用参考帧坐标 |
| `model/TrackPoint.java` | 加可空的 `refCx` / `refCy` 与 `hasReference()` |
| `event/TrackEventEngine.java` | 新增 `create(ByteTrackConfig, MotionModel, TrackEventRule...)` 重载 |

**API 兼容**：`ByteTracker.create(config)` 与 `update(List<BBox>)` / `update(List<BBox>, long)` 签名**全部不变**，默认 `motionModel == null` → 行为与现在**逐位一致**（连轨迹点的 `refCx` 都为 `null`），现有 71 项单测不需要改。

```java
ByteTracker tracker = ByteTracker.create(config, motionModel);
// 或
TrackEventEngine engine = TrackEventEngine.create(config, motionModel, lineRule, roiRule);
```

### 2.5 测试计划

| 测试 | 内容 |
|------|------|
| `AffineMotionModelTest`（新，12 项） | 恒等 / 平移 / 绕中心旋转 / 绕原点旋转 / 轴对齐仿射的逆变换精确还原 / 旋转下逆变换只是近似 / 累积 / 复合 / 复位 / 复制 / 退化矩阵 / 空框 |
| `MotionCompensationTest`（新，6 项） | **核心验收**：合成序列——目标在世界里静止，相机整体横摇（图像内容整体平移）。断言：<br>① 无 `MotionModel` 时 ID 断裂（出现新 ID）；<br>② 有 `MotionModel` 时 ID 稳定（不换 ID）；<br>③ 参考帧口径的速度为 0（表观速度非 0）；<br>④ 轨迹点双坐标系（图像坐标 + 参考帧中心恒定）；<br>⑤ 未启用补偿时参考帧坐标为 `null`；<br>⑥ ROI 计数稳定 |
| `ByteTrackerTest` / `ByteTrackerTimestampTest` | **不改**——默认无补偿，行为不变，直接作为回归防线 |

### 2.6 验收标准

- [x] 默认路径（`motionModel == null`）行为与当前**完全一致**（现有 71 项单测全绿，track 模块共 89 项全绿）
- [x] 合成相机运动序列下，补偿后 ID 不再断裂
- [x] 轨迹点与对外框坐标系一致（不出现「实测点与预测点混坐标系」）
- [x] `mvn -pl mica-ai-core/mica-ai-track -am test` 全绿
- [x] 模块仍**零依赖**（`mica-ai-track/pom.xml` 不新增任何 dependency）

### 2.7 能实现的功能与 example 设计

#### 业务功能

| 场景 | 说明 |
|------|------|
| **球机巡航巡检** | PTZ 转动时轨迹 ID 不断裂，区域入侵 / 滞留时长 / 进出计数统计准确 |
| **移动式摄像机** | 车载、无人机挂载、机械臂、云台导轨 |
| **相机抖动抑制** | 抖动不再产生一串假 ID（每个抖动周期一个「僵尸轨迹」） |
| **事件方向正确性** | 相机转动时表观方向是失真的 → 折返 / 逆行会误报；补偿后方向按**参考帧坐标**计算，判定恢复正确 |
| **跨相机位姿衔接** | 位姿已知时，可把上一机位的轨迹映射到当前机位（为后续 ReID 铺路） |

#### example 端点（新增端点，**不改**现有 `/scenario/video-events/analyze`）

| 端点 | 请求 | 响应 | 演示什么 |
|------|------|------|---------|
| `POST /scenario/video-events/analyze-with-motion` | 现有 `AnalyzeRequest` + `increments`（每帧相对上一帧的 `a,b,c,d,e,f` 仿射增量，长度 = 帧数 − 1）+ `compareWithoutMotion` 开关 | JSON：两套 `summary`（`trackIdCount` / 事件计数 / `enters` / `leaves` / `maxInside`）+ `frames` | 同一份输入，**有补偿 vs 无补偿**的并列差异 |
| `POST /scenario/video-events/motion-demo` | 无（内置合成序列） | JSON（两套对照 + 图像坐标路径 + 参考帧路径 + 三种速度口径） | **零输入演示**：相机横摇 + 静止目标，无补偿 ID 断裂 / ROI 重复计数，有补偿 ID 稳定 / 速度归零 |

**`motion-demo` 的合成数据设计**（内置，不依赖用户上传）：

- 目标在世界坐标静止，相机每帧横摇 → 图像中目标相对坐标每帧反向平移 `dx = 20px`
- 每帧增量仿射 `(a=1, b=0, c=-20, d=0, e=1, f=0)`
- 无补偿：目标框逐帧漂移，IoU 掉到 `matchThresh` 以下 → 轨迹断裂、产生多个 ID、ROI 重复计数；且轨迹碎成单点 → **跟踪器根本给不出速度**
- 有补偿：检测框被变换回参考帧 → ID 稳定、ROI 只进一次、参考帧口径速度为 0（而表观速度仍是 −200px/s）

这条示例是 CMC 能力的**自证**：不依赖外部视频源，纯合成数据就能复现「不补偿会坏」和「补偿后不坏」。

### 2.8 组合场景：两个能力协同的端到端示例

**小球机巡检管线**（`pipeline/inspection/InspectionPipeline`，风格对齐现有 `DocumentPipeline` / `IdPhotoPipeline`）：

```
输入：多帧图片（multipart）+ 每帧 PTZ 角度 / 仿射矩阵 + 事件规则（判定线 / ROI / 滞留阈值）
流程：切片检测（远处小目标） → 运动补偿跟踪（ID 不断裂） → 事件规则（越线 / ROI 进出 / 滞留 / 折返）
输出：每帧轨迹 + 事件流 + 叠加标注 PNG
```

这是**唯一同时证明两个能力协同**的示例：切片负责「看得见」，补偿负责「跟得住」，事件负责「判得准」。三者缺一，巡检链路就断在哪一环。

| 端点 | 请求 | 响应 |
|------|------|------|
| `POST /scenario/inspection/analyze` | multipart 多帧 + JSON 规则 | JSON：逐帧轨迹 + 事件 + summary |

---

## 3. 实施顺序与里程碑

按 AGENTS.md「**一个 PR 一个能力**」拆成两个独立 PR：

```
PR 1（切片推理，扩 mica-ai-detect）
  ├── SliceGrid / Slice / SliceMergeStrategy / SlicedDetectConfig / SlicedDetectEngine
  ├── SliceGridTest / SliceMergeStrategyTest
  ├── DetectIntegrationTest 扩展（小目标召回对比）
  ├── DetectProperties.Sliced + 自动装配
  ├── example：SlicedDetectController（/detect/sliced、/detect/sliced-compare、
  │          /detect/sliced-annotate、/detect/sliced-grid）
  └── README（core + starter）+ 根 README + CHANGELOG

PR 2（相机运动补偿，扩 mica-ai-track）
  ├── MotionModel / AffineMotionModel
  ├── ByteTracker / Track / TrackEventEngine 挂载
  ├── AffineMotionModelTest / MotionCompensationTest
  ├── example：/scenario/video-events/analyze-with-motion、/scenario/video-events/motion-demo
  └── README（core）+ 根 README + CHANGELOG

PR 3（组合场景，依赖 PR 1 + PR 2）
  ├── example：pipeline/inspection/InspectionPipeline + /scenario/inspection/analyze
  └── example README + 根 README + CHANGELOG
```

**example 的写法约束**（对齐现有 `DetectController` / `VideoEventScenarioController`）：

- Swagger 注解齐全（`@Tag` / `@Operation` / `@ApiResponses`）
- `@ConditionalOnProperty(prefix = "mica.ai.xxx", name = "enabled", ...)` 开关
- JSON 错误统一 `{"error": "..."}` + 400；推理异常 500
- **有状态对象每次请求新建**（跟踪器 / 事件引擎一路流一个实例）
- 现有 `/scenario/video-events/analyze` **不改签名**，新增独立端点，保持向后兼容

每个 PR 结束前跑对应模块 `mvn -pl mica-ai-core/<module> -am test`；两个 PR 全部结束后跑 `mvn -DskipTests install` + 全量 `mvn test`。

---

## 4. 统一 checklist（每个 PR 结束后复核）

- [ ] JDK 8 兼容：无 `var` / text block / record / sealed
- [ ] 测试方法名**英文 camelCase**（`parse_labelOnRight_takesValue` 风格），注释中文
- [ ] 单字符用 char 字面量（`'-'`），不写 `"-"`
- [ ] 仅用 SLF4J 日志，无 `System.out.println`
- [ ] 业务异常抛 `MicaAiException(ErrorCode.XXX, "...")`；`mica-ai-track` 刻意零依赖，参数校验用 `IllegalArgumentException`
- [ ] 无新增 Maven 模块 → **不改**根 `pom.xml` 的 `dependencyManagement` / `modules`
- [ ] 无新增模型 → `model-tools/` **不动**，根 `NOTICE` **不动**
- [ ] 集成测试用 `nu.pattern.OpenCV.loadLocally()`
- [ ] 公共 API 变更同步更新：模块 README + Starter README + 根 README 能力表
- [ ] CHANGELOG 按类型分组（feat → fix → perf → refactor），一条一段、零子条、只留最终态
- [ ] `mvn -pl <module> -am test` 全绿 + `mvn -DskipTests install` 全量通过

---

## 5. 暂缓项与触发条件

| 项 | 触发条件 | 当前判断 |
|----|---------|---------|
| RF-DETR 检测族（`DetectModelFamily` 新增一族） | 有用户带**自训 DETR 权重**来接入 | 触发后开工：需新写无 NMS 解码器（top-k + sigmoid）；模型 30.5M 参数 → FP32 约 120MB **超 50MB 红线，只能外置** |
| 通用实例分割（`mica-ai-seg` 或 detect 扩展） | 有明确的实例分割诉求 | 触发后开工：RF-DETR-Seg-N/S/M 均 33~36M 参数（ONNX 130MB+，全系超限）；输出要处理 protos + mask 阈值化 + 缩放回原图 + 裁剪 |
| 跨相机关联（ReID） | 多摄像头同一人关联诉求 | 需外观模型 → 破坏 track 零依赖定位，须**独立立项**；权重许可需单独核（OSNet / NP-ReID） |

---

## 6. 排除项重申（避免重踩坑）

| 排除项 | 原因 |
|--------|------|
| 图像 GMC（ORB / ECC）内置到 `mica-ai-track` | 要引 OpenCV，破坏零依赖定位；PTZ 有角度时运动学补偿更准 |
| 切片尺寸做成自由参数 | 模型输入尺寸固定，切片边长必须 == 模型输入边长，否则要么放大（丢小目标）要么填充（浪费算力） |
| 切片不做边缘补齐 | 边缘切片不足 `S×S` 时 `letterbox` 会放大，小目标优势归零 |
| 跨片合并不带重叠 | 边界物体被切两半，两半 IoU 常低于 NMS 阈值 → 输出「半个物体」 |
| 多尺度金字塔 / TTA | 本轮范围外，效果收益待实测后再议 |
| Grounding DINO 等 DETR 系接 ORT | `MultiScaleDeformableAttention` 非 ONNX 标准算子，ORT 主仓 contrib op 至今未合并 |
| 引入 PP-YOLOE-SOD / DM-Count 等垂直模型 | 垂直域 + 必须微调 + 体积超限，属应用层选型，不是 SDK 能力 |

---

## 7. 待拍板项

1. **版本号**：新增能力按语义化版本应走 **1.1.0**，当前 `revision=1.0.3-SNAPSHOT`（`Unreleased` 段已含 barcode + track 逐帧事件管线）。是否在 PR 1 动手前把 `revision` 提到 1.1.0，由 owner 定。
2. **是否拆两个 PR**：本计划按「一个 PR 一个能力」拆成 PR 1 / PR 2。若要合并成一个 PR，需接受 `mica-ai-detect` 与 `mica-ai-track` 同时改动。

---

**文档到此。PR 1（切片推理）可开工。**
