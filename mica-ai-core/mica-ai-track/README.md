# mica-ai-track

> ByteTrack 多目标跟踪（**纯算法、零模型**），刻意保持零依赖：不引 [mica-ai-common](../mica-ai-common/README.md)，避免把 ONNX Runtime 拖进来。**MIT 可商用**。

把逐帧检测框交给 `ByteTracker#update`，即得**稳定 track id + 轨迹年龄**，是检测之上最小可用的跟踪底座。

---

## 快速开始

```java
ByteTracker tracker = ByteTracker.create(
    ByteTrackConfig.builder().build());

// 每帧：检测器结果 → 跟踪器
List<Track> tracks = tracker.update(detections);
for (Track t : tracks) {
    if (t.isMature(tracker.getConfig().getConfirmAge())) {
        int id = t.getId();          // 稳定 id，跨帧不变
        BBox box = t.getCurrentBBox();
        int age = t.getAge();        // 累计命中帧数
    }
}
```

配合检测器使用：

```java
try (DetectEngine detector = DetectEngine.create(detectConfig);
     ByteTracker tracker = ByteTracker.create(ByteTrackConfig.builder().build())) {

    for (Mat frame : frames) {
        DetectResult result = detector.detect(frame);
        List<BBox> boxes = result.getBoxes().stream()
            .map(b -> new BBox(b.getX1(), b.getY1(), b.getX2(), b.getY2(), b.getScore()))
            .collect(Collectors.toList());

        List<Track> tracks = tracker.update(boxes);
    }
}
```

---

## 核心概念

### 两级阈值 + 两轮匹配

ByteTrack 的关键在于**低分框不直接丢**，而是参与第二轮匹配：

| 阶段 | 参与对象 | 阈值 | 说明 |
|------|---------|------|------|
| 第 1 轮 | 高分 det ↔ 全部轨迹 | `matchThresh`（IoU） | 高分框 = `score >= highThreshold` |
| 第 2 轮 | 低分 det ↔ 第 1 轮未匹配的轨迹 | `matchLowThresh`（IoU） | 低分框 = `lowThreshold <= score < highThreshold`；`score < lowThreshold` 直接丢弃 |
| 新建轨迹 | 两轮都没命中的 det | — | `age=1`，`lostFrames=0` |

**为什么要有低分框这一轮**：遮挡时目标检测分数会骤降，高分框直接丢会导致轨迹断裂；用低分框续命，轨迹 id 才能跨遮挡保持稳定。

### 轨迹成熟度

- `age`：累计**匹配成功**的帧数
- `lostFrames`：自上次被匹配以来经过的帧数
- `isMature(confirmAge)`：`age >= confirmAge` 才算成熟轨迹

**成熟轨迹优先返回**——刚出现 1 帧的新目标往往定位不准，先别用。

### 轨迹淘汰

连续 `trackBufferedThresh` 帧未命中即丢弃该轨迹。`tracker.reset()` 可整体清空（换视频流 / 场景切换时用）。

---

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `highThreshold` | `0.6` | 高分框 / 低分框分界线 |
| `lowThreshold` | `0.1` | 低分框丢弃线；低于此值的框两轮都进不去 |
| `matchThresh` | `0.8` | 第 1 轮（高分 det ↔ 轨迹）IoU 阈值 |
| `matchLowThresh` | `0.5` | 第 2 轮（低分 det ↔ 未匹配轨迹）IoU 阈值 |
| `trackBufferedThresh` | `30` | 连续多少帧未命中就丢弃该轨迹 |
| `confirmAge` | `3` | 新轨迹连续命中多少帧才算成熟 |

约束：`highThreshold > lowThreshold`、`matchThresh` / `matchLowThresh` ∈ `(0,1]`、`trackBufferedThresh` / `confirmAge` 为正整数。非法时 `validate()` 抛 `MicaAiException`。

---

## 公开 API

### `ByteTracker`

| 方法 | 返回 | 说明 |
|------|------|------|
| `update(List<BBox>)` | `List<Track>` | 喂入本帧检测框，返回当前轨迹（成熟轨迹优先） |
| `trackCount()` | `int` | 当前轨迹数 |
| `reset()` | `void` | 清空全部轨迹与 id 计数 |
| `getConfig()` | `ByteTrackConfig` | 当前配置 |

### `BBox`（不可变）

`x1 / y1 / x2 / y2`（左上 / 右下角）、`score`、`classId`；辅助方法 `cx() / cy() / width() / height() / iou(other) / isEmpty()`。

### `Track`

| 方法 | 说明 |
|------|------|
| `getId()` | 稳定 id，跨帧不变 |
| `getCurrentBBox()` | 最近一次匹配到的框（未匹配时为预测框） |
| `getAge()` | 累计命中帧数 |
| `getLostFrames()` | 自上次命中以来的帧数 |
| `getScore()` / `getClassId()` | 最近一次的分数 / 类别 |
| `isMature(confirmAge)` | 是否已成熟 |

---

## 内部算法

| 类 | 作用 |
|----|------|
| `KalmanFilter` | 匀速模型（状态 8 维：位置 + 速度），预测 + 更新 |
| `HungarianMatcher` | 匈牙利算法求最优匹配（无第三方依赖，纯 Java 实现） |

跟踪器**状态非线程安全**（内部维护轨迹表与 id 计数）。多线程场景请每个线程各持一个实例，或在调用方加锁。

---

## 测试覆盖

- `KalmanFilterTest`（8 项）：匀速轨迹的预测误差、协方差收敛
- `HungarianMatcherTest`（9 项）：最优匹配正确性、空矩阵、代价矩阵边界
- `ByteTrackerTest`（12 项）：id 稳定性、遮挡续命、低分框丢弃、成熟度、淘汰、reset
- `BBoxTest`（8 项）：IoU 计算、宽高换算、空框判定

---

## Spring Boot Starter

见 [`mica-ai-track-spring-boot-starter`](../../mica-ai-starters/mica-ai-track-spring-boot-starter/README.md)：一行 YAML 注入 `ByteTracker` Bean。

---

## 许可

本模块代码为 mica-ai 主仓 Apache-2.0 的一部分；算法实现参考了 [ByteTrack](https://github.com/FoundationVision/ByteTrack)（MIT）的公开方法论，**不引入任何模型权重**。
