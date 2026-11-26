# mica-ai-track

> ByteTrack 多目标跟踪 + 逐帧行为事件（**纯算法、零模型**），刻意保持零依赖：不引 [mica-ai-common](../mica-ai-common/README.md)，避免把 ONNX Runtime / OpenCV / Spring 拖进来。**MIT 可商用**。

把逐帧检测框交给 `ByteTracker#update`，得**稳定 track id + 轨迹年龄**；把「检测框 + 采集时间戳」交给 `TrackEventEngine#process`，还能直接得到**越线 / ROI 进出 / 滞留 / 折返**事件。

---

## 快速开始

### 只跟踪

```java
ByteTracker tracker = ByteTracker.create(ByteTrackConfig.builder().build());

for (List<BBox> frame : detectionsPerFrame) {
    List<Track> tracks = tracker.update(frame);
    for (Track t : tracks) {
        if (t.isMature(tracker.getConfig().getConfirmAge())) {
            int id = t.getId();
            BBox box = t.getCurrentBBox();
            int age = t.getAge();
        }
    }
}
```

配合检测器使用：

```java
try (DetectEngine detector = DetectEngine.create(detectConfig)) {
    ByteTracker tracker = ByteTracker.create(ByteTrackConfig.builder().build());
    for (Mat frame : frames) {
        DetectResult result = detector.detect(frame);
        List<BBox> boxes = result.getBoxes().stream()
            .map(b -> new BBox(b.getX1(), b.getY1(), b.getX2(), b.getY2(), b.getScore(), b.getClassId()))
            .collect(Collectors.toList());
        List<Track> tracks = tracker.update(boxes);
    }
}
```

### 逐帧事件管线（推荐用于视频流）

```java
Roi zone = Roi.rectangle(0, 0, 200, 200);
LineCrossingRule line = new LineCrossingRule("gate", 100, 0, 100, 400);
RoiCounterRule counter = new RoiCounterRule("zone", zone, AnchorPolicy.BOTTOM_CENTER, true);
DwellRule dwell = new DwellRule("zone", zone, 3000L);
DirectionRule uTurn = new DirectionRule("u-turn");

TrackEventEngine engine = TrackEventEngine.create(
    ByteTrackConfig.builder().build(), line, counter, dwell, uTurn);

// 每一帧：调用方负责解码与检测，这里只吃检测框 + 该帧采集时间戳
FrameResult result = engine.process(detections, frameTimestampMs);
for (TrackEvent event : result.getEvents()) {
    System.out.println(event.getType() + " track=" + event.getTrackId()
        + " @" + event.getTimestampMs() + "ms " + event.getDetail());
}
```

`TrackEventEngine` **不碰图像、不碰视频解码**：RTSP / FFmpeg 取帧属于 [mica-video](https://github.com/lets-mica) 的职责，检测器把 `Mat` 转成 `List<BBox>` 后喂进来即可。这样本模块保持零模型、零 ONNX、零 OpenCV、零 Spring。

> ⚠️ **有状态、非线程安全**：`ByteTracker` 与每条规则都保存跨帧状态，**一路视频流必须独占一个 `TrackEventEngine`**（按流 key 缓存，不要做成单例 Bean）。换流 / 场景切换时调 `engine.reset()`。

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

### 轨迹成熟度与淘汰

- `age`：累计**匹配成功**的帧数；`lostFrames`：自上次匹配以来的帧数
- `isMature(confirmAge)`：`age >= confirmAge` 才算成熟轨迹，**成熟轨迹优先返回**
- 连续 `trackBufferedThresh` 帧未命中即丢弃；`tracker.reset()` 整体清空

### 轨迹点与真实时间

每帧恰好追加一个 `TrackPoint`：命中帧记录**实测框**，未命中帧记录 **Kalman 预测框**（`observed=false`）。行为规则默认只用实测点，避免遮挡期预测框误报。

`update(List<BBox>, long timestampMs)` 把采集时间戳写进轨迹点；`Track` 因此能提供：

| 方法 | 说明 |
|---|---|
| `getPoints()` / `observedPoints(max)` | 轨迹点历史快照（时间正序） |
| `getLastBBox()` | 本帧采样点对应的框，事件输出与规则判定同源 |
| `firstSeenMs` / `lastSeenMs` / `dwellMs()` | 轨迹存活时长（毫秒） |
| `velocityXPerSecond()` / `velocityYPerSecond()` / `speedPerSecond()` | 按真实时间差算的**像素 / 秒**，掉帧不失真 |
| `directionDegrees()` / `directionLabel()` | 0° 向右、90° 向下的八方向标签（右 / 右下 / 下 / …） |
| `kf.velocityX()` / `kf.velocityY()` | Kalman 状态里的**像素 / 帧**速度，供按帧推理用 |

> `update(List<BBox>)` 为兼容旧调用保留，此时 `timestampMs = Track.NO_TIMESTAMP(-1)`：速度与方向不可用，**按毫秒计时的规则会直接抛异常**，不会把帧数猜成时长。

---

## 相机运动补偿（CMC）

球机云台转动、车载 / 无人机挂载摄像机移动、手持抖动时，**静止目标**的框在图像里整体位移 —— 轨迹框与检测框的 IoU 直接塌陷 → 轨迹断裂、ID 换新 → 越线 / ROI 进出 / 滞留 / 折返事件全部失真。

### 原理：参考帧

跟踪器内部（Kalman 状态、匹配代价矩阵）**一律在「参考帧」坐标系里计算**：参考帧 = 跟踪器创建那一刻的图像坐标系。每帧的检测框先经 `toReference()` 变换回参考帧，再与 Kalman 预测框算 IoU —— 于是**相机运动对匹配完全没有影响**。

> 为什么不是 BoT-SORT 那种「把轨迹框往前推一帧」：Kalman 速度本身就是图像坐标下的速度，已经吸收了一部分相机运动，再往前推会把相机运动**重复计入**，匀速平移时反而更差。

### 用法

```java
// 参考帧 = 运动模型在跟踪器创建那一刻的状态（通常是恒等变换）
AffineMotionModel motion = AffineMotionModel.identity();
ByteTracker tracker = ByteTracker.create(config, motion);
TrackEventEngine engine = TrackEventEngine.create(config, motion, counter, dwell);

for (int frame = 0; frame < frames.size(); frame++) {
    if (frame > 0) {
        // 本帧相对上一帧的相机运动（图像 GMC 估出来的增量，或 PTZ 角度推出的近似仿射）
        motion.append(AffineMotionModel.translation(-12f, 0f));
    }
    engine.process(detections.get(frame), timestampMs.get(frame));
}
```

> ⚠️ **调用契约**：`append` 的增量是「本帧相对**上一帧**」的相机运动，**参考帧那一帧不追加增量**，从第 2 帧起每帧追加一次。

### 坐标系语义

| 位置 | 坐标系 | 用途 |
|------|--------|------|
| 轨迹点 `cx/cy`、`x1..y2` | **当前帧 / 图像坐标** | ROI 进出、越线这类「按画面位置判定」的规则 |
| 轨迹点 `refCx/refCy` | **参考帧 / 相机固定坐标** | 速度与方向 → 目标在**世界里的真实运动**，而非表观运动 |
| Kalman 状态、匹配代价 | **参考帧** | 相机运动对匹配零影响 |

未启用补偿时 `refCx/refCy` 为 `null`（`TrackPoint#hasReference()` 返回 false），速度回退到图像坐标，**行为与历史版本逐位一致**。

### `AffineMotionModel`

2×3 仿射（`x' = a*x + b*y + c`，`y' = d*x + e*y + f`），纯几何、零依赖：

| 方法 | 说明 |
|------|------|
| `identity()` / `translation(dx, dy)` | 恒等 / 纯平移 |
| `rotation(radians)` | 绕**原点**旋转（图像左上角） |
| `rotationAround(radians, cx, cy)` | 绕指定中心旋转 —— **相机滚转用这个**（绕主点，框中心不动） |
| `of(a,b,c,d,e,f)` / `copy()` / `reset()` | 构造 / 复制 / 复位 |
| `compose(first, second)` | 复合（先应用 `second` 再应用 `first`） |
| `append(incremental)` | 累积本帧增量（`this = this ∘ incremental`），**有状态**、非线程安全 |
| `toCurrent(box)` / `toReference(box)` | 参考帧 ↔ 当前帧；退化矩阵（不可逆）逆变换退化为恒等 |

**框的变换方式**：取四角点变换后求轴对齐外接矩形 —— 旋转会让外接矩形变大，这是几何事实不是缺陷；也意味着旋转场景下逆变换只能还原到「原框被外接矩形包住」的程度。

### 边界与选型

- **运动估计由调用方喂入**：本模块**不做图像 GMC 估计**（保持零 OpenCV 依赖）。图像 GMC（ORB / ECC 特征匹配）与 PTZ 角度换算的近似仿射，各家摄像机 SDK 给出的量纲与精度都不同，模块内置一套反而容易误用 —— 拿到位姿就喂进来，拿不到就先用「平移近似」。
- **参考帧会随相机持续转动而漂移**：长期巡航建议定期 `reset()` 并重建跟踪器，或由调用方在参考帧与当前帧之间做重锚定。
- **纯旋转场景收益有限**：外接矩形变大会让 IoU 匹配天然变松，此时更多依赖 `matchThresh` 调参。

---

## 行为事件规则

| 规则 | 触发 | 关键参数 |
|---|---|---|
| `LineCrossingRule` | 轨迹中心点从判定线一侧移到另一侧 | 线端点、`Direction`（BOTH / FORWARD / BACKWARD）、`requireMature`、`observedOnly` |
| `RoiCounterRule` | 锚点进入 / 离开矩形或多边形 ROI | `Roi`、`AnchorPolicy`、`requireMature`；提供 `getEnterCount()` / `getLeaveCount()` / `getCurrentInside()` |
| `DwellRule` | ROI 内驻留超过 `thresholdMs` | `Roi`、阈值毫秒、`AnchorPolicy`；离开后可重新计时再报 |
| `DirectionRule` | 连续两段位移夹角超过阈值（折返 / 逆行） | `turnAngleDegrees`（默认 120°）、`minDisplacement`（默认 5px）、`cooldownFrames`（默认 10） |

规则实现 `TrackEventRule#onFrame(FrameContext, List<TrackEvent>)`，可自行扩展。事件类型：

| `TrackEventType` | 含义 |
|---|---|
| `LINE_CROSSED` | 越线，`direction` 为 `FORWARD` / `BACKWARD` |
| `ROI_ENTERED` / `ROI_LEFT` | 进出 ROI；轨迹被删除时若仍在场，会补发 `ROI_LEFT`（`trackLost=true`） |
| `DWELL_EXCEEDED` | 滞留超时，`detail` 含 `stayMs=` |
| `DIRECTION_CHANGED` | 折返，`detail` 含 `turnAngle=` / `from=` / `to=` |

**锚点选择**：行人 / 车辆做区域判定建议用 `AnchorPolicy.BOTTOM_CENTER`（脚点 / 轮胎接地点），比几何中心稳定。

**Roi 构造**：`Roi.rectangle(x1,y1,x2,y2)` 或 `Roi.polygon(x0,y0,x1,y1,...)`（至少 3 个顶点，射线法判定）。

> ⚠️ **判定线的方向决定 `FORWARD` / `BACKWARD` 的含义**（实测确认）：
> 以 A(x1,y1) → B(x2,y2) 为基准，叉积 `(B-A) × (P-A)` **由负变正记 `FORWARD`、由正变负记 `BACKWARD`**。
> 对一条**竖直线**，这意味着：
>
> | 判定线画法 | 从左往右穿越 | 从右往左穿越 |
> |---|---|---|
> | 自下而上（`y1 > y2`） | `FORWARD` | `BACKWARD` |
> | 自上而下（`y1 < y2`） | `BACKWARD` | `FORWARD` |
>
> 想让「从左往右进场」报 `FORWARD`，就把线画成**自下而上**；或直接用 `Direction.FORWARD` /
> `Direction.BACKWARD` 只统计关心的方向。

---

## 配置项

| 字段 | 默认 | 说明 |
|------|------|------|
| `highThreshold` | `0.6` | 高分框 / 低分框分界线 |
| `lowThreshold` | `0.1` | 低分框丢弃线 |
| `matchThresh` | `0.8` | 第 1 轮 IoU 阈值；视频流目标位移大时可适当调低（如 0.3） |
| `matchLowThresh` | `0.5` | 第 2 轮 IoU 阈值 |
| `trackBufferedThresh` | `30` | 连续多少帧未命中就丢弃轨迹 |
| `confirmAge` | `3` | 新轨迹连续命中多少帧才算成熟 |
| `historySize` | `32` | 每条轨迹保留的采样点上限，**最小 3**（方向折返判定需要 3 个点） |

约束：`highThreshold > lowThreshold`、两个 match 阈值 ∈ `(0,1]`、`trackBufferedThresh` / `confirmAge` 为正整数、`historySize >= 3`。非法时 `validate()` 抛 **`IllegalArgumentException`**（本模块刻意零依赖，不共用 `MicaAiException`）。

---

## 公开 API

### `ByteTracker`

| 方法 | 返回 | 说明 |
|------|------|------|
| `update(List<BBox>)` | `List<Track>` | 无时间戳兼容重载 |
| `update(List<BBox>, long timestampMs)` | `List<Track>` | 带采集时间戳（毫秒），推荐用于视频流 |
| `trackCount()` / `reset()` / `getFrameIndex()` / `getConfig()` | — | 状态与配置 |
| `create(ByteTrackConfig)` / `create(ByteTrackConfig, MotionModel)` | `ByteTracker` | 构造；`MotionModel` 传 `null` 等同无补偿 |

### `TrackEventEngine`

| 方法 | 返回 | 说明 |
|------|------|------|
| `create(ByteTrackConfig, TrackEventRule...)` | `TrackEventEngine` | 用默认配置的跟踪器构造 |
| `create(ByteTrackConfig, MotionModel, TrackEventRule...)` | `TrackEventEngine` | 带相机运动补偿 |
| `create(ByteTracker, TrackEventRule...)` | `TrackEventEngine` | 复用已有跟踪器 |
| `process(List<BBox>, long timestampMs)` | `FrameResult` | 处理一帧；时间戳必须单调不减 |
| `process(List<BBox>)` | `FrameResult` | 无时间戳重载（滞留规则不可用） |
| `reset()` / `getTracker()` / `getRules()` | — | 状态与内部对象 |

`FrameResult` 含 `frameIndex` / `timestampMs` / `tracks` / `events`，均为不可变快照。`TrackEventEngine` 无资源可释放，**不实现 `AutoCloseable`**。

### 值对象

`BBox`（不可变检测框）、`TrackPoint`（采样点）、`TrackEvent`（事件）、`Roi`（多边形）、`FrameContext` / `FrameResult`（帧快照）。

---

## 内部算法

| 类 | 作用 |
|----|------|
| `KalmanFilter` | 匀速模型（状态 8 维：位置 + 速度），预测 + 更新 |
| `HungarianMatcher` | 匈牙利算法求最优匹配（无第三方依赖，纯 Java 实现） |
| `Directions` | 屏幕坐标系八方向换算 |
| `AffineMotionModel` | 2×3 仿射相机运动模型（参考帧 ↔ 当前帧），可累积 |

跟踪器**状态非线程安全**（内部维护轨迹表与 id 计数）。多线程场景请每个线程各持一个实例，或在调用方加锁。

---

## 测试覆盖

- `KalmanFilterTest`（8 项）：匀速轨迹的预测误差、协方差收敛、速度 getter
- `HungarianMatcherTest`（9 项）：最优匹配正确性、空矩阵、代价矩阵边界
- `ByteTrackerTest`（12 项）：id 稳定性、遮挡续命、低分框丢弃、成熟度、淘汰、reset
- `BBoxTest`（8 项）：IoU 计算、宽高换算、空框判定
- `ByteTrackerTimestampTest`（6 项）：时间戳、轨迹点上限、速度 / 方向、旧 API 兼容、reset
- `LineCrossingRuleTest` / `RoiCounterRuleTest` / `DwellRuleTest` / `DirectionRuleTest` / `TrackEventEngineTest`（28 项）：四条规则与端到端管线
- `AffineMotionModelTest`（12 项）：恒等 / 平移 / 绕中心旋转 / 逆变换 / 累积 / 复合 / 复位 / 复制 / 退化矩阵 / 空框
- `MotionCompensationTest`（6 项）：无补偿 ID 断裂、有补偿 ID 稳定、世界系速度归零、轨迹点双坐标系、无补偿时参考帧坐标为 null、ROI 计数稳定

合计 **89 项**，全部为合成检测框序列驱动，**零 ONNX / 零 OpenCV / 零图像 / 零视频库**。

```bash
mvn -pl mica-ai-core/mica-ai-track -am test
```

---

## Spring Boot Starter

见 [`mica-ai-track-spring-boot-starter`](../../mica-ai-starters/mica-ai-track-spring-boot-starter/README.md)：一行 YAML 注入 `ByteTracker` Bean。

**Starter 刻意不提供 `TrackEventEngine` Bean**——它和规则都是「一路流一个实例」的有状态对象，做成单例会让多路视频互相污染状态。请在业务侧按流 key 自行创建与缓存。

---

## 许可

本模块代码为 mica-ai 主仓 Apache-2.0 的一部分；算法实现参考了 [ByteTrack](https://github.com/FoundationVision/ByteTrack)（MIT）的公开方法论，**不引入任何模型权重**。行为事件规则为本仓库自行实现的纯几何逻辑。
