# mica-ai-track-spring-boot-starter

> [ByteTrack](https://github.com/FoundationVision/ByteTrack) 多目标跟踪 Spring Boot Starter，基于 [mica-ai-track](../../mica-ai-core/mica-ai-track/README.md) 核心模块。**MIT 可商用**。
>
> **纯算法、零模型**：一行 YAML 注入 `ByteTracker` Bean，把逐帧检测框交给它即得稳定 track id、轨迹点与跨帧速度。行为事件（越线 / ROI / 滞留 / 折返）由业务侧用 `TrackEventEngine` 自行按流创建，见 §3.1。

---

## 1. Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-track-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 需要同时引入 `spring-boot-starter`；与父项目保持一致，推荐 JDK 8+（Spring Boot 2.7.x）。
> 本能力**不引入任何模型权重**，无需下载、无需配置模型路径。

---

## 2. 配置项（`mica.ai.track` 前缀）

```yaml
mica:
  ai:
    track:
      enabled: true
      high-threshold: 0.6          # 高分框 / 低分框分界线
      low-threshold: 0.1           # 低分框丢弃线
      match-thresh: 0.8            # 第 1 轮（高分 det ↔ 轨迹）IoU 阈值
      match-low-thresh: 0.5        # 第 2 轮（低分 det ↔ 未匹配轨迹）IoU 阈值
      track-buffered-thresh: 30    # 连续多少帧未命中就丢弃该轨迹
      confirm-age: 3               # 新轨迹连续命中多少帧才算成熟
      history-size: 32             # 每条轨迹保留的采样点上限
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `ByteTracker` Bean |
| `high-threshold` | `0.6` | 高分框 / 低分框分界线 |
| `low-threshold` | `0.1` | 低分框丢弃线；低于此值的框两轮匹配都进不去 |
| `match-thresh` | `0.8` | 第 1 轮（高分 det ↔ 轨迹）IoU 阈值；视频流目标位移大时可适当调低 |
| `match-low-thresh` | `0.5` | 第 2 轮（低分 det ↔ 第 1 轮未匹配轨迹）IoU 阈值 |
| `track-buffered-thresh` | `30` | 连续多少帧未命中就丢弃该轨迹 |
| `confirm-age` | `3` | 新轨迹连续命中多少帧才算成熟（成熟轨迹优先返回） |
| `history-size` | `32` | 轨迹点历史上限，**最小 3**（方向折返判定需要 3 个点） |

> `high-threshold` 必须大于 `low-threshold`；`match-thresh` / `match-low-thresh` 取值 `(0,1]`；`track-buffered-thresh` / `confirm-age` 必须为正整数；`history-size >= 3`。本模块刻意零依赖，非法时启动即抛 **`IllegalArgumentException`**（不是 `MicaAiException`）。

---

## 3. 使用示例

```java
@RestController
@RequiredArgsConstructor
public class TrackController {

    private final DetectEngine detectEngine;
    private final ByteTracker byteTracker;

    /** 上传一帧 → 返回该帧的稳定 track id */
    @PostMapping("/track/frame")
    public List<Track> track(@RequestParam("file") MultipartFile file) throws IOException {
        DetectResult result = detectEngine.detectBytes(file.getBytes());
        List<BBox> boxes = result.getBoxes().stream()
            .map(b -> new BBox(b.getX1(), b.getY1(), b.getX2(), b.getY2(), b.getScore()))
            .collect(Collectors.toList());
        // 视频流请用带时间戳的重载，轨迹点才有真实采集时间
        return byteTracker.update(boxes, System.currentTimeMillis());
    }
}
```

### 公开 API（`ByteTracker`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `update(List<BBox>)` | `List<Track>` | 喂入本帧检测框（无时间戳兼容重载） |
| `update(List<BBox>, long timestampMs)` | `List<Track>` | 带采集时间戳，轨迹点 / 速度 / 滞留判定可用 |
| `trackCount()` | `int` | 当前轨迹数 |
| `getFrameIndex()` | `int` | 当前帧序号（从 1 开始，`reset()` 归零） |
| `reset()` | `void` | 清空全部轨迹与 id 计数（换视频流 / 场景切换时用） |
| `getConfig()` | `ByteTrackConfig` | 当前配置 |

### 3.1 逐帧事件管线为何不提供 Bean

`TrackEventEngine` 与四条规则（`LineCrossingRule` / `RoiCounterRule` / `DwellRule` / `DirectionRule`）都是**有状态**的：跟踪器、上一帧侧别、ROI 进入时间、上报帧号全都是跨帧累积的。做成单例 Bean 会让多路视频流互相污染状态。

请按流 key 自行创建与缓存：

```java
TrackEventEngine engine = TrackEventEngine.create(
    ByteTrackConfig.builder().build(),
    new LineCrossingRule("gate", 100, 400, 100, 0),
    new DwellRule("zone", Roi.rectangle(0, 0, 200, 200), 3000L));

FrameResult result = engine.process(boxes, frameTimestampMs);
```

> `DwellRule` 需要真实时间戳；用无时间戳的 `process(List)` 会直接抛异常，不会把帧数猜成时长。
>
> 判定线的端点顺序决定 `FORWARD` / `BACKWARD` 的含义：以上判定线**自下而上**，
> 因此「从左往右穿越」报 `FORWARD`。详见
> [核心模块 README §行为事件规则](../../mica-ai-core/mica-ai-track/README.md)。

### 3.2 相机运动补偿同样不提供 Bean

球机云台转动、车载 / 无人机摄像机移动时，静止目标的框在图像里整体位移，轨迹框与检测框的 IoU 塌陷 → 轨迹断裂、ID 换新 → 事件全部失真。`mica-ai-track` 提供 `AffineMotionModel`（2×3 仿射，参考帧 ↔ 当前帧）做这件事，但**运动模型同样是有状态、逐帧累积的**，所以也不做成 Bean：

```java
// 每路流一套：运动模型 + 跟踪器 + 规则
AffineMotionModel motion = AffineMotionModel.identity();
TrackEventEngine engine = TrackEventEngine.create(
    ByteTrackConfig.builder().build(),
    motion,
    new RoiCounterRule("zone", Roi.rectangle(0, 0, 200, 200), AnchorPolicy.CENTER, false));

for (int frame = 0; frame < frames.size(); frame++) {
    if (frame > 0) {
        // 本帧相对上一帧的相机运动（图像 GMC 或 PTZ 角度推出的增量）
        motion.append(AffineMotionModel.translation(-12f, 0f));
    }
    engine.process(boxesOf(frame), timestampOf(frame));
}
```

> 参考帧 = 运动模型在跟踪器创建那一刻的状态；**参考帧那一帧不追加增量**，从第 2 帧起每帧追加一次「相对上一帧」的增量。详见
> [核心模块 README §相机运动补偿](../../mica-ai-core/mica-ai-track/README.md)。

---

## 4. 两级阈值为什么这么设计

| 阶段 | 参与对象 | 阈值 | 说明 |
|------|---------|------|------|
| 第 1 轮 | 高分 det ↔ 全部轨迹 | `match-thresh` | `score >= high-threshold` 的框 |
| 第 2 轮 | 低分 det ↔ 第 1 轮未匹配的轨迹 | `match-low-thresh` | `low-threshold <= score < high-threshold` 的框 |
| 新建轨迹 | 两轮都没命中的 det | — | `age=1`，`lostFrames=0` |

**遮挡时目标检测分数会骤降**，若高分框直接丢弃，轨迹就会断裂。用低分框续命，track id 才能跨遮挡保持稳定——这正是 ByteTrack 的核心思路。

轨迹成熟度：`age`（累计命中帧数）达到 `confirm-age` 才算成熟；**成熟轨迹优先返回**，刚出现 1 帧的新目标往往定位不准。

---

## 5. 线程安全

`ByteTracker` 与 `TrackEventEngine` **状态非线程安全**（内部维护轨迹表、规则状态与 id 计数）。多线程消费同一视频流时，请每个线程各持一个实例，或在调用方加锁。

---

## 6. 自定义 Engine

Starter 通过 `@ConditionalOnMissingBean` 优先使用用户声明的 `ByteTracker` Bean：

```java
@Bean
public ByteTracker myTracker(TrackProperties props) {
    return ByteTracker.create(props.toConfig());
}
```

---

## 7. 测试覆盖

- [`ByteTrackerTest`](../../mica-ai-core/mica-ai-track/src/test/java/net/dreamlu/mica/ai/track/core/ByteTrackerTest.java)：id 稳定性、遮挡续命、低分框丢弃、成熟度、淘汰、reset
- [`ByteTrackerTimestampTest`](../../mica-ai-core/mica-ai-track/src/test/java/net/dreamlu/mica/ai/track/core/ByteTrackerTimestampTest.java)：时间戳、轨迹点上限、速度 / 方向、旧 API 兼容
- 事件规则与端到端管线：`LineCrossingRuleTest` / `RoiCounterRuleTest` / `DwellRuleTest` / `DirectionRuleTest` / `TrackEventEngineTest`
- [`HungarianMatcherTest`](../../mica-ai-core/mica-ai-track/src/test/java/net/dreamlu/mica/ai/track/algo/HungarianMatcherTest.java) · [`KalmanFilterTest`](../../mica-ai-core/mica-ai-track/src/test/java/net/dreamlu/mica/ai/track/algo/KalmanFilterTest.java)
- [`TrackPropertiesTest`](src/test/java/net/dreamlu/mica/ai/track/autoconfigure/TrackPropertiesTest.java)：默认值、kebab-case 绑定、`toConfig()` 装配一致性（含 `history-size`）、非法配置快速失败
- 相机运动补偿（核心模块内）：`AffineMotionModelTest` / `MotionCompensationTest` —— 仿射几何 + 无补偿 ID 断裂 / 有补偿 ID 稳定
