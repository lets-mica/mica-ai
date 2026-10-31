# mica-ai-track-spring-boot-starter

> [ByteTrack](https://github.com/FoundationVision/ByteTrack) 多目标跟踪 Spring Boot Starter，基于 [mica-ai-track](../../mica-ai-core/mica-ai-track/README.md) 核心模块。**MIT 可商用**。
>
> **纯算法、零模型**：一行 YAML 注入 `ByteTracker` Bean，把逐帧检测框交给它即得稳定 track id 与轨迹年龄。

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
```

| 配置 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用自动装配；`false` 时不装配 `ByteTracker` Bean |
| `high-threshold` | `0.6` | 高分框 / 低分框分界线 |
| `low-threshold` | `0.1` | 低分框丢弃线；低于此值的框两轮匹配都进不去 |
| `match-thresh` | `0.8` | 第 1 轮（高分 det ↔ 轨迹）IoU 阈值 |
| `match-low-thresh` | `0.5` | 第 2 轮（低分 det ↔ 第 1 轮未匹配轨迹）IoU 阈值 |
| `track-buffered-thresh` | `30` | 连续多少帧未命中就丢弃该轨迹 |
| `confirm-age` | `3` | 新轨迹连续命中多少帧才算成熟（成熟轨迹优先返回） |

> `high-threshold` 必须大于 `low-threshold`；`match-thresh` / `match-low-thresh` 取值 `(0,1]`；`track-buffered-thresh` / `confirm-age` 必须为正整数。非法时启动即抛 `MicaAiException`。

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
        return byteTracker.update(boxes);
    }
}
```

### 公开 API（`ByteTracker`）

| 方法 | 返回 | 说明 |
|------|------|------|
| `update(List<BBox>)` | `List<Track>` | 喂入本帧检测框，返回当前轨迹（成熟轨迹优先） |
| `trackCount()` | `int` | 当前轨迹数 |
| `reset()` | `void` | 清空全部轨迹与 id 计数（换视频流 / 场景切换时用） |
| `getConfig()` | `ByteTrackConfig` | 当前配置 |

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

`ByteTracker` **状态非线程安全**（内部维护轨迹表与 id 计数）。多线程消费同一视频流时，请每个线程各持一个实例，或在调用方加锁。

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
- [`HungarianMatcherTest`](../../mica-ai-core/mica-ai-track/src/test/java/net/dreamlu/mica/ai/track/algo/HungarianMatcherTest.java) · [`KalmanFilterTest`](../../mica-ai-core/mica-ai-track/src/test/java/net/dreamlu/mica/ai/track/algo/KalmanFilterTest.java)
