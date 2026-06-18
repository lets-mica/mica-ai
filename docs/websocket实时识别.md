# WebSocket 实时语音识别方案

基于 mica-ai-voice 模块，通过 WebSocket 实时推送音频流，实现边说边出字的语音识别。

---

## 1. 整体架构

```
┌──────────────┐          WebSocket           ┌──────────────────────┐
│   客户端      │  ◄──── binary audio ──────►  │      Java 服务端      │
│ (浏览器/App)  │  ──── JSON results ────►     │                      │
└──────────────┘                              │  ┌────────────────┐  │
                                              │  │ AudioPipeline  │  │
                                              │  │ (Ring Buffer)  │  │
                                              │  └───────┬────────┘  │
                                              │          ▼           │
                                              │  ┌────────────────┐  │
                                              │  │     VAD        │  │
                                              │  │ (端点检测)      │  │
                                              │  └───────┬────────┘  │
                                              │          ▼           │
                                              │  ┌────────────────┐  │
                                              │  │  SenseVoice    │  │
                                              │  │  (chunk 推理)   │  │
                                              │  └───────┬────────┘  │
                                              │          ▼           │
                                              │  ┌────────────────┐  │
                                              │  │ ResultSender   │  │
                                              │  │ (JSON 推送)     │  │
                                              │  └────────────────┘  │
                                              └──────────────────────┘
```

### 核心流程

1. 客户端采集麦克风音频，通过 WebSocket 持续发送 PCM 二进制帧
2. 服务端将音频写入 `AudioPipeline` 环形缓冲区
3. `VAD` 持续检测语音活动状态
4. 缓冲区累积满一个 chunk（如 3 秒）→ 触发 `SenseVoice.recognize()` → 推送 **partial** 结果
5. VAD 检测到说话结束（静音超过阈值）→ 触发最终识别 → 推送 **final** 结果

---

## 2. 通信协议

### 客户端 → 服务端

| 类型 | 格式 | 说明 |
|------|------|------|
| 音频帧 | Binary (Int16 PCM) | 16kHz 单声道，每帧 4096 samples (~256ms) |
| 控制指令 | JSON Text | `{"action": "start"}` / `{"action": "stop"}` |

### 服务端 → 客户端

```json
// 中间结果（chunk 识别完成时推送）
{"type": "partial", "text": "那我来测", "ts": 1.5}

// 最终结果（VAD 检测到说话结束时推送）
{"type": "final", "text": "那我来测试一下Fun-ASR-Nano。", "ts": 3.4, "hotwords": ["Fun-ASR-Nano"]}

// 静音
{"type": "silence"}

// 错误
{"type": "error", "msg": "识别失败: ..."}
```

---

## 3. 客户端实现

### 浏览器（JavaScript）

```javascript
const ws = new WebSocket('ws://localhost:8080/asr');
const mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: { sampleRate: 16000, channelCount: 1 }
});

const audioContext = new AudioContext({ sampleRate: 16000 });
const source = audioContext.createMediaStreamSource(mediaStream);
const processor = audioContext.createScriptProcessor(4096, 1, 1);

processor.onaudioprocess = (e) => {
    const float32 = e.inputBuffer.getChannelData(0);
    // Float32 → Int16 PCM
    const int16 = new Int16Array(float32.length);
    for (let i = 0; i < float32.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
    }
    ws.send(int16.buffer);
};

source.connect(processor);
processor.connect(audioContext.destination);

// 接收识别结果
ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    switch (data.type) {
        case 'partial':
            // 实时更新界面（灰色文字）
            showPartialText(data.text);
            break;
        case 'final':
            // 追加到最终文本（黑色文字）
            appendFinalText(data.text, data.hotwords);
            break;
        case 'silence':
            // 可选：显示"正在聆听..."
            break;
    }
};
```

### 移动端 / 桌面端

原理相同：采集 PCM → WebSocket 发送 binary → 接收 JSON results。
- Android: `AudioRecord` API
- iOS: `AVAudioEngine`
- 桌面: `javax.sound.sampled.TargetDataLine`

---

## 4. 服务端实现

### 4.1 需要新增的组件

| 组件 | 职责 | 说明 |
|------|------|------|
| `AudioPipeline` | 音频缓冲 + 重采样 | 环形缓冲区，累积音频并按需取出 |
| `SileroVAD` | 语音端点检测 | 基于 Silero VAD ONNX 模型 (~2MB) |
| `AsrWebSocketHandler` | WebSocket 协议处理 | 串联以上组件 |

### 4.2 AudioPipeline

```java
public class AudioPipeline {
    private final float[] ringBuffer;
    private int writePos = 0;
    private int readableSamples = 0;
    private final int sampleRate = 16000;

    public AudioPipeline(int capacitySeconds) {
        this.ringBuffer = new float[capacitySeconds * sampleRate];
    }

    /** 追加客户端推送的 Int16 PCM 音频 */
    public synchronized void append(byte[] pcm16) {
        int samples = pcm16.length / 2;
        ByteBuffer buf = ByteBuffer.wrap(pcm16).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < samples; i++) {
            ringBuffer[writePos] = buf.getShort() / 32768.0f;
            writePos = (writePos + 1) % ringBuffer.length;
        }
        readableSamples = Math.min(readableSamples + samples, ringBuffer.length);
    }

    /** 取出指定时长的音频（从最早的数据开始） */
    public synchronized float[] drain(double seconds) {
        int count = Math.min((int)(seconds * sampleRate), readableSamples);
        float[] result = new float[count];
        int readPos = (writePos - readableSamples + ringBuffer.length) % ringBuffer.length;
        for (int i = 0; i < count; i++) {
            result[i] = ringBuffer[(readPos + i) % ringBuffer.length];
        }
        readableSamples -= count;
        return result;
    }

    /** 缓冲区中有多少秒的音频 */
    public synchronized double bufferedSeconds() {
        return (double) readableSamples / sampleRate;
    }
}
```

### 4.3 SileroVAD

```java
public class SileroVAD implements AutoCloseable {
    private final OrtSession session;
    private float[][] state;       // LSTM 隐状态
    private int silenceFrames = 0;
    private static final int SILENCE_THRESHOLD = 5; // 连续 5 帧静音 → 说话结束

    public SileroVAD(String modelPath, OrtEnvironment env) throws OrtException {
        this.session = env.createSession(modelPath, new OrtSession.SessionOptions());
        this.state = new float[2][512]; // SR=16k, window=512
    }

    /**
     * 处理一段音频（~30ms），返回是否有语音活动。
     */
    public boolean process(float[] audio) {
        // ONNX 推理 → speech probability
        float prob = inference(audio);
        if (prob > 0.5f) {
            silenceFrames = 0;
            return true;
        } else {
            silenceFrames++;
            return false;
        }
    }

    /** 是否检测到说话结束（静音超过阈值） */
    public boolean isSpeechEnd() {
        return silenceFrames >= SILENCE_THRESHOLD;
    }

    public void reset() {
        state = new float[2][512];
        silenceFrames = 0;
    }
}
```

### 4.4 AsrWebSocketHandler

```java
public class AsrWebSocketHandler {
    private final SenseVoice voice;
    private final SileroVAD vad;
    private final AudioPipeline pipeline;

    // 配置
    private static final double CHUNK_SECONDS = 3.0;  // 每 chunk 识别时长
    private static final double MIN_CHUNK = 1.0;       // 最小识别时长

    public AsrWebSocketHandler(SenseVoice voice, SileroVAD vad) {
        this.voice = voice;
        this.vad = vad;
        this.pipeline = new AudioPipeline(60); // 60 秒缓冲区
    }

    /**
     * 收到客户端音频帧时调用。
     */
    public void onAudio(Session wsSession, byte[] pcm16) {
        // 1. 写入缓冲区
        pipeline.append(pcm16);

        // 2. VAD 检测
        float[] vadChunk = pipeline.peekLast(0.03f); // 取最后 30ms
        boolean hasSpeech = vad.process(vadChunk);

        // 3. 累积够 chunk → 触发 partial 识别
        if (pipeline.bufferedSeconds() >= CHUNK_SECONDS) {
            float[] audio = pipeline.drain(CHUNK_SECONDS);
            CompletableFuture.runAsync(() -> {
                TranscriptionResult result = voice.recognize(audio);
                sendJson(wsSession, Map.of(
                    "type", "partial",
                    "text", result.text(),
                    "ts", result.timings().total()
                ));
            });
        }

        // 4. VAD 检测到说话结束 → 触发 final 识别
        if (vad.isSpeechEnd() && pipeline.bufferedSeconds() >= MIN_CHUNK) {
            float[] audio = pipeline.drain(pipeline.bufferedSeconds());
            vad.reset();
            TranscriptionResult result = voice.recognize(audio);
            sendJson(wsSession, Map.of(
                "type", "final",
                "text", result.text(),
                "hotwords", result.hotwords(),
                "ts", result.timings().total()
            ));
        }

        // 5. 长时间无语音 → 推送 silence
        if (vad.isSpeechEnd() && pipeline.bufferedSeconds() < MIN_CHUNK) {
            sendJson(wsSession, Map.of("type", "silence"));
            vad.reset();
        }
    }
}
```

---

## 5. 延迟分析

### chunk 大小 vs 延迟

| chunk 大小 | 用户体感延迟 | 识别精度 | 适用场景 |
|-----------|-------------|---------|---------|
| **1s** | ~1-2s | 一般（短句容易截断） | 实时字幕 |
| **3s** | ~3-4s | 较好 | 语音转写、会议记录 |
| **5s** | ~5-6s | 最佳 | 离线转录、长音频 |

### 时间线示例（chunk=3s）

```
0.0s  ─── 用户开始说话 ──────────────────────────►
1.0s  ─── 继续 ─────────────────────────────────►
2.0s  ─── 继续 ─────────────────────────────────►
3.0s  ─── chunk 满 → 识别 ──► partial "那我来测"
4.0s  ─── 继续 ─────────────────────────────────►
5.0s  ─── 继续 ─────────────────────────────────►
6.0s  ─── chunk 满 → 识别 ──► partial "试一下Fun-ASR"
7.0s  ─── 用户停止说话 ─────────────────────────►
7.8s  ─── VAD 静音 800ms ──► final "那我来测试一下Fun-ASR-Nano。"
```

---

## 6. 性能优化

### 6.1 推理线程池

每个 WebSocket 连接独占一个推理线程会导致资源浪费。建议使用线程池：

```java
ExecutorService asrPool = Executors.newFixedThreadPool(
    Runtime.getRuntime().availableProcessors()
);

// 每次识别提交到线程池
asrPool.submit(() -> {
    TranscriptionResult result = voice.recognize(audio);
    sendJson(wsSession, ...);
});
```

### 6.2 并发限制

SenseVoice 单实例不支持并发推理。多连接场景下需要实例池：

```java
// 预创建 N 个 SenseVoice 实例
BlockingQueue<SenseVoice> instancePool = new LinkedBlockingQueue<>();

// 每个请求借一个实例
SenseVoice voice = instancePool.take();
try {
    result = voice.recognize(audio);
} finally {
    instancePool.offer(voice);
}
```

### 6.3 VAD 前置过滤

VAD 判断为静音时不触发识别，避免浪费算力：

```
实际效果：10 分钟会议中有效语音约 3-5 分钟
         → 减少 50-70% 的推理调用
```

---

## 7. 依赖清单

| 依赖 | 用途 | 大小 |
|------|------|------|
| `mica-ai-voice` | ASR 引擎 | 现有 |
| `silero-vad.onnx` | 语音端点检测 | ~2MB |
| WebSocket 框架 | 通信层 | Spring WebSocket / Undertow / Jetty |

Silero VAD 模型获取：

```bash
# 从 Silero 仓库下载 ONNX 模型
pip install silero-vad
python -c "from silero_vad import load_silero_vad; model = load_silero_vad(onnx=True)"
# 或直接下载：https://github.com/snakers4/silero-vad
```

---

## 8. 更低的延迟方案

如果对延迟有更严苛的要求（< 500ms），需要换用**流式模型**：

| 方案 | 延迟 | 改动量 | 说明 |
|------|------|--------|------|
| chunk=1s + SenseVoice | ~1-2s | 小 | 调参数即可 |
| chunk=0.5s + 重叠 | ~1s | 中 | 改分段逻辑 |
| Paraformer-streaming | ~300ms | 大 | 换模型 + 重写 Encoder |
| Zipformer-streaming | ~200ms | 大 | 换模型 + KV Cache |

**推荐路线**：先用 chunk=2-3s + VAD 跑通全链路，体验满意后再考虑换流式模型。
