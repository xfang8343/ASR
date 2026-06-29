# 🎙️ ASR Speech Recognition Project

基于 [OpenAI Whisper](https://github.com/openai/whisper) 的语音识别项目，支持音频文件识别与实时麦克风输入，兼容中英文多语言场景。

---

## 📁 项目结构

```
ASR/
├── asr_mp3.py          # 音频文件识别脚本（支持转录与翻译）
├── asr_micro.py        # 实时麦克风识别主入口
├── mic/
│   ├── __init__.py
│   ├── recorder.py     # 麦克风录音 + Silero-VAD 语音活动检测
│   ├── transcriber.py  # Whisper 识别（自动检测语言）
│   ├── translator.py   # 翻译接口（Coming Soon）
│   └── saver.py        # 识别结果追加保存
├── result/             # 识别结果输出目录
└── test_audio/
    ├── ZH.mp3          # 中文测试音频
    └── EN.mp3          # 英文测试音频
```

---

## ✨ 功能简介

### `asr_mp3.py` — 音频文件识别与翻译

运行后交互选择功能：

**1. 语音转录**
- 支持 `.mp3` 等常见音频格式
- 自动检测语言，或手动指定中文 / 英文
- 中文输出自动转为简体（opencc）
- 结果保存至 `./result/<filename>_transcript.txt`

**2. 语音翻译**
- 中译英：Whisper 原生 `translate` 任务，输出中文原文 + 英文译文对照
- 英译中：Whisper 转录英文原文 + Helsinki-NLP 本地模型翻译为中文
- 结果分别保存至 `./result/<filename>_translation_zh2en.txt` / `_translation_en2zh.txt`

### `asr_micro.py` — 实时麦克风识别

- 实时捕获麦克风输入
- 内置 Silero-VAD 语音活动检测，自动判断说话起止
- 静音超过 1.5 秒自动触发 Whisper 识别
- 自动检测中英文，无需手动选择语言
- 识别结果打印至终端，同时追加保存至 `./result/micro_transcript.txt`
- 语音翻译功能预留接口（Coming Soon）

---

## ⚙️ 关键配置

### `asr_mp3.py`

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `MODEL_SIZE` | `tiny` / `small` / `medium` / `large` | 模型大小，推荐 `small`；`tiny` 精度较低 |
| `LANGUAGE` | `zh` / `en` / `None` | 指定语言；`None` 为自动检测 |
| `EN2ZH_MODEL` | 本地路径 或 HuggingFace 模型名 | 英译中翻译模型路径 |

```python
MODEL_SIZE  = "small"                   # 可换 tiny / medium / large
LANGUAGE    = None                      # zh=中文，en=英文，None=自动检测
EN2ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"
```

### `mic/recorder.py` — 麦克风设备配置

不同设备的麦克风索引和采样率不同，**首次使用前需按以下步骤确认并修改配置区**：

**步骤 1 — 查询本机可用录音设备：**
```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```
找到 `in` 数量 > 0 的设备，记下其 `index` 编号。

**步骤 2 — 查询该设备的默认采样率：**
```bash
python -c "import sounddevice as sd; print(sd.query_devices(<index>))"
```
找到 `default_samplerate` 字段的值。

**步骤 3 — 修改 `mic/recorder.py` 顶部配置区：**
```python
DEVICE_INDEX       = <步骤1 查到的 index>
DEVICE_SAMPLE_RATE = <步骤2 查到的 default_samplerate>
```

其余参数通常无需修改：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VAD_THRESHOLD` | `0.3` | Silero-VAD 置信度阈值，越高越严格 |
| `SILENCE_TIMEOUT` | `1.5` | 静音超过此秒数触发识别 |
| `FRAME_DURATION` | `0.032` | 每帧时长（秒），不低于 0.032 |

---

## 🚀 使用方法

### 环境依赖

```bash
# 核心依赖
pip install openai-whisper opencc-python-reimplemented

# 英译中翻译功能
pip install transformers sentencepiece sacremoses

# 麦克风识别功能
pip install sounddevice scipy torch torchaudio

# 系统级依赖（Ubuntu / Debian）
sudo apt install ffmpeg portaudio19-dev
```

> Silero-VAD 模型首次运行时通过 `torch.hub` 自动下载（约 1MB），需联网，之后缓存在本地离线可用。

### 运行音频文件识别 / 翻译

```bash
python asr_mp3.py ./test_audio/ZH.mp3
```

运行后按提示选择功能（转录 / 翻译）及翻译方向，结果自动保存至 `./result/`。

### 运行实时麦克风识别

```bash
python asr_micro.py
```

按 `Ctrl+C` 停止识别。

---

## 📝 输出文件命名

| 功能 | 输出文件名示例 |
|------|--------------|
| 音频转录 | `ZH_transcript.txt` |
| 音频中译英 | `ZH_translation_zh2en.txt` |
| 音频英译中 | `EN_translation_en2zh.txt` |
| 麦克风转录 | `micro_transcript.txt` |
| 麦克风翻译（预留） | `micro_translation_zh2en.txt` / `micro_translation_en2zh.txt` |

所有文件保存在 `./result/` 目录下，麦克风转录结果格式：
```
[14:23:01][zh] 今天天气很好。
[14:23:08][en] Hello, how are you?
```

---

## 🔧 常见问题排查

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `PortAudio library not found` | 缺少系统音频库 | `sudo apt install portaudio19-dev` |
| `Invalid sample rate` | 设备不支持 16000Hz | 查询设备采样率，修改 `DEVICE_SAMPLE_RATE` |
| `Input audio chunk is too short` | 帧时长不足 32ms | 确认 `FRAME_DURATION = 0.032` |
| 说话无反应 | VAD 未检测到人声 | 降低 `VAD_THRESHOLD`（如改为 `0.3`），或确认 `DEVICE_INDEX` 正确 |
| 加载 Silero-VAD 卡住 | 首次下载需联网 | 开启代理后运行一次，之后离线可用 |
| `Network is unreachable`（git push）| 网络限制 | `export https_proxy=http://127.0.0.1:7897` |
| HuggingFace 模型下载失败 | 网络限制 | `export HF_ENDPOINT=https://hf-mirror.com` |

---

## 📌 备注

- Whisper 模型首次运行时自动下载，建议网络通畅或提前手动下载
- `small` 模型在速度与精度之间取得较好平衡，推荐日常使用
- 中文转录输出默认为简体中文（opencc 自动转换）
- 英译中翻译模型（Helsinki-NLP）首次运行自动下载后缓存在本地，后续离线可用