# 🎙️ ASR Speech Recognition Project

基于 [OpenAI Whisper](https://github.com/openai/whisper) 的语音识别项目，支持音频文件识别与实时麦克风输入，兼容中英文多语言场景。

---

## 📁 项目结构

```
ASR/
├── asr_mp3.py          # 音频文件识别脚本（支持转录与翻译）
├── asr_micro.py        # 实时麦克风识别脚本（含语音活动检测）
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
- 内置语音活动检测（VAD），自动判断说话起止
- 适用于口语识别场景

---

## ⚙️ 关键配置（`asr_mp3.py`）

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `MODEL_SIZE` | `tiny` / `small` / `medium` / `large` | 模型大小，推荐 `small`；`tiny` 精度较低 |
| `LANGUAGE` | `zh` / `en` / `None` | 指定语言；`None` 为自动检测 |
| `EN2ZH_MODEL` | 本地路径 或 HuggingFace 模型名 | 英译中翻译模型路径 |

```python
MODEL_SIZE  = "small"                        # 可换 tiny / medium / large
LANGUAGE    = None                           # zh=中文，en=英文，None=自动检测
EN2ZH_MODEL = "./models/opus-mt-en-zh"      # 英译中本地模型路径
```

---

## 🚀 使用方法

### 环境依赖

```bash
pip install openai-whisper opencc-python-reimplemented
pip install transformers sentencepiece   # 英译中功能需要

# 如使用麦克风功能，还需安装：
pip install sounddevice numpy
```

> Whisper 依赖 `ffmpeg`，请确保系统已安装：
> ```bash
> sudo apt install ffmpeg   # Ubuntu / Debian
> ```

### 英译中模型下载（首次使用）

`Helsinki-NLP/opus-mt-en-zh` 模型需提前下载至本地，之后完全离线运行：

> 网络受限时可设置镜像：`export HF_ENDPOINT=https://hf-mirror.com`

### 运行音频文件识别 / 翻译

```bash
python asr_mp3.py ./test_audio/ZH.mp3
```

运行后按提示选择功能（转录 / 翻译）及翻译方向，结果自动保存至 `./result/`。

### 运行实时麦克风识别

```bash
python asr_micro.py
```

---

## 📝 输出文件命名

| 功能 | 输出文件名示例 |
|------|--------------|
| 语音转录 | `ZH_transcript.txt` |
| 中译英 | `ZH_translation_zh2en.txt` |
| 英译中 | `EN_translation_en2zh.txt` |

所有文件保存在 `./result/` 目录下，内容包含音频路径、原文及译文对照。

---

## 📌 备注

- Whisper 模型首次运行时自动下载，建议网络通畅或提前手动下载
- `small` 模型在速度与精度之间取得较好平衡，推荐日常使用
- 中文转录输出默认为简体中文（opencc 自动转换）