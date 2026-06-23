# 🎙️ ASR Speech Recognition Project

基于 [OpenAI Whisper](https://github.com/openai/whisper) 的语音识别项目，支持音频文件识别与实时麦克风输入，兼容中英文多语言场景。

---

## 📁 项目结构

```
ASR
├── asr_mp3.py          # 音频文件识别脚本（支持翻译导出）
├── asr_micro.py        # 实时麦克风识别脚本（含语音活动检测）
├── result/             # 识别结果输出目录
└── test_audio/
    ├── ZH.mp3          # 中文测试音频
    └── EN.mp3          # 英文测试音频
```

---

## ✨ 功能简介

### `asr_mp3.py` — 音频文件识别

- 支持 `.mp3` 等常见音频格式
- 自动识别语言，或手动指定中文 / 英文
- 识别结果自动导出至 `./result` 目录
- 支持语音翻译功能

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

```python
MODEL_SIZE = "small"   # 可换 tiny / medium / large
LANGUAGE   = None      # zh=中文，en=英文，None=自动检测
```

---

## 🚀 使用方法

### 环境依赖

```bash
pip install openai-whisper
# 如使用麦克风功能，还需安装：
pip install sounddevice numpy
```

> Whisper 依赖 `ffmpeg`，请确保系统已安装：
> ```bash
> sudo apt install ffmpeg   # Ubuntu / Debian
> ```

### 运行音频文件识别

```bash
python asr_mp3.py ./test_audio/ZH.mp3
```

识别完成后，结果将保存至 `./result` 目录。

### 运行实时麦克风识别

```bash
python asr_micro.py
```

---

## 📝 输出示例

识别结果以文本文件形式保存在 `./result/` 目录下，文件名与输入音频对应。

---

## 📌 备注

- 模型首次运行时会自动下载，建议网络通畅或提前手动下载模型权重
- `small` 模型在速度与精度之间取得较好平衡，推荐日常使用
- 中文识别默认输出简体中文；如需繁体中文，可在代码中指定 `language="zh"` 并调整相关参数
