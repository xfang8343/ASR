"""
mic/transcriber.py — Whisper 实时转录模块

设计说明：
  接收 recorder.py 返回的 numpy 音频数组，调用 Whisper 完成转录。

  为什么单独抽出此模块：
    - 录音与识别是两个独立关注点，分离后各自可独立测试和替换
    - 未来若切换识别引擎（如 faster-whisper），只需修改此文件
    - transcriber 与 translator 平行，便于 asr_micro.py 统一调度

  能做到：
    - 自动检测语言（中文 / 英文均可），无需用户手动选择
    - 中文输出自动转为简体（opencc）
    - 直接接受内存中的 numpy 数组，无需写临时文件

  做不到：
    - 实时逐字输出（Whisper 是离线模型，需完整音频才能识别）
    - 说话人分离
"""

import whisper
import opencc
import numpy as np

# ────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────
MODEL_SIZE = "small"   # 可换 tiny / medium / large
# ────────────────────────────────────────────────


def load_whisper_model() -> whisper.Whisper:
    """加载 Whisper 模型，返回模型实例。"""
    print(f"[Whisper] 加载模型（{MODEL_SIZE}）...")
    model = whisper.load_model(MODEL_SIZE)
    print("[Whisper] 模型加载完成。\n")
    return model


def transcribe(audio: np.ndarray, model: whisper.Whisper) -> dict:
    """
    对一段音频执行转录。

    参数：
      audio — float32 numpy 数组，采样率 16kHz
      model — Whisper 模型实例

    返回：
      {
        "text":     转录文本（中文已转简体）,
        "language": Whisper 检测到的语言代码（如 "zh" / "en"）
      }

    为什么自动检测语言：
      - 麦克风输入场景下用户可能随时切换语言
      - 省去用户手动选择的步骤，降低使用门槛
      - 代价是每段音频多一次语言检测，略微增加延迟（通常 <0.5s）
    """
    result = model.transcribe(
        audio,
        fp16=False,      # CPU 环境避免警告
        verbose=False,
    )

    raw_text      = result["text"].strip()
    detected_lang = result.get("language", "unknown")

    # 中文输出转简体（英文不受影响）
    converter = opencc.OpenCC("t2s")
    text      = converter.convert(raw_text)

    return {
        "text":     text,
        "language": detected_lang,
    }