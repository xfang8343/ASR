"""
mp3/transcriber.py — Whisper 音频转录模块

设计说明：
  只负责"音频 → 文字"这一件事：
    1. 加载 Whisper 模型
    2. 对音频文件执行转录
    3. 中文输出自动转为简体（opencc）

  为什么单独抽出此模块：
    - 转录与翻译是两个独立关注点，分离后各自可独立测试和替换
    - translator.py 的中译英也需要调用转录，共用此模块避免重复代码
    - 未来若切换识别引擎（如 faster-whisper），只需修改此文件

  能做到：
    - 支持任意音频格式（mp3 / wav / m4a 等，依赖 ffmpeg）
    - 自动检测语言，或手动指定
    - 中文繁简自动转换

  做不到：
    - 实时流式识别（需完整音频文件）
    - 说话人分离
"""

import whisper
import opencc

# ────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────
MODEL_SIZE = "small"   # 可换 tiny / medium / large（tiny 精度较差）
LANGUAGE   = None      # zh=中文，en=英文，None=自动检测
# ────────────────────────────────────────────────


def load_model() -> whisper.Whisper:
    """加载 Whisper 模型，返回模型实例。"""
    print(f"[Whisper] 加载模型（{MODEL_SIZE}）...")
    model = whisper.load_model(MODEL_SIZE)
    print("[Whisper] 模型加载完成。\n")
    return model


def transcribe(model: whisper.Whisper, audio_path: str) -> dict:
    """
    对音频文件执行转录。

    参数：
      model      — Whisper 模型实例
      audio_path — 音频文件路径

    返回：
      {
        "text":     转录文本（中文已转简体）,
        "language": Whisper 检测到的语言代码（如 "zh" / "en"）
      }
    """
    print(f"[转录] 正在识别：{audio_path}")
    result = model.transcribe(
        audio_path,
        language=LANGUAGE,
        fp16=False,      # CPU 环境设为 False，避免警告
        verbose=False,
    )

    raw_text      = result["text"]
    detected_lang = result.get("language", "未知")
    print(f"      检测语言：{detected_lang}")

    # 繁体 → 简体（英文不受影响）
    print("[转录] 繁体 → 简体转换...")
    converter = opencc.OpenCC("t2s")
    text      = converter.convert(raw_text)

    return {
        "text":     text,
        "language": detected_lang,
    }


def transcribe_as_zh(model: whisper.Whisper, audio_path: str) -> str:
    """
    强制以中文转录，返回简体文本。
    供 translator.py 的中译英流程调用。
    """
    print(f"[转录] 正在转录中文原文：{audio_path}")
    result = model.transcribe(
        audio_path,
        language="zh",
        fp16=False,
        verbose=False,
    )
    converter = opencc.OpenCC("t2s")
    return converter.convert(result["text"])


def transcribe_as_en(model: whisper.Whisper, audio_path: str) -> str:
    """
    强制以英文转录，返回英文原文。
    供 translator.py 的英译中流程调用。
    """
    print(f"[转录] 正在转录英文原文：{audio_path}")
    result = model.transcribe(
        audio_path,
        language="en",
        fp16=False,
        verbose=False,
    )
    text = result["text"].strip()
    print(f"      英文原文：{text[:60]}{'...' if len(text) > 60 else ''}")
    return text