"""
mp3/translator.py — 语音翻译模块

设计说明：
  负责两个翻译方向：
    - 中译英：Whisper 原生 task="translate"，无需额外模型
    - 英译中：Whisper 转录英文原文 → Helsinki-NLP MarianMT 翻译为中文

  为什么这样设计：
    - 两个方向实现差异较大，分为两个函数，职责清晰
    - MarianMT 模型仅在英译中时加载，避免不必要的内存占用
    - transcriber.py 的转录函数被复用，不重复实现

  能做到：
    - 中译英：质量较高（Whisper 原生支持）
    - 英译中：本地离线运行（Helsinki-NLP），无需联网

  做不到：
    - 其他语言对之间的翻译（需替换模型）
    - 长文本分段翻译（当前 max_length=512 token）
"""

import whisper
from transformers import MarianMTModel, MarianTokenizer

from mp3 import transcriber

# ────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────
EN2ZH_MODEL = "Helsinki-NLP/opus-mt-en-zh"   # 英译中模型（首次运行自动下载）
# ────────────────────────────────────────────────


def _load_marian_model():
    """
    加载 Helsinki-NLP 英译中模型。
    首次运行自动从 HuggingFace 下载并缓存，后续离线可用。
    """
    print(f"[翻译] 加载翻译模型：{EN2ZH_MODEL}")
    tokenizer = MarianTokenizer.from_pretrained(EN2ZH_MODEL)
    model     = MarianMTModel.from_pretrained(EN2ZH_MODEL, use_safetensors=False)
    print("[翻译] 翻译模型加载完成。\n")
    return tokenizer, model


def translate_zh_to_en(model: whisper.Whisper, audio_path: str) -> dict:
    """
    中译英：
      Step 1 - 转录中文原文（繁简转换）
      Step 2 - Whisper task="translate" 直接输出英文译文

    参数：
      model      — Whisper 模型实例
      audio_path — 音频文件路径

    返回：
      { "original": 中文原文, "translated": 英文译文 }
    """
    # Step 1：转录中文原文
    original = transcriber.transcribe_as_zh(model, audio_path)

    # Step 2：Whisper 原生翻译为英文
    print(f"[翻译] 正在翻译为英文...")
    result = model.transcribe(
        audio_path,
        task="translate",
        language="zh",
        fp16=False,
        verbose=False,
    )
    translated = result["text"].strip()

    return {
        "original":   original,
        "translated": translated,
    }


def translate_en_to_zh(model: whisper.Whisper, audio_path: str) -> dict:
    """
    英译中：
      Step 1 - 转录英文原文
      Step 2 - MarianMT 将英文文本翻译为中文

    参数：
      model      — Whisper 模型实例
      audio_path — 音频文件路径

    返回：
      { "original": 英文原文, "translated": 中文译文 }
    """
    # Step 1：转录英文原文
    original = transcriber.transcribe_as_en(model, audio_path)

    # Step 2：MarianMT 翻译为中文
    tokenizer, trans_model = _load_marian_model()
    inputs = tokenizer(
        [original],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    translated_ids = trans_model.generate(**inputs)
    translated     = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    return {
        "original":   original,
        "translated": translated,
    }