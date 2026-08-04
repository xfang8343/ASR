"""
mic/saver.py — 识别结果保存模块

设计说明：
  将"保存逻辑"从主入口和识别模块中剥离，单独管理。

  优点：
    - 文件命名规则、输出目录等集中在此配置，修改时只需改一处
    - 追加写入（append）模式：同一次运行的多段识别结果写入同一文件，
      方便回顾完整对话内容
    - 转录与翻译结果使用不同文件名，互不干扰
"""

import os
from datetime import datetime

# ────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────
OUTPUT_DIR = "./result"
# ────────────────────────────────────────────────


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def save_transcript(text: str, language: str) -> str:
    """
    追加保存转录结果至 micro_transcript.txt。

    参数：
      text     — 转录文本
      language — Whisper 检测到的语言代码

    返回：
      保存路径
    """
    _ensure_dir()
    save_path = os.path.join(OUTPUT_DIR, "micro_transcript.txt")

    with open(save_path, "a", encoding="utf-8") as f:
        f.write(f"[{_timestamp()}][{language}] {text}\n")

    return save_path


def save_translation(original: str, translated: str, direction: str) -> str:
    """
    追加保存翻译结果（原文 + 译文对照）。

    参数：
      original   — 原文文本
      translated — 译文文本
      direction  — 翻译方向，"zh2en" 或 "en2zh"

    返回：
      保存路径
    """
    _ensure_dir()
    filename  = f"micro_translation_{direction}.txt"
    save_path = os.path.join(OUTPUT_DIR, filename)

    with open(save_path, "a", encoding="utf-8") as f:
        f.write(f"[{_timestamp()}]\n")
        f.write(f"  原文 / Original  : {original}\n")
        f.write(f"  译文 / Translated : {translated}\n\n")

    return save_path