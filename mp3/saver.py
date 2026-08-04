"""
mp3/saver.py — 识别结果保存模块

设计说明：
  将"保存逻辑"从主入口和功能模块中剥离，统一管理。

  优点：
    - 文件命名规则、输出目录集中在此配置，修改时只需改一处
    - 转录与翻译结果使用不同文件名后缀，互不干扰
    - 与 mic/saver.py 保持对称的设计风格

  输出文件命名规则：
    转录：  <音频文件名>_transcript.txt
    中译英：<音频文件名>_translation_zh2en.txt
    英译中：<音频文件名>_translation_en2zh.txt
"""

import os

# ────────────────────────────────────────────────
# 配置
# ────────────────────────────────────────────────
OUTPUT_DIR = "./result"
# ────────────────────────────────────────────────


def _get_output_path(audio_path: str, suffix: str) -> str:
    """根据音频文件名生成输出路径，如 ./result/ZH_transcript.txt"""
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, f"{base_name}_{suffix}.txt")


def save_transcription(text: str, language: str, audio_path: str) -> str:
    """
    保存转录结果，返回保存路径。

    参数：
      text       — 转录文本
      language   — Whisper 检测到的语言代码
      audio_path — 原始音频路径（用于生成文件名）
    """
    save_path = _get_output_path(audio_path, "transcript")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"【音频文件】\n{audio_path}\n\n")
        f.write(f"【检测语言】\n{language}\n\n")
        f.write("【转录结果】\n")
        f.write(text + "\n")

    return save_path


def save_translation(original: str, translated: str,
                     direction: str, audio_path: str) -> str:
    """
    保存翻译结果（原文 + 译文对照），返回保存路径。

    参数：
      original   — 原文文本
      translated — 译文文本
      direction  — 翻译方向，"zh2en" 或 "en2zh"
      audio_path — 原始音频路径（用于生成文件名）
    """
    save_path = _get_output_path(audio_path, f"translation_{direction}")

    # 根据方向决定标签
    if direction == "zh2en":
        orig_label = "【中文原文 / Original Chinese】"
        trans_label = "【英文译文 / English Translation】"
    else:
        orig_label = "【英文原文 / Original English】"
        trans_label = "【中文译文 / Chinese Translation】"

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"【音频文件 / Audio File】\n{audio_path}\n\n")
        f.write(f"{orig_label}\n{original}\n\n")
        f.write(f"{trans_label}\n{translated}\n")

    return save_path