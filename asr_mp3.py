"""
asr_mp3.py — 音频语音识别 & 语音翻译脚本
支持功能：
  1. 语音转录：将音频转为简体中文文字，保存至 ./result/<filename>_transcript.txt
  2. 语音翻译
       中译英：转录中文原文 + 翻译为英文，保存至 ./result/<filename>_translation_zh2en.txt
       英译中：转录英文原文 + 翻译为中文，保存至 ./result/<filename>_translation_en2zh.txt

用法：
  python asr_mp3.py ./test_audio/ZH.mp3
"""

import whisper
import opencc
import os
import sys

from transformers import MarianMTModel, MarianTokenizer


# ────────────────────────────────────────────────
# 配置区
# ────────────────────────────────────────────────
MODEL_SIZE      = "small"              # 可换 tiny / medium / large（tiny 精度较差）
LANGUAGE        = None                 # zh=中文，en=英文，None=自动检测
OUTPUT_DIR      = "./result"
EN2ZH_MODEL     = "Helsinki-NLP/opus-mt-en-zh"   # 英译中本地模型
# ────────────────────────────────────────────────


def get_output_filename(audio_path: str, suffix: str) -> str:
    """根据音频文件名生成输出文件名，如 ZH_transcript.txt"""
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    return f"{base_name}_{suffix}.txt"


def load_whisper_model() -> whisper.Whisper:
    print(f"[1/?] 加载 Whisper {MODEL_SIZE} 模型...")
    model = whisper.load_model(MODEL_SIZE)
    print("      模型加载完成。\n")
    return model


def load_translation_model(model_name: str):
    """加载 Helsinki-NLP 翻译模型（首次运行自动下载）。"""
    print(f"[加载] 翻译模型：{model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model     = MarianMTModel.from_pretrained(model_name, use_safetensors=False)
    print("      翻译模型加载完成。\n")
    return tokenizer, model


def select_function() -> str:
    """让用户选择功能，同时提供中英文提示。"""
    print("=" * 50)
    print("请选择功能 / Please select a function:")
    print("  1. 语音转录   Speech Transcription")
    print("  2. 语音翻译   Speech Translation")
    print("=" * 50)

    while True:
        choice = input("请输入编号 / Enter number (1 or 2): ").strip()
        if choice in ("1", "2"):
            return choice
        print("[提示 / Hint] 请输入 1 或 2 / Please enter 1 or 2.\n")


def select_translation_direction() -> str:
    """让用户选择翻译方向，中英文双语提示。"""
    print("\n" + "=" * 50)
    print("请选择翻译方向 / Please select translation direction:")
    print("  1. 中文 → 英文   Chinese → English")
    print("  2. 英文 → 中文   English → Chinese")
    print("=" * 50)

    while True:
        choice = input("请输入编号 / Enter number (1 or 2): ").strip()
        if choice == "1":
            return "zh2en"
        if choice == "2":
            return "en2zh"
        print("[提示 / Hint] 请输入 1 或 2 / Please enter 1 or 2.\n")


# ── 功能 1：语音转录 ──────────────────────────────

def transcribe(model: whisper.Whisper, audio_path: str) -> str:
    """识别音频，返回简体中文文本。"""
    print(f"[2/?] 正在识别：{audio_path}")
    result = model.transcribe(
        audio_path,
        language=LANGUAGE,
        fp16=False,      # CPU 环境设为 False，避免警告
        verbose=False,
    )

    raw_text = result["text"]
    detected_lang = result.get("language", "未知")
    print(f"      检测语言：{detected_lang}")

    print("[3/?] 繁体 → 简体转换...")
    converter = opencc.OpenCC("t2s")   # t2s = Traditional to Simplified
    simplified = converter.convert(raw_text)

    return simplified


def save_transcription(text: str, audio_path: str) -> str:
    """保存转录结果，返回保存路径。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename  = get_output_filename(audio_path, "transcript")
    save_path = os.path.join(OUTPUT_DIR, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"【音频文件】\n{audio_path}\n\n")
        f.write("【转录结果】\n")
        f.write(text + "\n")

    return save_path


# ── 功能 2a：语音翻译（中译英）───────────────────

def translate_zh_to_en(model: whisper.Whisper, audio_path: str) -> tuple:
    """
    中译英（Whisper 原生支持）：
      Step 1 - transcribe(language="zh") 获取中文原文，并转为简体
      Step 2 - transcribe(task="translate", language="zh") 获取英文译文
    """
    print(f"[2/?] 正在转录中文原文：{audio_path}")
    result_zh = model.transcribe(
        audio_path,
        language="zh",
        fp16=False,
        verbose=False,
    )
    raw_zh = result_zh["text"]

    print("[3/?] 繁体 → 简体转换...")
    converter    = opencc.OpenCC("t2s")
    original_text = converter.convert(raw_zh)

    print(f"[4/?] 正在翻译为英文：{audio_path}")
    result_en = model.transcribe(
        audio_path,
        task="translate",
        language="zh",
        fp16=False,
        verbose=False,
    )
    translated_text = result_en["text"].strip()

    return original_text, translated_text


def save_translation_zh2en(original: str, translated: str, audio_path: str) -> str:
    """保存中译英结果（原文 + 译文对照），返回保存路径。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename  = get_output_filename(audio_path, "translation_zh2en")
    save_path = os.path.join(OUTPUT_DIR, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"【音频文件 / Audio File】\n{audio_path}\n\n")
        f.write("【英文原文 / Original English】\n")
        f.write(original + "\n\n")
        f.write("【中文译文 / Chinese Translation】\n")
        f.write(translated + "\n")

    return save_path


# ── 功能 2b：语音翻译（英译中）───────────────────

def translate_en_to_zh(whisper_model: whisper.Whisper, audio_path: str) -> tuple:
    """
    英译中（Helsinki-NLP 本地模型）：
      Step 1 - Whisper transcribe(language="en") 获取英文原文
      Step 2 - MarianMT 将英文文本翻译为中文
    """
    print(f"[2/?] 正在转录英文原文：{audio_path}")
    result_en = whisper_model.transcribe(
        audio_path,
        language="en",
        fp16=False,
        verbose=False,
    )
    original_text = result_en["text"].strip()
    print(f"      英文原文：{original_text[:60]}{'...' if len(original_text) > 60 else ''}")

    print(f"[3/?] 加载翻译模型并翻译为中文...")
    tokenizer, trans_model = load_translation_model(EN2ZH_MODEL)

    inputs = tokenizer(
        [original_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    translated_ids  = trans_model.generate(**inputs)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    return original_text, translated_text


def save_translation_en2zh(original: str, translated: str, audio_path: str) -> str:
    """保存英译中结果（原文 + 译文对照），返回保存路径。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename  = get_output_filename(audio_path, "translation_en2zh")
    save_path = os.path.join(OUTPUT_DIR, filename)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"【音频文件 / Audio File】\n{audio_path}\n\n")
        f.write("【英文原文 / Original English】\n")
        f.write(original + "\n\n")
        f.write("【中文译文 / Chinese Translation】\n")
        f.write(translated + "\n")

    return save_path


# ── 主程序 ────────────────────────────────────────

def main():
    # 1. 检查音频路径
    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]
    else:
        audio_path = "test.mp3"
        print(f"[提示] 未指定音频文件，使用默认：{audio_path}")
        print(f"       用法：python asr_mp3.py 你的音频.mp3\n")

    if not os.path.exists(audio_path):
        print(f"[错误] 找不到音频文件：{audio_path}")
        sys.exit(1)

    # 2. 加载 Whisper 模型
    whisper_model = load_whisper_model()

    # 3. 选择功能
    func_choice = select_function()

    # 4. 执行对应功能
    if func_choice == "1":
        print("\n[INFO] 已选择：语音转录\n")
        text      = transcribe(whisper_model, audio_path)
        save_path = save_transcription(text, audio_path)

        print(f"\n{'─' * 40}")
        print(f"转录完成！结果已保存到：{save_path}")
        print(f"{'─' * 40}")

    elif func_choice == "2":
        print("\n[INFO] 已选择：语音翻译\n")
        direction = select_translation_direction()

        if direction == "zh2en":
            print("\n[INFO] 翻译方向：中文 → 英文\n")
            original, translated = translate_zh_to_en(whisper_model, audio_path)
            save_path = save_translation_zh2en(original, translated, audio_path)

            print(f"\n{'─' * 40}")
            print(f"翻译完成！结果已保存到：{save_path}")
            print(f"{'─' * 40}")

        elif direction == "en2zh":
            print("\n[INFO] 翻译方向：英文 → 中文\n")
            original, translated = translate_en_to_zh(whisper_model, audio_path)
            save_path = save_translation_en2zh(original, translated, audio_path)

            print(f"\n{'─' * 40}")
            print(f"翻译完成！结果已保存到：{save_path}")
            print(f"{'─' * 40}")


if __name__ == "__main__":
    main()