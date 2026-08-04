"""
asr_mp3.py — 音频语音识别 & 语音翻译主入口

设计说明：
  本文件只负责：
    1. 用户交互（功能菜单、翻译方向选择）
    2. 独立运行时自行加载 Whisper 模型
    3. 调度 mp3/ 包内各功能模块

  与 asr_core.py 的关系：
    - 独立运行：python asr_mp3.py ./test_audio/ZH.mp3
      → main() 自行加载模型，调用 run()
    - 通过 asr_core.py 调用：
      → asr_core.py 统一加载模型后，直接调用 run(model, audio_path)
      → 模型只加载一次，无需重复等待

用法：
  python asr_mp3.py ./test_audio/ZH.mp3
"""

import os
import sys

from mp3 import transcriber, translator, saver


def select_function() -> str:
    """主功能菜单，同时显示中英文。"""
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
    """翻译方向选择菜单，同时显示中英文。"""
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


def _get_audio_path() -> str:
    """从命令行参数获取音频路径，未指定时使用默认值。"""
    if len(sys.argv) >= 2:
        return sys.argv[1]
    audio_path = "test.mp3"
    print(f"[提示] 未指定音频文件，使用默认：{audio_path}")
    print(f"       用法：python asr_mp3.py 你的音频.mp3\n")
    return audio_path


def run(model, audio_path: str):
    """
    音频识别 & 翻译核心流程。

    参数：
      model      — 已加载的 Whisper 模型实例
      audio_path — 音频文件路径

    设计说明：
      将核心逻辑抽为独立函数，使 asr_core.py 可以直接调用，
      同时 main() 独立运行时也调用此函数，避免代码重复。
    """
    # 1. 选择功能
    func_choice = select_function()

    # 2. 执行对应功能
    if func_choice == "1":
        print("\n[INFO] 已选择：语音转录\n")
        result    = transcriber.transcribe(model, audio_path)
        save_path = saver.save_transcription(
            result["text"], result["language"], audio_path
        )
        print(f"\n{'─' * 40}")
        print(f"转录完成！结果已保存到：{save_path}")
        print(f"{'─' * 40}")

    elif func_choice == "2":
        print("\n[INFO] 已选择：语音翻译\n")
        direction = select_translation_direction()

        if direction == "zh2en":
            print("\n[INFO] 翻译方向：中文 → 英文\n")
            result = translator.translate_zh_to_en(model, audio_path)
        else:
            print("\n[INFO] 翻译方向：英文 → 中文\n")
            result = translator.translate_en_to_zh(model, audio_path)

        save_path = saver.save_translation(
            result["original"], result["translated"], direction, audio_path
        )
        print(f"\n{'─' * 40}")
        print(f"翻译完成！结果已保存到：{save_path}")
        print(f"{'─' * 40}")


def main():
    # 1. 获取音频路径
    audio_path = _get_audio_path()
    if not os.path.exists(audio_path):
        print(f"[错误] 找不到音频文件：{audio_path}")
        sys.exit(1)

    # 2. 独立运行时自行加载模型
    model = transcriber.load_model()

    # 3. 执行核心流程
    run(model, audio_path)


if __name__ == "__main__":
    main()