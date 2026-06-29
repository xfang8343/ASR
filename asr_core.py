"""
asr_core.py — ASR 项目统一主入口

设计说明：
  本文件负责：
    1. 顶层功能菜单（音频文件识别 / 实时麦克风识别）
    2. 统一加载 Whisper 模型（只加载一次，传入各功能模块复用）
    3. 按用户选择调用 asr_mp3.run() 或 asr_micro.run()

  为什么统一加载模型：
    - Whisper 模型加载耗时约 10~30 秒
    - 若每个子脚本各自加载，切换功能时需重复等待
    - 由 asr_core.py 加载一次后传入，切换功能无需重新加载

  与子脚本的关系：
    - asr_mp3.py  / asr_micro.py 均可独立运行，自行加载模型
    - 通过 asr_core.py 运行时，模型由此处统一加载并传入

用法：
  python asr_core.py
  python asr_core.py ./test_audio/ZH.mp3   # 预先指定音频路径
"""

import sys

import asr_mp3
import asr_micro
from mp3 import transcriber
from mic import recorder
from mic import transcriber as mic_transcriber


def select_mode() -> str:
    """
    顶层功能菜单，同时显示中英文。
    返回 "mp3" 或 "micro"。
    """
    print("\n" + "=" * 50)
    print("请选择功能 / Please select a function:")
    print("  1. 音频文件识别   Audio File Recognition  (asr_mp3)")
    print("  2. 实时麦克风识别 Live Microphone Recognition (asr_micro)")
    print("=" * 50)

    while True:
        choice = input("请输入编号 / Enter number (1 or 2): ").strip()
        if choice == "1":
            return "mp3"
        if choice == "2":
            return "micro"
        print("[提示 / Hint] 请输入 1 或 2 / Please enter 1 or 2.\n")


def main():
    print("=" * 50)
    print("  🎙️  ASR Speech Recognition Project")
    print("=" * 50)

    # 1. 选择顶层功能
    mode = select_mode()

    # 2. 统一加载 Whisper 模型（两个功能共用同一份模型）
    print("\n[Core] 加载 Whisper 模型...")
    whisper_model = transcriber.load_model()

    # 3. 按模式调用对应子脚本的 run()
    if mode == "mp3":
        # 获取音频路径（支持命令行预先指定）
        if len(sys.argv) >= 2:
            audio_path = sys.argv[1]
        else:
            audio_path = input("\n请输入音频文件路径 / Enter audio file path: ").strip()

        import os
        if not os.path.exists(audio_path):
            print(f"[错误] 找不到音频文件：{audio_path}")
            sys.exit(1)

        print()
        asr_mp3.run(whisper_model, audio_path)

    elif mode == "micro":
        # 额外加载 VAD 模型（仅麦克风模式需要）
        print("[Core] 加载 Silero-VAD 模型...")
        vad_model = recorder.load_vad_model()

        print()
        asr_micro.run(whisper_model, vad_model)


if __name__ == "__main__":
    main()