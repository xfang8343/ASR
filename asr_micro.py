"""
asr_micro.py — 实时麦克风语音识别主入口

设计说明：
  本文件只负责：
    1. 用户交互（功能菜单选择）
    2. 独立运行时自行加载所需模型
    3. 循环调度 recorder → transcriber / translator → saver

  与 asr_core.py 的关系：
    - 独立运行：python asr_micro.py
      → main() 自行加载模型，调用 run()
    - 通过 asr_core.py 调用：
      → asr_core.py 统一加载模型后，直接调用 run(whisper_model, vad_model)
      → 模型只加载一次，无需重复等待

用法：
  python asr_micro.py
"""

from mic import recorder, transcriber, saver
# translator 在翻译功能实现后取消注释：
# from mic import translator


def select_function() -> str:
    """主功能菜单，同时显示中英文。"""
    print("=" * 50)
    print("请选择功能 / Please select a function:")
    print("  1. 实时语音转录   Live Transcription")
    print("  2. 实时语音翻译   Live Translation  （Coming Soon）")
    print("=" * 50)

    while True:
        choice = input("请输入编号 / Enter number (1): ").strip()
        if choice == "1":
            return "transcribe"
        if choice == "2":
            print("\n[提示] 翻译功能即将推出，敬请期待。/ Coming soon.\n")
        else:
            print("[提示 / Hint] 请输入 1 / Please enter 1.\n")


def run_transcription(vad_model, whisper_model):
    """
    转录主循环：持续录音 → 识别 → 打印 → 保存，直到用户按 Ctrl+C。

    循环设计说明：
      每次 recorder.record_once() 返回一段完整语音片段，
      交给 transcriber.transcribe() 识别后立即输出并追加写入文件。
      下一轮立即开始等待新的语音输入，实现连续识别。
    """
    print("\n[INFO] 实时转录已启动，按 Ctrl+C 停止。\n")
    save_path = None

    try:
        while True:
            # 1. 录音（VAD 自动判断起止）
            audio = recorder.record_once(vad_model)
            if audio is None:
                break

            # 2. Whisper 识别
            result = transcriber.transcribe(audio, whisper_model)
            text   = result["text"]
            lang   = result["language"]

            # 3. 打印到终端
            print(f"[{lang}] {text}\n")

            # 4. 追加保存
            save_path = saver.save_transcript(text, lang)

    except KeyboardInterrupt:
        pass

    if save_path:
        print(f"\n[INFO] 转录结果已保存至：{save_path}")


def run(whisper_model, vad_model):
    """
    麦克风识别核心流程。

    参数：
      whisper_model — 已加载的 Whisper 模型实例
      vad_model     — 已加载的 Silero-VAD 模型实例

    设计说明：
      将核心逻辑抽为独立函数，使 asr_core.py 可以直接调用，
      同时 main() 独立运行时也调用此函数，避免代码重复。
    """
    func = select_function()

    if func == "transcribe":
        run_transcription(vad_model, whisper_model)

    # 翻译功能实现后在此添加：
    # elif func == "translate":
    #     run_translation(vad_model, whisper_model)


def main():
    # 1. 独立运行时自行加载模型
    vad_model     = recorder.load_vad_model()
    whisper_model = transcriber.load_whisper_model()

    # 2. 执行核心流程
    run(whisper_model, vad_model)


if __name__ == "__main__":
    main()