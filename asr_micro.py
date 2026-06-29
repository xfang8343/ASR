"""
asr_micro.py — 实时麦克风语音识别主入口

设计说明：
  本文件只负责：
    1. 用户交互（功能菜单选择）
    2. 加载所需模型
    3. 循环调度 recorder → transcriber / translator → saver

  不包含任何录音、识别、翻译、保存的具体实现，
  全部委托给 mic/ 包内的各功能模块。

  这样设计的好处：
    - 主入口保持简洁，逻辑一目了然
    - 各模块可独立测试，互不耦合
    - 未来 asr_core.py 统一调度时，直接 import 并调用 run() 即可

用法：
  python asr_micro.py
"""

from mic import recorder, transcriber, saver
# translator 在翻译功能实现后取消注释：
# from mic import translator


def select_function() -> str:
    """
    主功能菜单。
    同时显示中英文，照顾只懂单一语言的用户。
    """
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


def main():
    # 1. 功能选择
    func = select_function()

    # 2. 加载模型（统一在主入口加载，避免各模块重复加载）
    vad_model     = recorder.load_vad_model()
    whisper_model = transcriber.load_whisper_model()

    # 3. 执行对应功能
    if func == "transcribe":
        run_transcription(vad_model, whisper_model)

    # 翻译功能实现后在此添加：
    # elif func == "translate":
    #     run_translation(vad_model, whisper_model)


if __name__ == "__main__":
    main()