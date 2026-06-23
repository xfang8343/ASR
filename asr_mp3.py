import whisper
import opencc
import os
import sys


MODEL_SIZE   = "small"          # 可换 small / medium / large
LANGUAGE     = None            # zh=中文，en=英文，None=自动检测
OUTPUT_DIR   = "./result"      
OUTPUT_FILE  = "asr_mp3_2.txt"      #修改为原音频文件名称_result.txt



def transcribe(audio_path: str) -> str:
    """加载模型并识别音频，返回简体中文文本。"""

    if not os.path.exists(audio_path):
        print(f"[错误] 找不到音频文件：{audio_path}")
        sys.exit(1)

    print(f"[1/3] 加载 Whisper {MODEL_SIZE} 模型...")
    model = whisper.load_model(MODEL_SIZE)

    print(f"[2/3] 正在识别：{audio_path}")
    result = model.transcribe(
        audio_path,
        language=LANGUAGE,
        fp16=False,          # CPU 环境设为 False，避免警告
        verbose=False,
    )

    raw_text = result["text"]
    detected_lang = result.get("language", "未知")
    print(f"      检测语言：{detected_lang}")

    print("[3/3] 繁体 → 简体转换...")
    converter = opencc.OpenCC("t2s")   # t2s = Traditional to Simplified
    simplified = converter.convert(raw_text)

    return simplified


def save_result(text: str) -> str:
    """将识别结果保存到文件，返回保存路径。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(text)
    return save_path


def main():
    # 支持命令行传入音频路径，默认用 test.mp3
    if len(sys.argv) >= 2:
        audio_path = sys.argv[1]
    else:
        audio_path = "test.mp3"
        print(f"[提示] 未指定音频文件，使用默认：{audio_path}")
        print(f"       用法：python asr_mp3.py 你的音频.mp3\n")

    text = transcribe(audio_path)
    save_path = save_result(text)

    print(f"\n{'─'*40}")
    print(f"识别完成！结果已保存到：{save_path}")
    print(f"{'─'*40}")
    # print(text)


if __name__ == "__main__":
    main()
