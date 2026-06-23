语音识别（openai-whisper）
asr_mp3.py
MODEL_SIZE   = "small"          # 可换 tiny/small / medium / large(tiny模型效果较差)
LANGUAGE     = None            # zh=中文，en=英文，None=自动检测    原音频语言类型选择
脚本使用方法：python asr_mp3.py ./test_audio/ZH.mp3     ##音频路径
脚本作用：
1、音频识别，识别文字内容导出至./result
2、语音翻译


asr_micro.py脚本作用：
口语识别和语音活动检测


./test_audio中为ZH、EN音频