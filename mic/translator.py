"""
mic/translator.py — 实时翻译接口模块（预留）

设计说明：
  当前版本仅定义接口函数，不实现翻译逻辑。

  为什么现在就建立此文件：
    - 保持模块结构完整，asr_micro.py 的菜单逻辑现在就可以引用此模块
    - 未来实现翻译时，只需填充函数体，主入口文件无需改动
    - 与 asr_mp3.py 的翻译方案（Whisper translate + Helsinki-NLP）保持对称

  预计实现方案（待开发）：
    - 中译英：Whisper task="translate"，对麦克风录音直接翻译
    - 英译中：Whisper 转录英文 → Helsinki-NLP opus-mt-en-zh 翻译为中文
"""


def translate_zh_to_en(audio, model) -> dict:
    """
    中译英（Coming Soon）

    参数：
      audio — float32 numpy 数组，采样率 16kHz
      model — Whisper 模型实例

    返回（预期格式）：
      {
        "original":   中文原文,
        "translated": 英文译文
      }
    """
    # TODO: 实现中译英
    # result = model.transcribe(audio, task="translate", language="zh", fp16=False)
    raise NotImplementedError("中译英功能即将推出 / Coming soon")


def translate_en_to_zh(audio, whisper_model, trans_model, tokenizer) -> dict:
    """
    英译中（Coming Soon）

    参数：
      audio         — float32 numpy 数组，采样率 16kHz
      whisper_model — Whisper 模型实例（用于转录英文原文）
      trans_model   — MarianMT 模型实例（用于英译中）
      tokenizer     — MarianMT tokenizer

    返回（预期格式）：
      {
        "original":   英文原文,
        "translated": 中文译文
      }
    """
    # TODO: 实现英译中
    # Step 1: whisper_model.transcribe(audio, language="en", fp16=False)
    # Step 2: MarianMT 翻译英文 → 中文
    raise NotImplementedError("英译中功能即将推出 / Coming soon")