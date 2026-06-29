"""
mic/recorder.py — 麦克风录音 + Silero-VAD 语音活动检测模块

设计说明：
  将"录音"与"识别"解耦。本模块只负责：
    1. 从麦克风持续采集音频帧
    2. 用 Silero-VAD 判断每帧是否有人声
    3. 检测到静音超过阈值时，将完整语音片段以 numpy 数组返回

  优点：
    - 上层模块（transcriber / translator）无需关心录音细节
    - VAD 参数（静音阈值、采样率等）集中在此文件配置，易于调整
    - 返回 numpy 数组，可直接传入 Whisper，无需落盘临时文件

  局限：
    - 当前仅支持单声道 16kHz 输入（Whisper & Silero-VAD 要求）
    - 不支持多说话人区分（说话人分离需额外模型）
"""

import numpy as np
import sounddevice as sd
import torch

# ────────────────────────────────────────────────
# 录音配置
# ────────────────────────────────────────────────
SAMPLE_RATE    = 16000   # 采样率，Whisper 和 Silero-VAD 均要求 16kHz
CHANNELS       = 1       # 单声道
FRAME_DURATION = 0.032    # 每帧时长（秒），32ms
SILENCE_TIMEOUT = 1.5    # 静音超过此秒数则认为说话结束，触发识别
VAD_THRESHOLD  = 0.3     # Silero-VAD 置信度阈值（0~1），越高越严格
# ────────────────────────────────────────────────

FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)   # 每帧采样点数


def load_vad_model():
    """
    加载 Silero-VAD 模型。

    为什么选 Silero-VAD：
      - 基于神经网络，对噪音环境鲁棒性优于传统能量阈值法
      - 模型体积小（~1MB），推理快，适合实时场景
      - 项目已依赖 torch，无需额外引入新框架
    """
    print("[VAD] 加载 Silero-VAD 模型...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    print("[VAD] 模型加载完成。\n")
    return model


def is_speech(frame: np.ndarray, vad_model) -> bool:
    """
    判断单帧音频是否包含人声。

    参数：
      frame     — 形状为 (FRAME_SIZE,) 的 float32 numpy 数组
      vad_model — Silero-VAD 模型实例

    返回：
      True = 有人声，False = 静音 / 噪音
    """

    # 归一化到 -1~1，Silero-VAD 要求
    max_val = np.abs(frame).max()
    if max_val > 0:
        frame = frame / max_val
    
    tensor = torch.from_numpy(frame).float()
    confidence = vad_model(tensor, SAMPLE_RATE).item()
    return confidence >= VAD_THRESHOLD

def record_once(vad_model) -> np.ndarray | None:
    """
    录制一段完整的语音片段（从检测到人声开始，到静音超时结束）。

    流程：
      1. 持续读取麦克风帧
      2. VAD 检测到人声 → 开始累积音频
      3. 静音超过 SILENCE_TIMEOUT 秒 → 停止，返回完整片段

    返回：
      numpy float32 数组（可直接传入 Whisper），
      若用户按 Ctrl+C 则返回 None
    """
    print("[录音] 等待说话中... （按 Ctrl+C 退出）")

    audio_buffer  = []   # 累积的语音帧
    silence_frames = 0   # 连续静音帧计数
    speech_started = False

    # 静音超时对应的帧数
    max_silence_frames = int(SILENCE_TIMEOUT / FRAME_DURATION)

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SIZE,
        ) as stream:
            while True:
                frame, _ = stream.read(FRAME_SIZE)
                frame = frame.flatten()   # (FRAME_SIZE, 1) → (FRAME_SIZE,)

                if is_speech(frame, vad_model):
                    if not speech_started:
                        print("[录音] 检测到人声，开始录制...")
                        speech_started = True
                    audio_buffer.append(frame)
                    silence_frames = 0
                else:
                    if speech_started:
                        audio_buffer.append(frame)   # 保留少量静音，避免截断尾音
                        silence_frames += 1
                        if silence_frames >= max_silence_frames:
                            print("[录音] 检测到停顿，识别中...\n")
                            break

    except KeyboardInterrupt:
        return None

    if not audio_buffer:
        return None

    return np.concatenate(audio_buffer, axis=0)