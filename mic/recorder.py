"""
mic/recorder.py — 麦克风录音 + Silero-VAD 语音活动检测模块

设计说明：
  将"录音"与"识别"解耦。本模块只负责：
    1. 以设备原生采样率从麦克风采集音频帧
    2. 重采样至 16000Hz（Whisper & Silero-VAD 要求）
    3. 用 Silero-VAD 判断每帧是否有人声
    4. 检测到静音超过阈值时，将完整语音片段以 numpy 数组返回

  优点：
    - 上层模块无需关心录音和采样率细节
    - VAD 参数集中配置，易于调整
    - 返回 16000Hz numpy 数组，可直接传入 Whisper，无需落盘临时文件

  局限：
    - 当前仅支持单声道输入
    - 不支持多说话人区分
"""

import numpy as np
import sounddevice as sd
import torch
from scipy.signal import resample_poly
from math import gcd


# ════════════════════════════════════════════════
# 【用户配置区】
# 如果运行时提示设备错误或没有声音，请按以下步骤排查和修改：
#
# 步骤 1 — 查询本机可用录音设备：
#   python -c "import sounddevice as sd; print(sd.query_devices())"
#   找到 "in" 数量 > 0 的设备，记下其 index 编号
#
# 步骤 2 — 查询该设备的默认采样率：
#   python -c "import sounddevice as sd; print(sd.query_devices(<index>))"
#   找到 "default_samplerate" 字段的值
#
# 步骤 3 — 修改下方两个参数：
#   DEVICE_INDEX       = <步骤1 查到的 index>
#   DEVICE_SAMPLE_RATE = <步骤2 查到的 default_samplerate>
#
# 示例（开发机环境）：
#   设备列表中 index=10 的 pipewire 设备，default_samplerate=44100.0
#   → DEVICE_INDEX = 10
#   → DEVICE_SAMPLE_RATE = 44100
# ════════════════════════════════════════════════
DEVICE_INDEX       = 10      # 录音设备索引，按上方步骤修改
DEVICE_SAMPLE_RATE = 44100   # 设备原生采样率（Hz），按上方步骤修改
# ════════════════════════════════════════════════

# ────────────────────────────────────────────────
# 以下参数通常无需修改
# ────────────────────────────────────────────────
TARGET_SAMPLE_RATE = 16000   # Whisper & Silero-VAD 要求的采样率，固定值
CHANNELS           = 1       # 单声道
FRAME_DURATION     = 0.032   # 每帧时长（秒），32ms（Silero-VAD 最小要求）
SILENCE_TIMEOUT    = 1.5     # 静音超过此秒数则认为说话结束
VAD_THRESHOLD      = 0.3     # Silero-VAD 置信度阈值（0~1），越高越严格
# ────────────────────────────────────────────────

# 每帧在设备采样率下的采样点数
FRAME_SIZE_DEVICE = int(DEVICE_SAMPLE_RATE * FRAME_DURATION)

# 重采样比例（最简分数）
_g = gcd(TARGET_SAMPLE_RATE, DEVICE_SAMPLE_RATE)
RESAMPLE_UP   = TARGET_SAMPLE_RATE // _g
RESAMPLE_DOWN = DEVICE_SAMPLE_RATE // _g


def _resample(frame: np.ndarray) -> np.ndarray:
    """将音频帧从 DEVICE_SAMPLE_RATE 重采样至 TARGET_SAMPLE_RATE。"""
    return resample_poly(frame, RESAMPLE_UP, RESAMPLE_DOWN).astype(np.float32)


def load_vad_model():
    """
    加载 Silero-VAD 模型。

    为什么选 Silero-VAD：
      - 基于神经网络，对噪音环境鲁棒性优于传统能量阈值法
      - 模型体积小（~1MB），推理快，适合实时场景
      - 项目已依赖 torch，无需额外引入新框架
    """
    print("[VAD] 加载 Silero-VAD 模型...")
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    print("[VAD] 模型加载完成。\n")
    return model


def is_speech(frame: np.ndarray, vad_model) -> bool:
    """
    判断单帧音频（16000Hz）是否包含人声。

    参数：
      frame     — 形状为 (N,) 的 float32 numpy 数组，已重采样至 16000Hz
      vad_model — Silero-VAD 模型实例

    返回：
      True = 有人声，False = 静音 / 噪音
    """
    # 归一化到 -1~1，Silero-VAD 要求
    max_val = np.abs(frame).max()
    if max_val > 0:
        frame = frame / max_val

    tensor     = torch.from_numpy(frame).float()
    confidence = vad_model(tensor, TARGET_SAMPLE_RATE).item()
    return confidence >= VAD_THRESHOLD


def record_once(vad_model) -> np.ndarray | None:
    """
    录制一段完整的语音片段（从检测到人声开始，到静音超时结束）。

    流程：
      1. 以 DEVICE_SAMPLE_RATE 持续读取麦克风帧
      2. 重采样至 16000Hz
      3. VAD 检测到人声 → 开始累积音频
      4. 静音超过 SILENCE_TIMEOUT 秒 → 停止，返回完整片段（16000Hz）

    返回：
      numpy float32 数组（16000Hz，可直接传入 Whisper），
      若用户按 Ctrl+C 则返回 None
    """
    print("[录音] 等待说话中... （按 Ctrl+C 退出）")

    audio_buffer   = []
    silence_frames = 0
    speech_started = False

    max_silence_frames = int(SILENCE_TIMEOUT / FRAME_DURATION)

    try:
        with sd.InputStream(
            samplerate=DEVICE_SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SIZE_DEVICE,
            device=DEVICE_INDEX,
        ) as stream:
            while True:
                frame, _ = stream.read(FRAME_SIZE_DEVICE)
                frame     = frame.flatten()
                frame_16k = _resample(frame)

                if is_speech(frame_16k, vad_model):
                    if not speech_started:
                        print("[录音] 检测到人声，开始录制...")
                        speech_started = True
                    audio_buffer.append(frame_16k)
                    silence_frames = 0
                else:
                    if speech_started:
                        audio_buffer.append(frame_16k)
                        silence_frames += 1
                        if silence_frames >= max_silence_frames:
                            print("[录音] 检测到停顿，识别中...\n")
                            break

    except KeyboardInterrupt:
        return None

    if not audio_buffer:
        return None

    return np.concatenate(audio_buffer, axis=0)