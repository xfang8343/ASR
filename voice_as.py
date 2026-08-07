#!/usr/bin/env python3
"""Push-to-talk bilingual voice assistant.

Run this file on the workstation::

    python voice_as.py server --host 0.0.0.0 --port 8000

Run it on the Ubuntu desktop::

    python voice_as.py client --server http://10.10.10.92:8000

Server environment variables (or matching command-line options):
``WHISPER_MODEL`` (large-v3), ``WHISPER_DEVICE`` (cuda),
``OLLAMA_URL`` (http://127.0.0.1:11434), ``OLLAMA_MODEL`` (qwen3:8b),
``PIPER_BIN`` (piper), ``PIPER_ZH_MODEL`` and ``PIPER_EN_MODEL``.

The program intentionally imports hardware/model libraries only in the mode
that needs them, so the client remains lightweight.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin


SYSTEM_PROMPT = (
    "You are a concise, friendly bilingual voice assistant. Respond in the "
    "same primary language as the user. If the user mixes Chinese and English, "
    "use both naturally when helpful. Keep replies conversational and suitable "
    "for speech. Do not use Markdown tables or long enumerations."
)


def chinese_ratio(text: str) -> float:
    """Return the fraction of CJK characters in non-whitespace text."""
    meaningful = [c for c in text if not c.isspace()]
    if not meaningful:
        return 0.0
    return sum("\u4e00" <= c <= "\u9fff" for c in meaningful) / len(meaningful)


def preferred_language(text: str, fallback: str = "en") -> str:
    return "zh" if chinese_ratio(text) > 0.2 else fallback


@dataclass(frozen=True)
class ServerConfig:
    temp_dir: Path
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    ollama_url: str
    ollama_model: str
    piper_bin: str
    piper_zh_model: str
    piper_en_model: str
    history_turns: int


def make_server_config(args: argparse.Namespace) -> ServerConfig:
    env = os.environ
    return ServerConfig(
        temp_dir=Path(args.temp_dir).expanduser(),
        whisper_model=args.whisper_model or env.get("WHISPER_MODEL", "large-v3"),
        whisper_device=args.whisper_device or env.get("WHISPER_DEVICE", "cuda"),
        whisper_compute_type=args.whisper_compute_type or env.get("WHISPER_COMPUTE_TYPE", "float16"),
        ollama_url=(args.ollama_url or env.get("OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/"),
        ollama_model=args.ollama_model or env.get("OLLAMA_MODEL", "qwen3:8b"),
        piper_bin=args.piper_bin or env.get("PIPER_BIN", "piper"),
        piper_zh_model=args.piper_zh_model or env.get("PIPER_ZH_MODEL", ""),
        piper_en_model=args.piper_en_model or env.get("PIPER_EN_MODEL", ""),
        history_turns=args.history_turns,
    )


def build_server_app(config: ServerConfig) -> Any:
    """Create FastAPI app; imports occur here so client mode needs no FastAPI."""
    try:
        import requests
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.responses import FileResponse
    except ImportError as exc:
        raise RuntimeError("Server dependencies missing. Install: fastapi uvicorn python-multipart requests faster-whisper") from exc

    config.temp_dir.mkdir(parents=True, exist_ok=True)
    input_dir, output_dir = config.temp_dir / "input", config.temp_dir / "output"
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    app = FastAPI(title="Voice Assistant API")
    histories: dict[str, list[dict[str, str]]] = {}
    history_lock = threading.Lock()
    model_lock = threading.Lock()
    whisper_model: Any = None

    def transcribe(audio_path: Path) -> tuple[str, str]:
        nonlocal whisper_model
        with model_lock:
            if whisper_model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError("faster-whisper is not installed") from exc
                whisper_model = WhisperModel(config.whisper_model, device=config.whisper_device,
                                             compute_type=config.whisper_compute_type)
        segments, info = whisper_model.transcribe(str(audio_path), beam_size=5, vad_filter=True)
        text = "".join(segment.text for segment in segments).strip()
        # VAD can reject quiet, clipped, or non-16-kHz microphone recordings.
        # Retry without VAD before reporting that no speech was found.
        if not text:
            segments, info = whisper_model.transcribe(str(audio_path), beam_size=5, vad_filter=False)
            text = "".join(segment.text for segment in segments).strip()
        return text, getattr(info, "language", "en") or "en"

    def ask_ollama(session_id: str, user_text: str) -> str:
        with history_lock:
            prior = histories.get(session_id, [])[-2 * config.history_turns:]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *prior,
                    {"role": "user", "content": user_text}]
        response = requests.post(config.ollama_url + "/api/chat", json={
            "model": config.ollama_model, "messages": messages, "stream": False,
            "options": {"temperature": 0.7},
        }, timeout=180)
        response.raise_for_status()
        answer = response.json().get("message", {}).get("content", "").strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response")
        with history_lock:
            histories.setdefault(session_id, []).extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ])
        return answer

    def synthesize(text: str, language: str, destination: Path) -> None:
        import subprocess
        model = config.piper_zh_model if language == "zh" else config.piper_en_model
        # During the Chinese-only setup, fall back to the available Chinese
        # voice even if Whisper labels a very short utterance as English.
        if not model and config.piper_zh_model:
            language = "zh"
            model = config.piper_zh_model
        if not model:
            raise RuntimeError("Piper model is not configured; set PIPER_ZH_MODEL/PIPER_EN_MODEL")
        try:
            completed = subprocess.run([config.piper_bin, "--model", model, "--output_file", str(destination)],
                                       input=text, text=True, capture_output=True, timeout=120, check=False)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Piper executable not found: {config.piper_bin}") from exc
        if completed.returncode != 0 or not destination.is_file():
            raise RuntimeError("Piper synthesis failed: " + completed.stderr.strip())

    def remove_expired_files(max_age_seconds: int = 3600) -> None:
        cutoff = time.time() - max_age_seconds
        for folder in (input_dir, output_dir):
            for item in folder.glob("*.wav"):
                try:
                    if item.stat().st_mtime < cutoff:
                        item.unlink()
                except OSError:
                    pass

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    async def chat(audio: UploadFile = File(...), session_id: str = Form("default")) -> dict[str, Any]:
        if not audio.filename or not audio.filename.lower().endswith(".wav"):
            raise HTTPException(400, "Please upload a WAV file in the 'audio' field")
        safe_session = re.sub(r"[^A-Za-z0-9_-]", "_", session_id)[:80] or "default"
        token = uuid.uuid4().hex
        input_path, output_path = input_dir / f"{token}.wav", output_dir / f"{token}.wav"
        started = time.perf_counter()
        try:
            input_path.write_bytes(await audio.read())
            asr_started = time.perf_counter()
            user_text, detected_language = transcribe(input_path)
            asr_time = time.perf_counter() - asr_started
            if not user_text:
                raise HTTPException(422, "No speech was recognized")
            llm_started = time.perf_counter()
            assistant_text = ask_ollama(safe_session, user_text)
            llm_time = time.perf_counter() - llm_started
            # Prefer the language detected from the user's speech.  An LLM may
            # append a short English phrase to a Chinese answer, which should
            # not make the server select an unconfigured English Piper voice.
            detected_language = (detected_language or "").lower()
            if detected_language.startswith("zh"):
                language = "zh"
            elif detected_language.startswith("en"):
                language = "en"
            else:
                language = preferred_language(assistant_text, detected_language)
            tts_started = time.perf_counter()
            synthesize(assistant_text, language, output_path)
            tts_time = time.perf_counter() - tts_started
            remove_expired_files()
            return {"user_text": user_text, "assistant_text": assistant_text, "language": language,
                    "audio_url": f"/audio/{token}.wav", "timing": {
                        "asr": round(asr_time, 3), "llm": round(llm_time, 3),
                        "tts": round(tts_time, 3), "total": round(time.perf_counter() - started, 3)}}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    # UploadFile is imported inside build_server_app so client mode stays lightweight.
    # Resolve the deferred annotation before FastAPI/Pydantic builds its request model.
    chat.__annotations__["audio"] = UploadFile
    chat.__annotations__["session_id"] = str
    app.post("/chat")(chat)

    @app.get("/audio/{audio_name}")
    def get_audio(audio_name: str) -> Any:
        if not re.fullmatch(r"[0-9a-f]{32}\.wav", audio_name):
            raise HTTPException(404, "Audio not found")
        path = output_dir / audio_name
        if not path.is_file():
            raise HTTPException(404, "Audio not found")
        return FileResponse(path, media_type="audio/wav", filename="assistant.wav")

    return app


class VoiceClient:
    def __init__(self, server: str, session_id: str, sample_rate: int, min_seconds: float,
                 max_seconds: float, output_device: int | str | None = None,
                 player: str = "sounddevice", alsa_device: str | None = None,
                 pw_target: str | None = None) -> None:
        self.server = server.rstrip("/")
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.min_seconds, self.max_seconds = min_seconds, max_seconds
        self.output_device = output_device
        self.player = player
        self.alsa_device = alsa_device
        self.pw_target = pw_target
        self.recording = False
        self.busy = False
        self.frames: list[Any] = []
        self.stream: Any = None
        self.lock = threading.Lock()

    def start_recording(self) -> None:
        import sounddevice as sd
        with self.lock:
            if self.busy or self.recording:
                return
            self.frames = []
            requested_rate = self.sample_rate
            try:
                stream = sd.InputStream(samplerate=requested_rate, channels=1, dtype="int16",
                                        callback=self._audio_callback)
            except sd.PortAudioError:
                # Some ALSA devices (including many laptop codecs) reject 16 kHz.
                # Record at the device's native rate; Whisper accepts and resamples WAV input.
                try:
                    device = sd.query_devices(kind="input")
                    fallback_rate = int(round(float(device["default_samplerate"])))
                    stream = sd.InputStream(samplerate=fallback_rate, channels=1, dtype="int16",
                                            callback=self._audio_callback)
                    self.sample_rate = fallback_rate
                    print(f"[提示] 麦克风不支持 {requested_rate} Hz，改用 {fallback_rate} Hz", flush=True)
                except Exception as exc:
                    print(f"[录音错误] 无法打开麦克风: {exc}", file=sys.stderr, flush=True)
                    return
            self.recording = True
            self.stream = stream
            self.stream.start()
        print("[录音中...]", flush=True)

    def _audio_callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            print(f"[音频警告] {status}", file=sys.stderr, flush=True)
        with self.lock:
            if self.recording:
                self.frames.append(indata.copy())

    def stop_recording(self) -> None:
        with self.lock:
            if not self.recording:
                return
            self.recording = False
            stream, self.stream = self.stream, None
            frames = self.frames
        if stream:
            stream.stop()
            stream.close()
        duration = sum(len(frame) for frame in frames) / self.sample_rate
        print(f"[录音结束] {duration:.1f} 秒", flush=True)
        if duration < self.min_seconds:
            print(f"[忽略] 录音少于 {self.min_seconds:.1f} 秒", flush=True)
            return
        if duration > self.max_seconds:
            print(f"[忽略] 录音超过 {self.max_seconds:.0f} 秒", flush=True)
            return
        self.busy = True
        threading.Thread(target=self._send_and_play, args=(frames,), daemon=True).start()

    def _send_and_play(self, frames: list[Any]) -> None:
        import numpy as np
        import requests
        import sounddevice as sd
        import soundfile as sf
        import tempfile
        try:
            audio = np.concatenate(frames, axis=0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                request_path = Path(handle.name)
            try:
                sf.write(request_path, audio, self.sample_rate, subtype="PCM_16")
                print("[上传并处理中...]", flush=True)
                with request_path.open("rb") as handle:
                    reply = requests.post(self.server + "/chat", files={"audio": ("audio.wav", handle, "audio/wav")},
                                          data={"session_id": self.session_id}, timeout=300)
                if not reply.ok:
                    try:
                        detail = reply.json().get("detail", reply.text)
                    except ValueError:
                        detail = reply.text
                    raise RuntimeError(f"服务器 HTTP {reply.status_code}: {detail}")
                result = reply.json()
                print(f"[识别语言] {result.get('language', 'unknown')}")
                print(f"[你] {result.get('user_text', '')}")
                print(f"[助手] {result.get('assistant_text', '')}")
                timing = result.get("timing", {})
                print("[耗时] " + ", ".join(f"{key.upper()}: {value} s" for key, value in timing.items()))
                audio_url = result.get("audio_url")
                if not audio_url:
                    raise RuntimeError("Server response did not include audio_url")
                sound_reply = requests.get(urljoin(self.server + "/", audio_url), timeout=120)
                sound_reply.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
                    output.write(sound_reply.content)
                    response_path = Path(output.name)
                try:
                    data, rate = sf.read(response_path, dtype="float32")
                    # Piper voices commonly output 22.05 kHz, while ALSA/HDMI
                    # playback devices often accept only 44.1/48 kHz.
                    output_device = (sd.query_devices(self.output_device)
                                     if self.output_device is not None
                                     else sd.query_devices(kind="output"))
                    output_rate = int(round(float(output_device["default_samplerate"])))
                    if output_rate != rate:
                        source_length = len(data)
                        target_length = max(1, round(source_length * output_rate / rate))
                        source_x = np.linspace(0.0, 1.0, source_length, endpoint=False)
                        target_x = np.linspace(0.0, 1.0, target_length, endpoint=False)
                        if getattr(data, "ndim", 1) == 1:
                            data = np.interp(target_x, source_x, data).astype(np.float32)
                        else:
                            data = np.column_stack([
                                np.interp(target_x, source_x, data[:, channel])
                                for channel in range(data.shape[1])
                            ]).astype(np.float32)
                        print(f"[提示] 播放设备不支持 {rate} Hz，重采样到 {output_rate} Hz", flush=True)
                        rate = output_rate
                    print("[正在播放...]", flush=True)
                    if self.player in ("aplay", "pw-play"):
                        if self.player == "pw-play":
                            command = ["pw-play"]
                            if self.pw_target:
                                command.extend(["--target", self.pw_target])
                        else:
                            command = ["aplay", "-q"]
                            if self.alsa_device:
                                command.extend(["-D", self.alsa_device])
                        command.append(str(response_path))
                        subprocess.run(command, check=True, timeout=120)
                    elif self.output_device is None:
                        sd.play(data, rate)
                    else:
                        sd.play(data, rate, device=self.output_device)
                    if self.player not in ("aplay", "pw-play"):
                        sd.wait()
                    print("[播放完成]", flush=True)
                finally:
                    response_path.unlink(missing_ok=True)
            finally:
                request_path.unlink(missing_ok=True)
        except Exception as exc:
            print(f"[错误] {exc}", file=sys.stderr, flush=True)
        finally:
            self.busy = False


def run_client(args: argparse.Namespace) -> None:
    try:
        from pynput import keyboard
        import numpy  # noqa: F401
        import requests  # noqa: F401
        import sounddevice  # noqa: F401
        import soundfile  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Client dependencies missing. Install: pynput sounddevice soundfile numpy requests") from exc
    output_device: int | str | None = args.output_device
    if isinstance(output_device, str) and output_device.isdigit():
        output_device = int(output_device)
    client = VoiceClient(args.server, args.session_id or uuid.uuid4().hex, args.sample_rate,
                         args.min_seconds, args.max_seconds, output_device,
                         args.player, args.alsa_device, args.pw_target)
    print("Voice Assistant Client")
    print(f"Server: {args.server}")
    print("按住 V 开始说话，松开 V 发送；按 ESC 退出")

    def on_press(key: Any) -> bool | None:
        if key == keyboard.Key.esc:
            return False
        if (getattr(key, "char", "") or "").lower() == "v":
            client.start_recording()
        return None

    def on_release(key: Any) -> None:
        if (getattr(key, "char", "") or "").lower() == "v":
            client.stop_recording()

    # Prevent the push-to-talk key from being echoed into the terminal or
    # forwarded to the active application while the client is running.
    with keyboard.Listener(on_press=on_press, on_release=on_release, suppress=True) as listener:
        listener.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push-to-talk client and voice-assistant server")
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("server", help="run FastAPI on the workstation")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=8000)
    server.add_argument("--temp-dir", default="./voice_assistant_temp")
    server.add_argument("--whisper-model")
    server.add_argument("--whisper-device")
    server.add_argument("--whisper-compute-type")
    server.add_argument("--ollama-url")
    server.add_argument("--ollama-model")
    server.add_argument("--piper-bin")
    server.add_argument("--piper-zh-model")
    server.add_argument("--piper-en-model")
    server.add_argument("--history-turns", type=int, default=8)
    client = commands.add_parser("client", help="run the lightweight desktop client")
    client.add_argument("--server", default="http://10.10.10.92:8000")
    client.add_argument("--session-id", default="")
    client.add_argument("--sample-rate", type=int, default=16000)
    client.add_argument("--min-seconds", type=float, default=0.3)
    client.add_argument("--max-seconds", type=float, default=30.0)
    client.add_argument("--output-device", default=None,
                        help="播放设备编号或名称；不指定则使用系统默认输出设备")
    client.add_argument("--player", choices=("sounddevice", "aplay", "pw-play"), default="sounddevice",
                        help="播放后端；PipeWire 系统建议使用 pw-play")
    client.add_argument("--alsa-device", default=None,
                        help="aplay 设备，例如 hw:1,0；仅 --player aplay 生效")
    client.add_argument("--pw-target", default=None,
                        help="PipeWire 输出节点 ID，例如 39；仅 --player pw-play 生效")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "server":
            try:
                import uvicorn
            except ImportError as exc:
                raise RuntimeError("Server dependency missing. Install: uvicorn") from exc
            uvicorn.run(build_server_app(make_server_config(args)), host=args.host, port=args.port)
        else:
            run_client(args)
    except RuntimeError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    except KeyboardInterrupt:
        print("\n已退出")


if __name__ == "__main__":
    main()
