# ASR & Remote Voice Assistant

一个面向中英文场景的语音项目，包含两部分：

1. 原有的音频文件/本地麦克风 ASR、翻译功能；
2. 类似 GPT Voice 的远程语音助手：本地电脑按住 `V` 录音，松开后上传到远程 GPU 工作站，完成语音识别、LLM 对话和语音合成，再把文字与音频返回本地播放。

## 语音助手效果

```text
本地 Ubuntu 客户端
    └─ 按住 V 录音 / 松开 V 上传 / 播放返回音频
            │
            └─ SSH 隧道（本地 8001 → 服务器 8000）
                    │
                    ▼
远程 GPU 工作站 10.10.10.92
    ├─ FastAPI + faster-whisper：语音识别
    ├─ Ollama + Qwen3：对话生成
    └─ Piper：中英文语音合成
```

本地客户端不运行 Whisper、LLM 或 TTS 模型；服务器负责计算，本地只负责录音、网络传输、显示和播放。

## 项目结构

```text
ASR/
├── voice_as.py                    # 远程语音助手：server/client 双模式
├── voice_client/
│   ├── environment.yml            # 本地客户端 Conda 环境
│   └── requirements.txt
├── voice_server/
│   ├── environment.yml            # GPU 工作站 Conda 环境
│   └── requirements.txt
├── voice_environment_setup.md     # 环境配置简版说明
├── asr_core.py                    # 原有 ASR 统一入口
├── asr_mp3.py                     # 音频文件识别/翻译
├── asr_micro.py                   # 原有实时麦克风识别
├── mp3/                           # 原有音频文件处理模块
├── mic/                           # 原有麦克风处理模块
├── test_audio/                    # 测试音频
└── result/                        # 原有识别结果
```

## 远程语音助手快速开始

以下流程使用无 sudo 的用户级 Ollama 安装，以及 SSH 隧道连接。假设：

- 工作站用户名：`fangxing`
- 工作站地址：`10.10.10.92`
- 工作站 GPU 1 用于 Ollama 和 Whisper
- 项目目录：`~/projects/ASR`
- 本地项目目录：`~/ASR`

### 1. 将项目传到工作站

在本地客户端执行：

```bash
cd ~/ASR
scp voice_as.py fangxing@10.10.10.92:/home/fangxing/projects/ASR/
scp -r voice_server fangxing@10.10.10.92:/home/fangxing/projects/ASR/
```

### 2. 创建客户端环境

在本地客户端执行：

```bash
cd ~/ASR
conda env create -f voice_client/environment.yml
conda activate voice_client
python -m pip check
```

客户端依赖包括 `pynput`、`sounddevice`、`soundfile`、`numpy` 和 `requests`，不需要 GPU、Whisper、FastAPI 或 Ollama。

检查本地录音设备：

```bash
python - <<'PY'
import sounddevice as sd
print(sd.query_devices())
print("Default input device:", sd.default.device[0])
PY
```

### 3. 安装并启动 Ollama（工作站）

如果有 sudo 权限，可以使用官方安装脚本：

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

如果没有 sudo 权限，使用用户级安装：

```bash
mkdir -p "$HOME/software/ollama"

curl -L https://ollama.com/download/ollama-linux-amd64.tar.zst \
  | tar --zstd -x -C "$HOME/software/ollama"

echo 'export PATH="$HOME/software/ollama/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
ollama -v
```

创建模型目录并启动 Ollama。工作站需要单独保持一个终端运行：

```bash
mkdir -p "$HOME/software/ollama-models"

CUDA_VISIBLE_DEVICES=1 \
OLLAMA_VULKAN=false \
OLLAMA_MODELS="$HOME/software/ollama-models" \
ollama serve
```

另开工作站终端下载并测试 Qwen：

```bash
ollama pull qwen3:8b
ollama list
ollama run qwen3:8b "请用一句中文介绍你自己"
curl http://127.0.0.1:11434/api/tags
```

`CUDA_VISIBLE_DEVICES=1` 会让 Ollama 使用物理 GPU 1；进程内部通常会把它重新编号为 `CUDA0`，这是正常现象。`OLLAMA_VULKAN=false` 用于避免桌面 GPU 0 被 Vulkan 后端占用。

### 4. 创建服务端 Conda 环境

在工作站执行：

```bash
cd ~/projects/ASR
conda env create -f voice_server/environment.yml
conda activate voice_server
python -m pip check
```

如果 PyPI 下载较慢，可以保留已创建的 Conda 环境，只用镜像补装 pip 依赖：

```bash
conda activate voice_server
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
python -m pip install -r voice_server/requirements.txt
```

服务端依赖包括 FastAPI、Uvicorn、faster-whisper、Piper 和 `python-multipart`；CUDA 驱动由系统提供，Conda 环境包含 CUDA 运行库。

### 5. 下载 Piper 语音模型

在工作站执行：

```bash
mkdir -p "$HOME/models/piper"
cd "$HOME/models/piper"
```

中文模型：

```bash
wget -c --timeout=30 --tries=3 \
  https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx \
  -O zh_CN-huayan-medium.onnx

wget -c --timeout=30 --tries=3 \
  https://hf-mirror.com/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json \
  -O zh_CN-huayan-medium.onnx.json
```

英文模型：

```bash
wget -c --timeout=30 --tries=3 \
  https://hf-mirror.com/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx \
  -O en_US-lessac-medium.onnx

wget -c --timeout=30 --tries=3 \
  https://hf-mirror.com/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \
  -O en_US-lessac-medium.onnx.json
```

检查并测试：

```bash
ls -lh "$HOME/models/piper"

printf '你好，这是中文语音合成测试。' \
  | piper --model "$HOME/models/piper/zh_CN-huayan-medium.onnx" \
          --output_file /tmp/test_zh.wav

file /tmp/test_zh.wav
```

必须同时存在每个模型的 `.onnx` 与 `.onnx.json` 文件。

### 6. 启动服务器 FastAPI

在工作站另一个终端执行（保持 Ollama 终端也在运行）：

```bash
cd ~/projects/ASR
conda activate voice_server

export PIPER_ZH_MODEL="$HOME/models/piper/zh_CN-huayan-medium.onnx"
export PIPER_EN_MODEL="$HOME/models/piper/en_US-lessac-medium.onnx"

CUDA_VISIBLE_DEVICES=1 \
OLLAMA_VULKAN=false \
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
python voice_as.py server \
  --host 0.0.0.0 \
  --port 8000 \
  --whisper-model large-v3 \
  --piper-zh-model "$PIPER_ZH_MODEL" \
  --piper-en-model "$PIPER_EN_MODEL"
```

看到以下输出表示 FastAPI 已启动：

```text
Uvicorn running on http://0.0.0.0:8000
```

检查服务器本机 API：

```bash
curl http://127.0.0.1:8000/health
```

应返回：

```json
{"status":"ok"}
```

首次语音请求会下载并加载 `large-v3`。模型缓存通常位于：

```text
~/.cache/huggingface/hub
```

可用以下命令观察下载进度：

```bash
watch -n 5 'du -sh ~/.cache/huggingface/hub'
```

### 7. 建立 SSH 隧道

由于服务器的 8000 端口可能被防火墙阻止，而 SSH 已经可用，推荐使用本地端口转发。在本地客户端另开一个终端，保持该命令运行：

```bash
ssh -o ExitOnForwardFailure=yes \
  -N -L 8001:127.0.0.1:8000 \
  fangxing@10.10.10.92
```

隧道含义：

```text
本地 127.0.0.1:8001 → SSH → 工作站 127.0.0.1:8000
```

本地测试：

```bash
curl http://127.0.0.1:8001/health
```

### 8. 启动本地语音客户端

在本地客户端另开一个终端：

```bash
cd ~/ASR
conda activate voice_client

python voice_as.py client \
  --server http://127.0.0.1:8001
```

交互方式：

1. 按住 `V` 开始录音；
2. 持续说话；
3. 松开 `V` 上传；
4. 等待服务器识别、回答和合成；
5. 本地显示文字并播放语音；
6. 按 `ESC` 退出。

## `voice_as.py` 设计说明

### Server 模式

```bash
python voice_as.py server --host 0.0.0.0 --port 8000
```

提供：

- `GET /health`：健康检查；
- `POST /chat`：接收 WAV 与 `session_id`，返回用户文本、助手文本、语言、音频 URL 和耗时；
- `GET /audio/{id}.wav`：下载合成音频。

服务器在进程内保存本次运行的会话历史，服务重启后历史清空。默认最多保留最近若干轮上下文。

### Client 模式

```bash
python voice_as.py client --server http://127.0.0.1:8001
```

客户端具有以下保护逻辑：

- 最短录音 0.3 秒，最长录音 30 秒；
- 默认尝试 16 kHz；若声卡不支持，则自动回退到设备默认采样率；
- 播放前将 Piper 的 22050 Hz 音频重采样到本地输出设备支持的采样率；
- 按键事件被抑制，避免 `V` 在终端中回显；
- HTTP 错误会显示服务器返回的具体 `detail`。

## 常见问题与踩坑记录

### 1. `sudo` 安装 Ollama 失败

错误：

```text
fangxing 未出现在 sudoers 文件中
```

原因是当前用户没有 sudo 权限。解决方法是使用用户级安装，把 Ollama 放到：

```text
~/software/ollama
```

模型放到：

```text
~/software/ollama-models
```

### 2. `ollama -v` 能运行，但无法连接

`ollama -v` 只表示客户端已安装，不表示服务端进程正在运行。必须另开终端执行：

```bash
CUDA_VISIBLE_DEVICES=1 \
OLLAMA_VULKAN=false \
OLLAMA_MODELS="$HOME/software/ollama-models" \
ollama serve
```

用下面的 API 检查服务：

```bash
curl http://127.0.0.1:11434/api/tags
```

### 3. 客户端访问 `10.10.10.92:8000` 超时

即使 `ping` 成功，TCP 8000 端口也可能被防火墙拦截。SSH 隧道不需要开放 8000 端口：

```bash
ssh -N -L 8001:127.0.0.1:8000 fangxing@10.10.10.92
```

客户端应访问 `http://127.0.0.1:8001`，而不是直接访问服务器的 8000 端口。

### 4. `Invalid sample rate`

不同 ALSA 设备支持的采样率不同。本机麦克风不支持 16000 Hz，而支持 44100 Hz。客户端现在会自动检测并回退，不再要求手动修改声卡配置。

播放设备也可能不支持 Piper 的 22050 Hz，客户端会在播放前自动重采样。

### 5. 录音时终端出现大量 `vvvv`

这是键盘监听事件被终端回显造成的，不是录音内容。客户端现在使用 `pynput` 的按键抑制功能，避免 `V` 被发送给终端或其他窗口。

### 6. `No speech was recognized`

Whisper 的 VAD 可能误判短句、噪声较大或采样率非 16 kHz 的音频。服务器现在会先使用 VAD，若没有文字则自动关闭 VAD 重试。

若仍无法识别，检查：

```bash
python -c "import sounddevice as sd; print(sd.query_devices())"
```

并将麦克风输入增益降低到约 50%～70%，避免峰值长期达到 1.0 导致削波。

### 7. Hugging Face 出现 `401 Unauthorized` 或 Xet 错误

服务器启动时使用：

```bash
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_XET=1
```

Whisper 模型下载缓存位于 `~/.cache/huggingface/hub`。首次 `large-v3` 下载可能耗时较长，不要重复启动多个服务或删除缓存。

### 8. FastAPI/Pydantic 报 `UploadFile` 未定义

这是 Python 延迟注解与函数内部导入 `UploadFile` 的兼容问题。`voice_as.py` 已在注册路由前显式解析注解；如果服务器仍出现该错误，请确认已把最新的 `voice_as.py` 重新传到服务器。

### 9. `Piper model is not configured`

启动 FastAPI 时必须传入模型路径：

```bash
--piper-zh-model "$PIPER_ZH_MODEL" \
--piper-en-model "$PIPER_EN_MODEL"
```

并确认文件存在：

```bash
test -f "$PIPER_ZH_MODEL" && echo "Chinese Piper model exists"
test -f "$PIPER_EN_MODEL" && echo "English Piper model exists"
```

### 10. 模型文件下载很慢

优先使用 `wget -c` 断点续传、Hugging Face 镜像，或在网络较快的电脑下载后通过 `scp` 传到工作站。不要删除已有缓存；中断下载后可以继续执行同一个命令。

## 原有 ASR 功能

远程语音助手之外，仓库仍保留原有的 Whisper ASR 和翻译功能。

### 统一入口

```bash
python asr_core.py
```

### 音频文件识别/翻译

```bash
python asr_mp3.py ./test_audio/ZH.mp3
```

支持自动检测语言、中文简体转换以及中英翻译，结果写入 `result/`。

### 原有本地麦克风识别

```bash
python asr_micro.py
```

该模式使用原有的 Silero-VAD 流程，与 `voice_as.py client` 的按键录音模式相互独立。

## 开发与验证记录

本项目从本地 ASR 扩展为远程语音对话系统，主要完成了：

- 设计并实现 `server/client` 双模式单文件程序；
- 配置客户端与服务端独立 Conda 环境；
- 在双 RTX 5090 工作站上验证 Ollama + Qwen3 GPU 推理；
- 部署 faster-whisper `large-v3` 和 Piper 中英文模型；
- 实现会话历史、语言识别、TTS 语言选择、耗时统计和 WAV 返回；
- 通过 SSH 隧道解决服务器端口无法直接访问的问题；
- 适配 ALSA 录音设备不支持 16 kHz、播放设备不支持 22.05 kHz 的情况；
- 处理 Hugging Face 镜像、Xet 401、模型缓存和断点续传问题；
- 修复 FastAPI/Pydantic `UploadFile` 延迟注解错误；
- 修复中文回答夹带英文导致错误选择英文 Piper 的问题；
- 增强 HTTP 错误显示、VAD 失败重试和终端按键回显处理。

## 性能参考

一次成功请求的终端输出示例：

```text
ASR: 0.199 s
LLM: 4.21 s
TTS: 0.837 s
Total: 5.247 s
```

实际耗时取决于 Whisper 模型是否已经加载、Qwen 上下文长度、网络隧道延迟、语音长度和磁盘缓存状态。

## 安全注意事项

- Ollama 默认只监听服务器本机 `127.0.0.1:11434`，不直接暴露到网络；
- FastAPI 使用 SSH 隧道时无需开放 8000 端口；
- 不建议将 FastAPI 或 Ollama 直接暴露到公网；
- SSH 密码、私钥、模型缓存路径不要提交到 Git；
- 生产部署应使用 systemd、专用用户、日志轮转和反向代理。

## 许可证与模型说明

代码依赖的 Whisper、faster-whisper、Ollama、Piper、Qwen3 及 Piper voice models 各自遵循其上游项目和模型许可证。发布仓库或进行商业使用前，请分别确认代码依赖和语音模型的许可条款。
