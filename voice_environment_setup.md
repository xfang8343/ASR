# 语音助手 Conda 环境

本项目把两个运行环境分开：客户端不需要模型或 GPU，服务器才运行
faster-whisper、Ollama 和 Piper。

## 客户端（本地 Ubuntu 电脑）

在包含 `voice_client` 目录的位置运行：

```bash
conda env create -f voice_client/environment.yml
conda activate voice_client
python voice_as.py client --server http://10.10.10.92:8000
```

若系统未识别麦克风，先在系统层安装或启用 PulseAudio/PipeWire 的输入设备；
`portaudio` 和 `libsndfile` 已由 Conda 环境提供。

## 服务端（10.10.10.92 工作站）

先确保 Ollama 已启动且模型已拉取：

```bash
ollama pull qwen3:8b
CUDA_VISIBLE_DEVICES=1 ollama serve
```

再创建并启动 Python 服务：

```bash
conda env create -f voice_server/environment.yml
conda activate voice_server
export PIPER_ZH_MODEL=/absolute/path/zh_CN-huayan-medium.onnx
export PIPER_EN_MODEL=/absolute/path/en_US-lessac-medium.onnx
python voice_as.py server --host 0.0.0.0 --port 8000
```

首次语音请求会下载并加载 `large-v3` Whisper 模型。若需要将 Whisper 固定在
GPU 1，可在启动服务前设置：

```bash
CUDA_VISIBLE_DEVICES=1 python voice_as.py server --host 0.0.0.0 --port 8000
```

检查服务是否可访问：

```bash
curl http://10.10.10.92:8000/health
```

应返回 `{"status":"ok"}`。

工作站服务器2个终端分别运行fastapi和ollama
fangxing@buendia-Z890-EAGLE-WIFI7:~/software$ mkdir -p "$HOME/software/ollama-models"

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

本地客户端 → SSH 隧道 → 服务器 FastAPI

本地2个终端分别运行SSH 隧道和脚本
ssh -o ExitOnForwardFailure=yes   -N -L 8001:127.0.0.1:8000   fangxing@10.10.10.92

python voice_as.py client   --server http://127.0.0.1:8001