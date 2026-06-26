FROM python:3.12-slim
WORKDIR /app

# Cài đặt các công cụ build C++ để compile thư viện như bitsandbytes / unsloth (nếu cần)
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config/ ./config/
COPY src/ ./src/

# Nhận Token từ tham số build (nếu model bị khóa/gated)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Tải trước model weights vào Docker Image để chạy Offline
RUN python -c "from huggingface_hub import snapshot_download; import yaml, os; \
    cfg = yaml.safe_load(open('config/settings.yaml')); \
    t = os.environ.get('HF_TOKEN', '').strip(); \
    snapshot_download(repo_id=cfg['llm']['model_name'], token=t if t else None)"

CMD ["python", "src/main.py"]