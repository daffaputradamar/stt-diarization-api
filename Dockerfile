FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt update && apt install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

COPY pyproject.toml .python-version ./
RUN uv sync --frozen

COPY . .

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
