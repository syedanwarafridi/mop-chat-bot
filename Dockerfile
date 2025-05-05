FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
COPY . /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libgl1 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
        && \
    rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "generation:app", "--host", "0.0.0.0", "--port", "8080"]
