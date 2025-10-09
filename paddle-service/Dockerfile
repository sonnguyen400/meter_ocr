
FROM python:3.10-slim
RUN apt-get update && apt-get install -y   libgl1  libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir  --default-timeout=1000 numpy==1.26.4 fastapi uvicorn[standard] opencv-python-headless paddlepaddle==2.6.0 paddleocr==2.7.0.3 requests python-multipart
WORKDIR /app
COPY app.py .
EXPOSE 8501
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8501"]
