
FROM python:3.10-slim
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6  python-multipart && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn[standard] opencv-python-headless numpy onnxruntime
WORKDIR /app
ENV MODEL_PATH=/models/meter_type.onnx
EXPOSE 8601
COPY app.py .
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8601"]
