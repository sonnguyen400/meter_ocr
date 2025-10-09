
# Meter OCR Service (Spring Boot + OpenCV + Tesseract + PaddleOCR + Classifier)

This repository provides a production-ready skeleton to build an OCR service to extract **meter reading** and **serial number** from electricity meter photos using:

- **Spring Boot** (REST API)
- **OpenCV (JavaCPP)** for preprocessing
- **Tesseract (tess4j)** for digit-focused OCR
- **PaddleOCR** (FastAPI microservice) for general text detection & recognition
- **Classifier service** (FastAPI + ONNX Runtime) for **auto-detecting meter type**
- **Docker Compose** to run everything together

## Quick Start

### 1) Requirements
- Docker & Docker Compose
- Ports available: `8080` (API), `8501` (Paddle service), `8601` (Classifier)

### 2) Run
```bash
docker compose up --build
```

### 3) Test OCR
```bash
curl -F "file=@/path/to/meter.jpg" "http://localhost:8080/ocr/meter?type=auto&debug=true" | jq
```

### 4) Save Debug Overlay
```bash
curl -F "file=@/path/to/meter.jpg" "http://localhost:8080/ocr/meter?type=auto&debug=true"  | jq -r .debug_overlay_b64 | base64 --decode > overlay.png
```

### 5) Profiles (multi-type tuning)
Use `type` query param to select profile:
```bash
curl -F "file=@meter.jpg" "http://localhost:8080/ocr/meter?type=lcd_digital"
```
Available default profiles: `lcd_digital`, `flip_mechanical`. Configure at:
`spring-boot-api/src/main/resources/meter-profiles.yml`.

### 6) Provide ONNX classifier model (optional)
Mount your model at `/models/meter_type.onnx` for the classifier-service (shape `(1,3,224,224)`) by adding a volume to `classifier` in `docker-compose.yml`:
```yaml
classifier:
  build: ./classifier-service
  ports: ["8601:8601"]
  volumes:
    - ./models:/models
```

## Endpoints
- `POST /ocr/meter?type=auto|lcd_digital|flip_mechanical&debug=true|false`
- `POST /ocr/meter/url?type=...&debug=...` (body `{ "url": "..." }`)
- `GET /health`
