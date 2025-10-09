
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from paddleocr import PaddleOCR
import uvicorn, numpy as np, cv2, requests

app = FastAPI(title="PaddleOCR Service")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

class UrlReq(BaseModel):
    url: str

@app.on_event("startup")
async def warmup():
    img = np.zeros((16,16,3), dtype=np.uint8)
    _ = ocr.ocr(img, cls=True)

@app.post("/paddle/ocr")
async def ocr_image(file: UploadFile = File(...)):
    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    result = ocr.ocr(img, cls=True)
    boxes = []
    for line in result:
        for box, (text, conf) in line:
            boxes.append({"polygon": box, "text": text, "conf": float(conf)})
    return {"boxes": boxes}

@app.post("/paddle/ocr/url")
async def ocr_image_url(req: UrlReq):
    r = requests.get(req.url, timeout=10)
    r.raise_for_status()
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    result = ocr.ocr(img, cls=True)
    boxes = []
    for line in result:
        for box, (text, conf) in line:
            boxes.append({"polygon": box, "text": text, "conf": float(conf)})
    return {"boxes": boxes}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8501)
