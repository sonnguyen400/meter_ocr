
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn, numpy as np, cv2, os
try:
    import onnxruntime as ort
except Exception:
    ort = None

app = FastAPI(title="Meter Type Classifier")
session = None
labels = ["lcd_digital", "flip_mechanical"]

@app.on_event("startup")
def load_model():
    global session
    model_path = os.environ.get("MODEL_PATH", "/models/meter_type.onnx")
    if ort and os.path.exists(model_path):
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    else:
        session = None

class Resp(BaseModel):
    type: str
    probs: dict
    used: str

@app.post("/classify", response_model=Resp)
async def classify(file: UploadFile = File(...)):
    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return Resp(type="lcd_digital", probs={"lcd_digital":0.5,"flip_mechanical":0.5}, used="fallback:decode_failed")

    if session is not None:
        inp = cv2.resize(img, (224,224))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        x = np.expand_dims(np.transpose(inp, (2,0,1)), 0)
        logits = session.run(None, {session.get_inputs()[0].name: x})[0].squeeze()
        ex = np.exp(logits - np.max(logits)); probs = ex / np.sum(ex)
        d = {labels[i]: float(probs[i]) for i in range(len(labels))}
        return Resp(type=labels[int(np.argmax(probs))], probs=d, used="onnx")
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edge_density = float(edges.mean())/255.0
        scharrx = cv2.Scharr(blur, cv2.CV_32F, 1, 0)
        scharry = cv2.Scharr(blur, cv2.CV_32F, 0, 1)
        sx = float(np.mean(np.abs(scharrx))); sy = float(np.mean(np.abs(scharry)))
        hv_ratio = sx/(sy+1e-6)
        import numpy as np
        p_lcd = np.clip(0.55 + 0.25*max(0.0, hv_ratio-1.0) + 0.2*max(0.0, 0.25-edge_density), 0.05, 0.95)
        d = {"lcd_digital": float(p_lcd), "flip_mechanical": float(1.0-p_lcd)}
        return Resp(type="lcd_digital" if p_lcd>=0.5 else "flip_mechanical", probs=d, used="heuristic")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8601)
