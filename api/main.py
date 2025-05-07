from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import uuid
import os

from fastapi.responses import StreamingResponse
from io import BytesIO
import cv2

import private_path

app = FastAPI()
model = YOLO(str(private_path.model_path))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_id = str(uuid.uuid4())
    input_path = f"temp_{image_id}.jpg"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

        # Запуск предсказания
        results = model.predict(source=input_path, save=False)

        # Извлечение результатов
        boxes = results[0].boxes
        output = []
        for box in boxes:
            coords = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            output.append({
                "class": model.names[cls],
                "confidence": round(conf, 3),
                "bbox": [round(x, 2) for x in coords]
            })

    # Очистка
    os.remove(input_path)

    return JSONResponse(content={"detections": output})

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    # Сохраним изображение
    image_id = str(uuid.uuid4())
    input_path = f"temp_{image_id}.jpg"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Предсказание
    results = model.predict(source=input_path, save=False)
    img = results[0].plot() # рисуем боксы на изображении

    # Преобразуем изображение в поток
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = BytesIO(img_encoded.tobytes())

    os.remove(input_path)
    return StreamingResponse(img_bytes, media_type="image/jpeg")