import sys
import os

# Добавляем корень проекта в путь поиска модулей
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import io
import torch
import uvicorn
import PIL.Image
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
  # 🔥 Обязательно!

from utils.model_func import (
    load_rubert_tokenizer,
    load_rubert_model,
    load_yolo_model,
    preprocess_text,
    transform_image,
    get_text_embeddings,
    detect_objects,
    get_class_name
)

logger = logging.getLogger('uvicorn.info')

class ImageResponse(BaseModel):
    detections: list[dict]
    message: str

class TextInput(BaseModel):
    text: str

class TextResponse(BaseModel):
    text: str
    predicted_class: int          # Номер класса (0-4)
    class_name: str               # 🔥 Название класса ("мода", "спорт" и т.д.)
    confidence: float             # Уверенность модели
    status: str

class TableInput(BaseModel):
    feature1: float
    feature2: float

class TableOutput(BaseModel):
    prediction: float

yolo_model = None
rubert_model = None
rubert_tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global yolo_model, rubert_model, rubert_tokenizer
    
    logger.info("Загрузка моделей...")
    
    # Загрузка RuBERT (токенизатор + модель)
    rubert_tokenizer = load_rubert_tokenizer()
    rubert_model = load_rubert_model()
    logger.info("RuBERT loaded")
    
    # Загрузка YOLO
    yolo_model = load_yolo_model()
    logger.info("YOLO loaded")
    
    yield
    
    # Очистка памяти при выключении
    del yolo_model, rubert_model, rubert_tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    """Healthcheck endpoint"""
    return {'status': 'API is running', 'models': ['yolo', 'rubert-tiny2']}

@app.post('/clf_image')
async def classify_image(file: UploadFile):
    """
    Эндпоинт для детекции объектов (YOLO).
    Принимает изображение, возвращает найденные объекты.
    """
    try:
        # Читаем файл в память
        contents = await file.read()
        image = PIL.Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Используем импортированную функцию detect_objects
        results = detect_objects(yolo_model, image, conf_threshold=0.5)
        
        # Формируем ответ
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": results[0].names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })
        
        return ImageResponse(detections=detections, message="OK")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/clf_text')
def clf_text(data: TextInput):
    """
    Эндпоинт для классификации текста (RuBERT).
    Возвращает предсказанный класс и вероятности.
    """
    try:
        inputs = preprocess_text(data.text, rubert_tokenizer)
        
        with torch.no_grad():
            logits = rubert_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
        
        # Получаем вероятности через softmax
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_class].item()
        
        return TextResponse(
            text=data.text,
            predicted_class=pred_class,
            class_name=get_class_name(pred_class),
            confidence=confidence,
            status="processed"
        )
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post('/clf_table')
def predict(x: TableInput):
    """
    Эндпоинт для табличных данных (заглушка).
    """
    # Здесь можно добавить sklearn модель через model_func если нужно
    return TableOutput(prediction=0.0)

if __name__ == "__main__":
    # Запуск: python api/main.py
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)
