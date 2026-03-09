import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel
import torchvision.transforms as T
from PIL import Image
import json

# Константы для нормализации изображений
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

# ========== МАППИНГ КЛАССОВ ==========
ID_TO_LABEL = {
    0: "мода",
    1: "технологии", 
    2: "финансы",
    3: "крипта",
    4: "спорт"
}

def get_class_name(class_id: int) -> str:
    """Возвращает название класса по ID"""
    return ID_TO_LABEL.get(class_id, f"unknown_{class_id}")

# ========== КЛАСС МОДЕЛИ (как при обучении) ==========

class MyBERTUnFreeze(nn.Module):
    def __init__(self, num_labels=5):
        super().__init__()
        self.bert = BertModel.from_pretrained("cointegrated/rubert-tiny2")
        hidden_size = self.bert.config.hidden_size
        
        # Классификатор (как при обучении)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        device = next(self.parameters()).device
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_out.pooler_output
        logits = self.linear(pooled_output)
        return logits

# ========== ФУНКЦИИ ЗАГРУЗКИ ==========

def load_rubert_tokenizer():
    """Загружает токенизатор"""
    tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2')
    return tokenizer

def load_rubert_model():
    """Загружает вашу кастомную модель с весами"""
    # 1. Создаём архитектуру
    model = MyBERTUnFreeze(num_labels=5)
    
    # 2. Загружаем веса
    state_dict = torch.load('api/utils/pytorch_model.bin', map_location='cpu')
    
    # 3. Если веса сохранены с ключом 'model_state_dict' или 'state_dict'
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    # 4. Загружаем веса (strict=False игнорирует несовпадения)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model

def load_yolo_model():
    """Загружает YOLO модель"""
    from ultralytics import YOLO
    model = YOLO('api/weights/yolo26m.pt')
    return model

# ========== ПРЕДОБРАБОТКА ==========

def preprocess_text(text, tokenizer, max_length=128):
    """Токенизация текста для BERT"""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask']
    }

def transform_image(img):
    """Предобработка изображения"""
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN, std=IMG_STD)
    ])
    return transforms(img).unsqueeze(0)

def get_text_embeddings(text, model, tokenizer, max_length=128):
    """Получение эмбеддингов текста через RuBERT"""
    inputs = preprocess_text(text, tokenizer, max_length)
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
    
    # Для кастомной модели возвращаем logits (классификация)
    return outputs

def detect_objects(model, img_path, conf_threshold=0.5):
    """Детекция объектов через YOLO"""
    results = model.predict(img_path, conf=conf_threshold, verbose=False)
    return results