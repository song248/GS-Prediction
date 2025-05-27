import os
import cv2
import numpy as np
import torch
import pickle
import time
import pandas as pd
from PyQt5.QtGui import QPixmap, QImage
from albumentations.pytorch import ToTensorV2
import albumentations as A
from pytorch_dcsaunet.DCSAU_Net import Model
from pytorch_tabnet.tab_model import TabNetRegressor

class ImageModel:
    def __init__(self):
        self.original_image = None
        self.processed_image = None
    def load_image(self, file_path):
        self.original_image = cv2.imread(file_path)
    def resize_image(self, width=300, height=300):
        if self.original_image is not None:
            self.processed_image = cv2.resize(self.original_image, (width, height))
    def convert_cv_qt(self, cv_img):
        if cv_img is None:
            return QPixmap()
        height, width, channel = cv_img.shape
        bytes_per_line = channel * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(q_img)

def get_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def inference_single_image(image: np.ndarray) -> np.ndarray:
    model_path = 'assets/epoch_last.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(image.shape) == 2:
        image = cv2.merge([image, image, image])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_transform()
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output >= 0.5).float()

    elapsed_time = time.time() - start_time
    mask = output.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    print("Inference done!")
    print(f"Inference time: {elapsed_time:.4f} seconds")
    cv2.imwrite('output_mask.png', mask)
    return mask

def predict_with_rfr(stats: dict) -> float:
    with open('./model/ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('./model/min_scaler.pkl', 'rb') as f:
        min_scaler = pickle.load(f)
    with open('./model/max_scaler.pkl', 'rb') as f:
        max_scaler = pickle.load(f)
    with open('./model/mean_scaler.pkl', 'rb') as f:
        mean_scaler = pickle.load(f)
    with open('./model/count_scaler.pkl', 'rb') as f:
        count_scaler = pickle.load(f)
    with open('./model/std_scaler.pkl', 'rb') as f:
        std_scaler = pickle.load(f)
    with open('./model/median_scaler.pkl', 'rb') as f:
        median_scaler = pickle.load(f)

    with open('./model/div_encoder.pkl', 'rb') as f:
        category_encoder = pickle.load(f)
    with open('./model/steel_encoder.pkl', 'rb') as f:
        steel_encoder = pickle.load(f)
    with open('./model/lot_encoder.pkl', 'rb') as f:
        lot_encoder = pickle.load(f)

    steel = stats["Steel"] if stats["Steel"] in steel_encoder.classes_ else 'other'
    lot = stats["Lot"] if stats["Lot"] in lot_encoder.classes_ else 'other'

    input_data = pd.DataFrame({
        'min_con_pt': [min_scaler.transform([[stats['Min']]])[0, 0]],
        'max_con_pt': [max_scaler.transform([[stats['Max']]])[0, 0]],
        'avg_con_pt': [mean_scaler.transform([[stats['Mean']]])[0, 0]],
        'cnt_con_pt': [count_scaler.transform([[stats['Count']]])[0, 0]],
        'std_con_pt': [std_scaler.transform([[stats['Std']]])[0, 0]],
        'med_con_pt': [median_scaler.transform([[stats['Median']]])[0, 0]],
        'category_enc': [category_encoder.transform([stats['Category']])[0]],
        'steel_enc': [steel_encoder.transform([steel])[0]],
        'lot_enc': [lot_encoder.transform([lot])[0]]
    })

    start_time = time.time()
    prediction = model.predict(input_data)[0]
    elapsed_time = time.time() - start_time
    print("Prediction done!")
    print(f"Prediction time: {elapsed_time:.4f} seconds")
    return prediction

def predict_with_tabnet(stats: dict, gs: float) -> dict:
    model = TabNetRegressor()
    model.load_model('assets/tabnet_model.zip')
    with open('./model/trained_columns.pkl', 'rb') as f:
        trained_columns = pickle.load(f)
    with open('./model/tabnet_encoder.pkl', 'rb') as f:
        encoders = pickle.load(f)

    input_row = {
        'div': stats['Category'],
        'steel_grade': stats['Steel'],
        'lot_no': stats['Lot'],
        'GS': gs,
        'mix': 0,
        'min_con': stats['Min'],
        'max_con': stats['Max'],
        'avg_con': stats['Mean'],
        'cnt_con': stats['Count'],
        'std_con': stats['Std'],
        'med_con': stats['Median']
    }
    df = pd.DataFrame([input_row])
    for col in ['div', 'steel_grade', 'lot_no']:
        val = df.at[0, col]
        if val not in encoders[col].classes_:
            print(f"Value '{val}' not in encoder classes for {col}, replacing with 'other'")
            val = 'other'
        encoded = encoders[col].transform([val])[0]
        df[col] = [encoded]
    df = df[trained_columns]
    assert df.shape[1] == len(trained_columns), f"Expected {len(trained_columns)} columns, got {df.shape[1]}"
    X = df.to_numpy(dtype=np.float32)

    start = time.time()
    preds = model.predict(X)[0]
    print(f"TabNet prediction time: {time.time() - start:.4f} seconds")
    return dict(zip(["TS", "YS", "RA", "EL"], preds))
