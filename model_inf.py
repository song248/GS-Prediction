import os
import cv2
import numpy as np
import torch
import pickle
import pandas as pd
from torch.autograd import Variable
from pytorch_dcsaunet.DCSAU_Net import Model
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PyQt5.QtGui import QPixmap, QImage

class ImageModel:
    """이미지 데이터를 관리하는 모델"""
    def __init__(self):
        self.original_image = None
        self.processed_image = None

    def load_image(self, file_path):
        """이미지를 로드하여 원본 저장"""
        self.original_image = cv2.imread(file_path)

    def resize_image(self, width=300, height=300):
        """이미지를 지정한 크기로 리사이즈"""
        if self.original_image is not None:
            self.processed_image = cv2.resize(self.original_image, (width, height))

    def convert_cv_qt(self, cv_img):
        """OpenCV 이미지를 QPixmap으로 변환"""
        if cv_img is None:
            return QPixmap()
        height, width, channel = cv_img.shape
        bytes_per_line = channel * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(q_img)

# ✅ 이미지 전처리용 transform
def get_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ✅ PyTorch 모델 추론 (DCSAU-Net)
def inference_single_image(image: np.ndarray) -> np.ndarray:
    model_path = 'assets/epoch_last.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(image.shape) == 2:  # Grayscale → RGB
        image = cv2.merge([image, image, image])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_transform()
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output >= 0.5).float()

    mask = output.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    print("Inference done!")
    cv2.imwrite('output_mask.png', mask)
    return mask

# ✅ 랜덤 포레스트 회귀 예측 (stats → 예측값)
def predict_with_rfr(stats: dict) -> float:
    """
    stats: {
        'min_con_pt': float,
        'max_con_pt': float,
        'avg_con_pt': float,
        'cnt_con_pt': int,
        'std_con_pt': float,
        'med_con_pt': float
    }
    """
    model_path = './model/random_forest_regressor.pkl'
    with open(model_path, 'rb') as f:
        loaded_rfr = pickle.load(f)
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

    input_data = pd.DataFrame({
        'min_con_pt': [min_scaler.transform([[stats['Min']]])[0, 0]],
        'max_con_pt': [max_scaler.transform([[stats['Max']]])[0, 0]],
        'avg_con_pt': [mean_scaler.transform([[stats['Mean']]])[0, 0]],
        'cnt_con_pt': [count_scaler.transform([[stats['Count']]])[0, 0]],
        'std_con_pt': [std_scaler.transform([[stats['Std']]])[0, 0]],
        'med_con_pt': [median_scaler.transform([[stats['Median']]])[0, 0]],
    })

    prediction = loaded_rfr.predict(input_data)[0]
    return prediction
