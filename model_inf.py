import os
import cv2
import numpy as np

import torch
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

def get_transform():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def inference_single_image(image: np.ndarray) -> np.ndarray:
    model_path = 'assets/epoch_last.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if len(image.shape) == 2:  # Grayscale (H, W)
        image = cv2.merge([image, image, image])  # → (H, W, 3)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[-1] == 3 else image

    transform = get_transform()
    transformed = transform(image=image_rgb)
    image_tensor = transformed['image'].unsqueeze(0).to(device)  # [1, C, H, W]

    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output >= 0.5).float()

    mask = output.squeeze().cpu().numpy()  # [H, W]
    mask = (mask * 255).astype(np.uint8)

    print(f"Inference done!")
    return mask