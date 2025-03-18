import cv2
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
