import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage

class ImageController:
    """이미지 로드 및 전처리를 제어하는 컨트롤러"""
    def __init__(self, model, view):
        self.model = model
        self.view = view

        # 업로드 버튼 클릭 시, 컨트롤러의 업로드 함수 실행
        self.view.upload_button.clicked.connect(self.upload_image)

    def upload_image(self):
        """파일 선택 후 이미지 처리"""
        file_path, _ = QFileDialog.getOpenFileName(self.view, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            self.model.load_image(file_path)  # 원본 이미지 로드

            # 원본 이미지 변환하여 View 업데이트
            original_pixmap = self.convert_cv_qt(self.model.original_image)
            self.view.set_original_image(original_pixmap)

            # ✅ 전처리 수행
            processed_img = self.preprocess_image(self.model.original_image)

            # ✅ 전처리된 이미지도 QPixmap으로 변환하여 전달
            processed_pixmap = self.convert_cv_qt(processed_img)
            self.view.set_processed_image(processed_pixmap)

    def preprocess_image(self, img):
        """이미지 전처리 과정"""
        try:
            # ✅ 1. 이진화 (Grayscale 변환)
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # ✅ 2. 이미지 리사이즈 (1440, 1024)
            resized_image = cv2.resize(gray_image, (1440, 1024))

            # ✅ 3. 중앙부 512x512 크롭
            h, w = resized_image.shape  # 흑백 이미지이므로 채널 없음
            center_x, center_y = w // 2, h // 2
            cropped_image = resized_image[
                center_y - 256:center_y + 256, center_x - 256:center_x + 256
            ]

            # ✅ 4. Bilateral Filter 적용
            filtered_image = cv2.bilateralFilter(cropped_image, -1, 10, 10)

            # ✅ 5. Dilation 적용 (커널: 3x3)
            kernel = np.ones((3, 3), np.uint8)
            dilated_image = cv2.dilate(filtered_image, kernel, iterations=1)

            return dilated_image

        except Exception as e:
            print(f"이미지 전처리 오류: {e}")
            return img  # 오류 발생 시 원본 이미지 반환

    def convert_cv_qt(self, cv_img):
        """OpenCV 이미지를 QPixmap으로 변환 (Grayscale 또는 RGB 지원)"""
        if cv_img is None:
            return QPixmap()

        # ✅ Grayscale (1채널) 이미지 처리
        if len(cv_img.shape) == 2:
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # ✅ 컬러 (RGB) 이미지 처리
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        return QPixmap.fromImage(q_img)
