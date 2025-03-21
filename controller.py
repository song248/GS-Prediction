import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from model_inf import inference_single_image  # ✅ 모델 추론 함수 가져오기

class ImageController:
    """이미지 로드, 전처리, 추론을 제어하는 컨트롤러"""
    def __init__(self, model, view):
        self.model = model
        self.view = view

        # 버튼 클릭 연결
        self.view.upload_button.clicked.connect(self.upload_image)

    def upload_image(self):
        """이미지 업로드 및 전체 파이프라인 실행"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.view, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            # ✅ 1. 원본 이미지 로드 및 표시
            self.model.load_image(file_path)
            original_pixmap = self.convert_cv_qt(self.model.original_image)
            self.view.set_original_image(original_pixmap)

            # ✅ 2. 전처리 이미지 생성 및 표시
            processed_img = self.preprocess_image(self.model.original_image)
            processed_pixmap = self.convert_cv_qt(processed_img)
            self.view.set_processed_image(processed_pixmap)

            # ✅ 3. 모델 추론 결과 이미지 생성 및 표시
            result_mask = inference_single_image(processed_img)
            result_pixmap = self.convert_cv_qt(result_mask)
            self.view.set_result_image(result_pixmap)

    def preprocess_image(self, img):
        """이미지 전처리: grayscale → resize → crop → filter → dilate"""
        try:
            # 1. Grayscale
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 2. Resize
            resized_image = cv2.resize(gray_image, (1440, 1024))

            # 3. 중앙부 크롭 (512x512)
            h, w = resized_image.shape  # 흑백 이미지이므로 채널 없음
            center_x, center_y = w // 2, h // 2
            cropped_image = resized_image[
                center_y - 256:center_y + 256, center_x - 256:center_x + 256
            ]

            # 4. Bilateral Filter
            filtered_image = cv2.bilateralFilter(cropped_image, -1, 10, 10)
            inverted = cv2.bitwise_not(filtered_image)

            # 5. Dilation
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            thick_lines = cv2.bitwise_not(dilated)

            return thick_lines

        except Exception as e:
            print(f"[전처리 오류] {e}")
            return img

    def convert_cv_qt(self, cv_img):
        """OpenCV 이미지를 PyQt용 QPixmap으로 변환"""
        if cv_img is None:
            return QPixmap()

        if len(cv_img.shape) == 2:  # Grayscale
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(
                cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
        else:  # RGB
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(
                cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()

        return QPixmap.fromImage(q_img)
