import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from skimage.morphology import skeletonize
from model_inf import inference_single_image

class ImageController:
    """이미지 로드, 전처리, 추론을 제어하는 컨트롤러"""
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.upload_button.clicked.connect(self.upload_image)

    def upload_image(self):
        """이미지 업로드 및 전체 파이프라인 실행"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.view, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            # 1. 원본 이미지
            self.model.load_image(file_path)
            original_pixmap = self.convert_cv_qt(self.model.original_image)
            self.view.set_original_image(original_pixmap)

            # 2. 전처리
            processed_img = self.preprocess_image(self.model.original_image)
            processed_pixmap = self.convert_cv_qt(processed_img)
            self.view.set_processed_image(processed_pixmap)

            # 3. 모델 추론 결과 → 후처리 적용 → 결과 표시
            result_mask = inference_single_image(processed_img)
            final_img = self.postprocess_mask(result_mask)
            result_pixmap = self.convert_cv_qt(final_img)
            self.view.set_result_image(result_pixmap)

    def preprocess_image(self, img):
        """이미지 전처리"""
        try:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (1440, 1024))
            h, w = resized_image.shape
            center_x, center_y = w // 2, h // 2
            cropped_image = resized_image[
                center_y - 256:center_y + 256, center_x - 256:center_x + 256
            ]
            filtered_image = cv2.bilateralFilter(cropped_image, -1, 10, 10)
            inverted = cv2.bitwise_not(filtered_image)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            thick_lines = cv2.bitwise_not(dilated)
            return thick_lines
        except Exception as e:
            print(f"[전처리 오류] {e}")
            return img

    def convert_cv_qt(self, cv_img):
        """OpenCV 이미지를 QPixmap으로 변환"""
        if cv_img is None:
            return QPixmap()
        if len(cv_img.shape) == 2:
            height, width = cv_img.shape
            bytes_per_line = width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            height, width, channel = cv_img.shape
            bytes_per_line = 3 * width
            q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(q_img)

    @staticmethod
    def postprocess_mask(mask: np.ndarray) -> np.ndarray:
        """추론된 마스크 후처리 → skeleton → 흰 배경 + 검정 선"""
        _, binary_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
        pad_img = np.pad(binary_img, pad_width=((5,5),(5,5)), mode='constant', constant_values=0)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(pad_img, kernel, iterations=10)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        skeleton = skeletonize(closed // 255).astype(np.uint8) * 255
        connected_skeleton = connect_endpoints_to_edges(skeleton, threshold=50)
        connected_skeleton = connected_skeleton[5:-5, 5:-5]
        final_img = np.ones_like(connected_skeleton, dtype=np.uint8) * 255
        final_img[connected_skeleton == 255] = 0
        real_final = cv2.cvtColor(final_img, cv2.COLOR_GRAY2BGR)
        
        return real_final


# 끝점을 정확히 찾는 함수
def find_endpoints(skel):
    endpoints = []
    h, w = skel.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 255:
                neighborhood = skel[y-1:y+2, x-1:x+2]
                if np.sum(neighborhood == 255) == 2:  # 본인 포함 2픽셀(즉 주변 픽셀 1개만 연결된 상태)
                    endpoints.append((y, x))
    return endpoints

# 끝점들을 가장자리와 연결하는 함수
def connect_endpoints_to_edges(skeleton, threshold=50):
    h, w = skeleton.shape
    endpoints = find_endpoints(skeleton)
    connected = skeleton.copy()

    for y, x in endpoints:
        # 각 끝점에서 가장 가까운 가장자리 판단
        distances = {'left': x, 'right': w - x - 1, 'top': y, 'bottom': h - y - 1}
        closest_edge, min_dist = min(distances.items(), key=lambda item: item[1])

        # threshold 내의 가장자리로 연결
        if min_dist < threshold:
            if closest_edge == 'left':
                cv2.line(connected, (x, y), (0, y), 255, 1)
            elif closest_edge == 'right':
                cv2.line(connected, (x, y), (w - 1, y), 255, 1)
            elif closest_edge == 'top':
                cv2.line(connected, (x, y), (x, 0), 255, 1)
            elif closest_edge == 'bottom':
                cv2.line(connected, (x, y), (x, h - 1), 255, 1)

    return connected