import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from skimage.morphology import skeletonize
from model_inf import inference_single_image, predict_with_rfr

class ImageController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.upload_button.clicked.connect(self.upload_image)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self.view, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.model.load_image(file_path)
            original_pixmap = self.convert_cv_qt(self.model.original_image)
            self.view.set_original_image(original_pixmap)

            processed_img = self.preprocess_image(self.model.original_image)
            processed_pixmap = self.convert_cv_qt(processed_img)
            self.view.set_processed_image(processed_pixmap)

            result_mask = inference_single_image(processed_img)
            real_final, contours = self.postprocess_mask(result_mask)
            result_pixmap = self.convert_cv_qt(real_final)
            self.view.set_result_image(result_pixmap)

            # ✅ 통계 계산
            areas = [cv2.contourArea(c) for c in contours]
            if areas:
                stats = {
                    "cnt_con_pt": len(areas),
                    "min_con_pt": round(min(areas), 2),
                    "max_con_pt": round(max(areas), 2),
                    "avg_con_pt": round(np.mean(areas), 2),
                    "std_con_pt": round(np.std(areas), 2),
                    "med_con_pt": round(np.median(areas), 2),
                }
            else:
                stats = {k: 0 for k in [
                    "cnt_con_pt", "min_con_pt", "max_con_pt",
                    "avg_con_pt", "std_con_pt", "med_con_pt"
                ]}

            # ✅ 테이블 채우기
            display_keys = {
                "cnt_con_pt": "Count",
                "min_con_pt": "Min",
                "max_con_pt": "Max",
                "avg_con_pt": "Mean",
                "std_con_pt": "Std",
                "med_con_pt": "Median"
            }
            for i, key in enumerate(display_keys):
                self.view.set_table_item(i, display_keys[key], stats[key])

            # ✅ 모델 예측
            prediction = predict_with_rfr({
                "Min": stats["min_con_pt"],
                "Max": stats["max_con_pt"],
                "Mean": stats["avg_con_pt"],
                "Count": stats["cnt_con_pt"],
                "Std": stats["std_con_pt"],
                "Median": stats["med_con_pt"]
            })
            self.view.set_prediction_value(prediction)

    def preprocess_image(self, img):
        try:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (1440, 1024))
            h, w = resized_image.shape
            cx, cy = w // 2, h // 2
            cropped = resized_image[cy - 256:cy + 256, cx - 256:cx + 256]
            filtered = cv2.bilateralFilter(cropped, -1, 10, 10)
            inverted = cv2.bitwise_not(filtered)
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            thick_lines = cv2.bitwise_not(dilated)
            return thick_lines
        except Exception as e:
            print(f"[전처리 오류] {e}")
            return img

    def convert_cv_qt(self, cv_img):
        if cv_img is None:
            return QPixmap()
        if len(cv_img.shape) == 2:
            h, w = cv_img.shape
            q_img = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = cv_img.shape
            q_img = QImage(cv_img.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(q_img)

    @staticmethod
    def postprocess_mask(mask: np.ndarray):
        """후처리 + contour detection + 시각화"""
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY_INV)
        padded = np.pad(binary, ((5, 5), (5, 5)), mode='constant', constant_values=0)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(padded, kernel, iterations=10)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
        skeleton = skeletonize(closed // 255).astype(np.uint8) * 255
        connected = connect_endpoints_to_edges(skeleton, threshold=50)
        cropped_back = connected[5:-5, 5:-5]
        final_img = np.ones_like(cropped_back, dtype=np.uint8) * 255
        final_img[cropped_back == 255] = 0

        eroded = cv2.erode(final_img, kernel, iterations=2)
        contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_color = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image_color, contours, -1, (0, 255, 0), 2)

        return image_color, contours

def find_endpoints(skel):
    endpoints = []
    h, w = skel.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel[y, x] == 255:
                area = skel[y-1:y+2, x-1:x+2]
                if np.sum(area == 255) == 2:
                    endpoints.append((y, x))
    return endpoints

def connect_endpoints_to_edges(skeleton, threshold=50):
    h, w = skeleton.shape
    endpoints = find_endpoints(skeleton)
    connected = skeleton.copy()
    for y, x in endpoints:
        d = {'left': x, 'right': w - x - 1, 'top': y, 'bottom': h - y - 1}
        edge, dist = min(d.items(), key=lambda i: i[1])
        if dist < threshold:
            if edge == 'left':
                cv2.line(connected, (x, y), (0, y), 255, 1)
            elif edge == 'right':
                cv2.line(connected, (x, y), (w - 1, y), 255, 1)
            elif edge == 'top':
                cv2.line(connected, (x, y), (x, 0), 255, 1)
            elif edge == 'bottom':
                cv2.line(connected, (x, y), (x, h - 1), 255, 1)
    return connected
