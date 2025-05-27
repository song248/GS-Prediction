import cv2
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from skimage.morphology import skeletonize
from model_inf import inference_single_image, predict_with_rfr, predict_with_tabnet

class ImageController:
    def __init__(self, model, view):
        self.model = model
        self.view = view
        self.view.upload_button.clicked.connect(self.upload_image)
        self.view.predict_button.clicked.connect(self.run_prediction)
        self.result_mask = None
        self.result_image = None

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self.view, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if not file_path:
            return
        self.model.load_image(file_path)
        self.view.set_original_image(self.convert_cv_qt(self.model.original_image))
        processed_img = self.preprocess_image(self.model.original_image)
        self.view.set_processed_image(self.convert_cv_qt(processed_img))
        self.result_mask = inference_single_image(processed_img)

    def run_prediction(self):
        if self.result_mask is None:
            print("[예측 오류] 먼저 이미지를 업로드하세요.")
            return
        real_final, contours = self.postprocess_mask(self.result_mask)
        self.result_image = real_final
        self.view.set_result_image(self.convert_cv_qt(real_final))
        areas = [cv2.contourArea(c) for c in contours]
        if areas:
            stats = {
                "Count": len(areas),
                "Min": round(min(areas), 2),
                "Max": round(max(areas), 2),
                "Mean": round(np.mean(areas), 2),
                "Std": round(np.std(areas), 2),
                "Median": round(np.median(areas), 2),
            }
        else:
            stats = {k: 0 for k in ["Count", "Min", "Max", "Mean", "Std", "Median"]}

        for i, key in enumerate(stats):
            self.view.set_table_item(i, key, stats[key])
        category = self.view.get_category_input()
        steel = self.view.get_steel_input()
        lot = self.view.get_lot_input()

        input_data = {
            **stats,
            "Category": category,
            "Steel": steel,
            "Lot": lot
        }

        gs_prediction = predict_with_rfr(input_data)
        self.view.set_prediction_value(gs_prediction)
        tabnet_results = predict_with_tabnet(input_data, gs_prediction)
        tabnet_keys = ["TS", "YS", "RA", "EL"]
        for i, key in enumerate(tabnet_keys, start=6):
            self.view.set_table_item(i, key, round(tabnet_results[key], 2))

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
            return cv2.bitwise_not(dilated)
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
        contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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