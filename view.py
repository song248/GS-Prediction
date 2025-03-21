from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt

class ImageView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("이미지 업로드 및 처리 결과")
        self.setGeometry(100, 100, 1600, 720)  # ✅ 창 가로 크기 확장

        # Title
        self.title_label = QLabel("Image Upload", self)
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignLeft)
        self.title_label.setGeometry(20, 20, 200, 40)

        # Upload button
        self.upload_button = QPushButton("이미지 업로드", self)
        self.upload_button.setFixedSize(120, 40)
        self.upload_button.setGeometry(20, 70, 120, 40)

        # ✅ 입력 이미지 (왼쪽)
        self.original_label = QLabel("입력 이미지", self)
        self.original_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setGeometry(50, 130, 200, 40)  # ← 더 왼쪽으로 이동

        self.original_image_label = QLabel(self)
        self.original_image_label.setGeometry(50, 180, 560, 400)  # ← 왼쪽으로 이동
        self.original_image_label.setStyleSheet("border: 1px solid black;")

        # ✅ 전처리 이미지 (가운데)
        self.processed_label = QLabel("전처리된 이미지", self)
        self.processed_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setGeometry(660, 130, 200, 40)  # ← 기존보다 왼쪽

        self.processed_image_label = QLabel(self)
        self.processed_image_label.setGeometry(660, 180, 400, 400)
        self.processed_image_label.setStyleSheet("border: 1px solid black;")

        # ✅ 추론 결과 이미지 (오른쪽)
        self.result_label = QLabel("모델 추론 결과", self)
        self.result_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setGeometry(1130, 130, 200, 40)  # ← 기존보다 왼쪽

        self.result_image_label = QLabel(self)
        self.result_image_label.setGeometry(1130, 180, 400, 400)
        self.result_image_label.setStyleSheet("border: 1px solid black;")

    def set_original_image(self, pixmap):
        pixmap = pixmap.scaled(560, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_image_label.setPixmap(pixmap)

    def set_processed_image(self, pixmap):
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.processed_image_label.setPixmap(pixmap)

    def set_result_image(self, pixmap):
        pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.result_image_label.setPixmap(pixmap)
