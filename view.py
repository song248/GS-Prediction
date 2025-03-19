from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt

class ImageView(QWidget):
    """이미지를 표시하는 UI 뷰"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """UI 레이아웃 설정 (절대 좌표 배치)"""
        self.setWindowTitle("이미지 업로드 및 전처리")  
        self.setGeometry(100, 100, 1280, 720)  # 창 크기 설정
        self.setStyleSheet("background-color: white;")  # 배경색 추가 (가독성 증가)

        # "Image Upload" 문구 (좌측 상단 배치)
        self.title_label = QLabel("Image Upload", self)
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignLeft)
        self.title_label.setGeometry(20, 20, 200, 40)

        # 업로드 버튼
        self.upload_button = QPushButton("이미지 업로드", self)
        self.upload_button.setFixedSize(120, 40)
        self.upload_button.setGeometry(20, 70, 120, 40)

        # 원본 이미지 문구
        self.original_label = QLabel("원본 이미지", self)
        self.original_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setGeometry(100, 130, 200, 40)

        # 전처리된 이미지 문구
        self.processed_label = QLabel("전처리된 이미지", self)
        self.processed_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setGeometry(800, 130, 200, 40)

        # 원본 이미지 QLabel
        self.original_image_label = QLabel(self)
        self.original_image_label.setGeometry(100, 180, 580, 400)
        self.original_image_label.setStyleSheet("border: 1px solid black;")  # 테두리 추가

        # 전처리된 이미지 QLabel
        self.processed_image_label = QLabel(self)
        self.processed_image_label.setGeometry(800, 180, 400, 400)
        self.processed_image_label.setStyleSheet("border: 1px solid black;")

    def set_original_image(self, pixmap):
        """원본 이미지 표시"""
        pixmap = pixmap.scaled(580, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_image_label.setPixmap(pixmap)

    def set_processed_image(self, pixmap):
        """전처리된 이미지 표시 (✅ QPixmap을 직접 받아 처리)"""
        try:
            pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.processed_image_label.setPixmap(pixmap)

        except Exception as e:
            print(f"이미지 표시 오류: {e}")
