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
        self.setWindowTitle("이미지 업로드 및 리사이즈")  
        self.setGeometry(100, 100, 1280, 720)  # 720P 해상도 설정
        self.setStyleSheet("background-color: white;")  # 배경색 추가 (가독성 증가)

        # "Image Upload" 문구 (좌측 상단 배치)
        self.title_label = QLabel("Image Upload", self)
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))  # 굵은 글씨 설정
        self.title_label.setAlignment(Qt.AlignLeft)
        self.title_label.setGeometry(20, 20, 200, 40)  # 🔹 (x=20, y=20) 위치에 배치

        # 업로드 버튼 (절대 좌표 배치)
        self.upload_button = QPushButton("이미지 업로드", self)
        self.upload_button.setFixedSize(120, 40)  # 버튼 크기 고정
        self.upload_button.setGeometry(20, 70, 120, 40)  # 🔹 (x=20, y=70) 위치에 배치

        # 원본 이미지 문구 (버튼 바로 아래에 배치)
        self.original_label = QLabel("원본 이미지", self)
        self.original_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setGeometry(100, 130, 200, 40)  # 🔹 (x=100, y=130) 버튼 아래에 배치

        # 리사이즈 이미지 문구 (원본 이미지 문구와 같은 높이에 배치)
        self.processed_label = QLabel("리사이즈 이미지", self)
        self.processed_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setGeometry(800, 130, 200, 40)  # 🔹 (x=800, y=130) 동일한 높이로 맞춤

        # ✅ 원본 이미지 크기 확대 (600x600)
        self.original_image_label = QLabel(self)
        self.original_image_label.setGeometry(100, 180, 580, 400)  # 🔹 (x=100, y=180, width=600, height=600)
        self.original_image_label.setStyleSheet("border: 1px solid black;")  # 테두리 추가 (이미지 공간 표시)

        # ✅ 리사이즈된 이미지 크기 유지 (500x500)
        self.processed_image_label = QLabel(self)
        self.processed_image_label.setGeometry(800, 180, 400, 400)  # 🔹 (x=800, y=180, width=500, height=500)
        self.processed_image_label.setStyleSheet("border: 1px solid black;")  # 테두리 추가 (이미지 공간 표시)

    def set_original_image(self, pixmap):
        """원본 이미지 표시 (크기를 QLabel 크기에 맞게 조정)"""
        self.original_image_label.setPixmap(pixmap.scaled(580, 400, Qt.KeepAspectRatio))

    def set_processed_image(self, pixmap):
        """리사이즈된 이미지 표시 (크기를 QLabel 크기에 맞게 조정)"""
        self.processed_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
