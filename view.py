from PyQt5.QtWidgets import QWidget, QPushButton, QLabel
from PyQt5.QtGui import QFont
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

        # 원본 이미지 문구 (버튼 아래 배치)
        self.original_label = QLabel("원본 이미지", self)
        self.original_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setGeometry(150, 300, 200, 40)  # 🔹 원하는 위치(x=150, y=300)에 배치

        # 리사이즈 이미지 문구 (원본 이미지 문구와 같은 높이에 배치)
        self.processed_label = QLabel("리사이즈 이미지", self)
        self.processed_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setGeometry(800, 300, 200, 40)  # 🔹 원하는 위치(x=800, y=300)에 배치
