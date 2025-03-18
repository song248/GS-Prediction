from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout

class ImageView(QWidget):
    """이미지를 표시하는 UI 뷰"""
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """UI 레이아웃 설정"""
        self.setWindowTitle("이미지 업로드 및 리사이즈")  
        self.setGeometry(100, 100, 1280, 720)  # 화면 크기를 1280x720(HD)로 설정

        self.layout = QVBoxLayout()

        # 업로드 버튼
        self.upload_button = QPushButton("이미지 업로드")
        self.layout.addWidget(self.upload_button)

        # 이미지 표시 레이아웃
        self.image_layout = QHBoxLayout()
        self.original_label = QLabel("원본 이미지")
        self.processed_label = QLabel("리사이즈 이미지")

        self.image_layout.addWidget(self.original_label)
        self.image_layout.addWidget(self.processed_label)

        self.layout.addLayout(self.image_layout)
        self.setLayout(self.layout)
