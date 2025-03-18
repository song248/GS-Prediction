from PyQt5.QtWidgets import QFileDialog

class ImageController:
    """이미지 업로드 및 변환을 제어하는 컨트롤러"""
    def __init__(self, model, view):
        self.model = model
        self.view = view

        # 업로드 버튼 클릭 시 함수 실행
        self.view.upload_button.clicked.connect(self.upload_image)

    def upload_image(self):
        """파일 선택 후 이미지 처리"""
        file_path, _ = QFileDialog.getOpenFileName(None, "이미지 선택", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            self.model.load_image(file_path)
            self.model.resize_image()

            # View 업데이트
            original_pixmap = self.model.convert_cv_qt(self.model.original_image)
            processed_pixmap = self.model.convert_cv_qt(self.model.processed_image)

            self.view.original_label.setPixmap(original_pixmap)
            self.view.processed_label.setPixmap(processed_pixmap)
