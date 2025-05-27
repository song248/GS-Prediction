from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLabel, QTableWidget, QTableWidgetItem,
    QComboBox, QLineEdit
)
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtCore import Qt

class ImageView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("이미지 업로드 및 처리 결과")
        self.setGeometry(100, 100, 1900, 720)
        self.setStyleSheet("background-color: white;")

        # 제목
        self.title_label = QLabel("Image Upload", self)
        self.title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignLeft)
        self.title_label.setGeometry(20, 20, 200, 40)

        # 업로드 버튼
        self.upload_button = QPushButton("Upload", self)
        self.upload_button.setFixedSize(120, 40)
        self.upload_button.setGeometry(20, 70, 120, 40)

        # 드롭다운 (A, B, C, D)
        self.label_dropdown = QLabel("Rater", self)
        self.label_dropdown.setFont(QFont("Arial", 10, QFont.Bold))
        self.label_dropdown.setGeometry(220, 45, 150, 20)

        self.category_combo = QComboBox(self)
        self.category_combo.setGeometry(220, 70, 120, 30)
        self.category_combo.addItems(["A", "B", "C", "D"])

        # steel 입력칸
        self.label_steel = QLabel("Steel", self)
        self.label_steel.setFont(QFont("Arial", 10, QFont.Bold))
        self.label_steel.setGeometry(360, 45, 100, 20)

        self.steel_input = QLineEdit(self)
        self.steel_input.setGeometry(360, 70, 120, 30)

        # lot 입력칸
        self.label_lot = QLabel("Lot", self)
        self.label_lot.setFont(QFont("Arial", 10, QFont.Bold))
        self.label_lot.setGeometry(500, 45, 100, 20)

        self.lot_input = QLineEdit(self)
        self.lot_input.setGeometry(500, 70, 120, 30)

        # 예측 버튼
        self.predict_button = QPushButton("Run Prediction", self)
        self.predict_button.setFixedSize(160, 40)
        self.predict_button.setGeometry(640, 70, 160, 40)

        # 이미지 및 테이블
        self.original_label = QLabel("Input Image", self)
        self.original_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.original_label.setGeometry(50, 150, 200, 30)
        self.original_image_label = QLabel(self)
        self.original_image_label.setGeometry(50, 180, 560, 400)
        self.original_image_label.setStyleSheet("border: 1px solid black;")

        self.processed_label = QLabel("Processed Image", self)
        self.processed_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.processed_label.setGeometry(660, 150, 200, 30)
        self.processed_image_label = QLabel(self)
        self.processed_image_label.setGeometry(660, 180, 400, 400)
        self.processed_image_label.setStyleSheet("border: 1px solid black;")

        self.result_label = QLabel("Inference Image", self)
        self.result_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.result_label.setGeometry(1130, 150, 200, 30)
        self.result_image_label = QLabel(self)
        self.result_image_label.setGeometry(1130, 180, 400, 400)
        self.result_image_label.setStyleSheet("border: 1px solid black;")

        # 테이블
        self.table = QTableWidget(self)
        self.table.setRowCount(10)
        self.table.setColumnCount(2)
        self.table.setGeometry(1560, 180, 300, 380)
        self.table.setHorizontalHeaderLabels(["Attr", "Value"])
        self.table.verticalHeader().setVisible(False)

        self.table.setShowGrid(True)
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: rgba(0, 0, 0, 80);
                border: none;
            }
            QTableWidget::item {
                border: 1px solid rgba(0, 0, 0, 80);
                padding: 2px;
            }
        """)
        self.table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                border: 1px solid rgba(0, 0, 0, 80);
                background-color: #f8f8f8;
                padding: 4px;
            }
        """)

        # 예측 결과 표시
        self.prediction_label = QLabel("Grain Score: N/A", self)
        self.prediction_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.prediction_label.setGeometry(1560, 580, 300, 40)

    # 이미지 표시
    def set_original_image(self, pixmap):
        self.original_image_label.setPixmap(pixmap.scaled(560, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_processed_image(self, pixmap):
        self.processed_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_result_image(self, pixmap):
        self.result_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_table_item(self, row, key, value):
        self.table.setItem(row, 0, QTableWidgetItem(str(key)))
        self.table.setItem(row, 1, QTableWidgetItem(str(value)))

    def set_prediction_value(self, value):
        self.prediction_label.setText(f"Grain Score: {value:.2f}")

    def get_category_input(self):
        return self.category_combo.currentText()

    def get_steel_input(self):
        return self.steel_input.text()

    def get_lot_input(self):
        return self.lot_input.text()
