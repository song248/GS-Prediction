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
        self.setGeometry(100, 100, 1680, 590)
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

        # 이미지
        self.original_label = QLabel("Input Image", self)
        self.original_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.original_label.setGeometry(30, 140, 200, 30)
        self.original_image_label = QLabel(self)
        self.original_image_label.setGeometry(30, 170, 560, 400)
        self.original_image_label.setStyleSheet("border: 1px solid black;")

        self.processed_label = QLabel("Processed Image", self)
        self.processed_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.processed_label.setGeometry(610, 140, 200, 30)
        self.processed_image_label = QLabel(self)
        self.processed_image_label.setGeometry(610, 170, 400, 400)
        self.processed_image_label.setStyleSheet("border: 1px solid black;")

        self.result_label = QLabel("Inference Image", self)
        self.result_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.result_label.setGeometry(1030, 140, 200, 30)
        self.result_image_label = QLabel(self)
        self.result_image_label.setGeometry(1030, 170, 400, 400)
        self.result_image_label.setStyleSheet("border: 1px solid black;")

        # Origin Table
        self.stat_table = QTableWidget(self)
        self.stat_table.setRowCount(6)
        self.stat_table.setColumnCount(2)
        self.stat_table.setGeometry(1450, 170, 300, 210)
        self.stat_table.setHorizontalHeaderLabels(["Attr", "Value"])
        self.stat_table.verticalHeader().setVisible(False)
        self.stat_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.stat_table.setFixedHeight(self.stat_table.rowHeight(0) * 6 + self.stat_table.horizontalHeader().height())

        # Result Table
        self.result_table = QTableWidget(self)
        self.result_table.setRowCount(5)
        self.result_table.setColumnCount(2)
        self.result_table.setGeometry(1450, 390, 300, 180)
        self.result_table.setHorizontalHeaderLabels(["Attr", "Value"])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.result_table.setFixedHeight(self.result_table.rowHeight(0) * 5 + self.result_table.horizontalHeader().height())

        for table in [self.stat_table, self.result_table]:
            table.setShowGrid(True)
            table.setStyleSheet("""
                QTableWidget {
                    gridline-color: rgba(0, 0, 0, 80);
                    border: none;
                }
                QTableWidget::item {
                    border: 1px solid rgba(0, 0, 0, 80);
                    padding: 2px;
                }
            """)
            table.horizontalHeader().setStyleSheet("""
                QHeaderView::section {
                    border: 1px solid rgba(0, 0, 0, 80);
                    background-color: #f8f8f8;
                    padding: 4px;
                }
            """)

    # 이미지 표시
    def set_original_image(self, pixmap):
        self.original_image_label.setPixmap(pixmap.scaled(560, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_processed_image(self, pixmap):
        self.processed_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_result_image(self, pixmap):
        self.result_image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def set_stat_table_item(self, row, key, value):
        self.stat_table.setItem(row, 0, QTableWidgetItem(str(key)))
        self.stat_table.setItem(row, 1, QTableWidgetItem(str(value)))

    def set_result_table_item(self, row, key, value):
        self.result_table.setItem(row, 0, QTableWidgetItem(str(key)))
        self.result_table.setItem(row, 1, QTableWidgetItem(str(value)))

    def set_prediction_value(self, value):
        self.set_result_table_item(0, "GS", round(value, 2))

    def get_category_input(self):
        return self.category_combo.currentText()

    def get_steel_input(self):
        return self.steel_input.text()

    def get_lot_input(self):
        return self.lot_input.text()
