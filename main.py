from PyQt5.QtWidgets import QApplication
import sys
from model import ImageModel
from view import ImageView
from controller import ImageController

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # MVC 패턴 초기화
    model = ImageModel()
    view = ImageView()
    controller = ImageController(model, view)

    # 창 띄우기
    view.setWindowTitle("Grain Score Prediction")
    view.show()

    sys.exit(app.exec_())  # PyQt 이벤트 루프 실행
