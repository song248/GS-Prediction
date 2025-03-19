from PyQt5.QtWidgets import QApplication
import sys
from model import ImageModel  # ✅ model 추가
from view import ImageView
from controller import ImageController

if __name__ == "__main__":
    app = QApplication(sys.argv)

    model = ImageModel()  # ✅ model 추가
    view = ImageView()
    controller = ImageController(model, view)  # ✅ model을 전달하도록 수정

    view.show()
    sys.exit(app.exec_())
