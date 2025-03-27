from PyQt5.QtWidgets import QApplication
import sys
from model_inf import ImageModel
from view import ImageView
from controller import ImageController

if __name__ == "__main__":
    app = QApplication(sys.argv)

    model = ImageModel()
    view = ImageView()
    controller = ImageController(model, view)

    view.show()
    sys.exit(app.exec_())
 