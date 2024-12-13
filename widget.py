import sys

from PySide6.QtWidgets import QApplication, QWidget, \
        QTableWidget, QTableWidgetItem
from PySide6.QtGui import QPixmap
from ui_form import Ui_Widget

import pandas as pd
import numpy as np
from sklearn.datasets import load_digits

from functions.helper_functions import (generate_letters,
                                        train_test_valid_split)
from NeuralNetworkVisualizer import NeuralNetworkVisualizer
from network import NeuralNetwork


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
