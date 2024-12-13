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

        self.ui.data_set_combo_box \
            .currentIndexChanged.connect(self.dataset_index_changed)

        self.ui.inputs_spin_box.valueChanged.connect(self.inputs_changed)
        self.ui.outputs_spin_box.valueChanged.connect(self.outputs_changed)

        self.ui.hidden_layers_spin_box \
            .valueChanged.connect(self.hidden_layers_changed)
        self.ui.layer_1_spin_box.valueChanged.connect(self.n_layer_changed)
        self.ui.layer_2_spin_box.valueChanged.connect(self.n_layer_changed)
        self.ui.layer_3_spin_box.valueChanged.connect(self.n_layer_changed)
        self.ui.layer_4_spin_box.valueChanged.connect(self.n_layer_changed)
        self.ui.layer_5_spin_box.valueChanged.connect(self.n_layer_changed)
        self.ui.layer_6_spin_box.valueChanged.connect(self.n_layer_changed)
        self.ui.layer_7_spin_box.valueChanged.connect(self.n_layer_changed)
        self.ui.layer_8_spin_box.valueChanged.connect(self.n_layer_changed)

        self.ui.weights_init_type_combo_box \
            .currentIndexChanged.connect(self.init_type_changed)
        self.ui.loss_function_combo_box \
            .currentIndexChanged.connect(self.loss_function_changed)
        self.ui.activation_function_combo_box \
            .currentIndexChanged.connect(self.activation_function_changed)

        self.ui.overall_neurons_number_label \
            .setNum(self.get_number_of_neurons())
        self.ui.overall_weigts_number_label \
            .setNum(self.get_number_of_weights())

        self.ui.learning_rate_double_spin_box \
            .valueChanged.connect(self.learning_rate_changed)
        self.ui.epochs_number_spin_box \
            .valueChanged.connect(self.epochs_number_changed)

        self.network_visualizer = NeuralNetworkVisualizer()
        self.ui.network_overview_vertical_layout \
            .addWidget(self.network_visualizer)

        self.ui.add_row_button.clicked.connect(self.add_row_pressed)
        self.ui.remove_row_button.clicked.connect(self.remove_row_pressed)
        self.ui.save_data_button.clicked.connect(self.save_button_pressed)

        self.ui.train_button.clicked.connect(self.train_button_pressed)
        self.ui.test_button.clicked.connect(self.test_button_pressed)
        self.ui.predict_button.clicked.connect(self.predict_button_pressed)

        self.ui.tab_widget.tabBarClicked.connect(self.tab_bar_clicked)

        self.blockSignals(True)
        self.ui.weights_init_type_combo_box.setCurrentIndex(-1)
        self.ui.loss_function_combo_box.setCurrentIndex(-1)
        self.ui.activation_function_combo_box.setCurrentIndex(-1)
        self.ui.data_set_combo_box.setCurrentIndex(-1)
        self.ui.hidden_layers_spin_box.setValue(0)
        self.ui.layer_1_spin_box.setValue(0)
        self.ui.learning_rate_double_spin_box.setValue(-0.001)
        self.ui.epochs_number_spin_box.setValue(-1)
        self.blockSignals(False)

        self.ui.weights_init_type_combo_box.setCurrentIndex(0)
        self.ui.loss_function_combo_box.setCurrentIndex(0)
        self.ui.activation_function_combo_box.setCurrentIndex(0)
        self.ui.hidden_layers_spin_box.setValue(1)
        self.ui.data_set_combo_box.setCurrentIndex(1)
        self.ui.learning_rate_double_spin_box.setValue(0.001)
        self.ui.epochs_number_spin_box.setValue(1000)

        self.ui.custom_data_table.insertRow(0)

    # Handling signals

    def dataset_index_changed(self, index: int):
        match index:
            case 0:
                self.handle_numbers()
            case 1:
                self.handle_letters()
            case 2:
                self.handle_custom()

        self.ui.inputs_spin_box.setValue(self.inputs)
        self.ui.outputs_spin_box.setValue(self.outputs)

    def inputs_changed(self, value: int):
        self.inputs = value
        self.layers[0] = value
        self.network_visualizer.draw_network(self.layers)

        self.ui.overall_neurons_number_label \
            .setNum(self.get_number_of_neurons())
        self.ui.overall_weigts_number_label \
            .setNum(self.get_number_of_weights())

    def outputs_changed(self, value: int):
        self.outputs = value
        self.layers[-1] = value
        self.network_visualizer.draw_network(self.layers)

        self.ui.overall_neurons_number_label \
            .setNum(self.get_number_of_neurons())
        self.ui.overall_weigts_number_label \
            .setNum(self.get_number_of_weights())

    def hidden_layers_changed(self, value: int):
        if len(self.hidden_layers) > value:
            self.hidden_layers = self.hidden_layers[:value]
        elif len(self.hidden_layers) < value:
            self.hidden_layers.append(0)

        self.layers[1:-1] = self.hidden_layers

        layer_spin_boxes = [f'layer_{i}_spin_box' for i in range(1, 9)]
        command = 'self.set_widget_availability(self.ui.{}, {})'

        for spin_box in layer_spin_boxes:
            exec(command.format(spin_box, False))
        for i in range(value):
            exec(command.format(layer_spin_boxes[i], True))

        self.network_visualizer.draw_network(self.layers)

        self.ui.overall_neurons_number_label \
            .setNum(self.get_number_of_neurons())
        self.ui.overall_weigts_number_label \
            .setNum(self.get_number_of_weights())

    def n_layer_changed(self, value: int):
        widget_name = self.sender().objectName()
        index = [int(n) for n in widget_name.split('_') if n.isdigit()][0]
        self.change_hidden_layers(value, index - 1)

        self.network_visualizer.draw_network(self.layers)

        self.ui.overall_neurons_number_label \
            .setNum(self.get_number_of_neurons())
        self.ui.overall_weigts_number_label \
            .setNum(self.get_number_of_weights())

    def init_type_changed(self, _):
        self.init_type = self.ui.weights_init_type_combo_box.currentText()

    def loss_function_changed(self, _):
        self.loss = self.ui.loss_function_combo_box.currentText()

    def activation_function_changed(self, _):
        self.activation = self.ui.activation_function_combo_box.currentText()

    def learning_rate_changed(self, value: float):
        self.learning_rate = value

    def epochs_number_changed(self, value: int):
        self.max_epochs = value

    def add_row_pressed(self):
        self.insert_row(self.ui.custom_data_table)

    def remove_row_pressed(self):
        rowPosition = self.ui.custom_data_table.rowCount()
        self.ui.custom_data_table.removeRow(rowPosition)

    def save_button_pressed(self):
        values = self.read_table_values(self.ui.custom_data_table)
        self.X = pd.DataFrame(values[:, :self.inputs])
        self.y = pd.DataFrame(values[:, self.inputs:])
        self.classes = [f'y{i}' for i in range(1, self.outputs + 1)]

    def train_button_pressed(self):
        self.network = NeuralNetwork(
                inputs=self.inputs,
                outputs=self.outputs,
                hidden=self.hidden_layers,
                learning_rate=self.learning_rate,
                epochs=self.max_epochs,
                loss_name=self.loss,
                activation_name=self.activation
        )
        _, self.X_train, self.y_train, \
            self.X_valid, self.y_valid, \
            self.X_test, self.y_test = train_test_valid_split(self.X, self.y)

        self.network.train(X=self.X_train,
                           y=self.y_train,
                           X_valid=self.X_valid,
                           y_valid=self.y_valid,
                           classes=self.classes,
                           init_type=self.init_type)

        self.network.write_loss_graph_into_file("loss.png")
        picture = QPixmap("loss.png")
        self.ui.training_graph_label.setPixmap(picture)

    def test_button_pressed(self):
        predicted = self.network.predict(self.X_test)
        predicted_index = np.argmax(predicted, axis=1)
        true_index = np.argmax(self.y_test, axis=1)

        accuracy_score = np.sum(predicted_index == true_index) \
            / len(self.y_test) \
            * 100
        self.ui.accuracy_value_label.setText(f"{accuracy_score:.2f}%")

        values = np.hstack([np.array(self.classes)[[predicted_index]].reshape(-1, 1),
                            np.array(self.classes)[[true_index]].reshape(-1, 1),
                            self.X_test.to_numpy()])
        self.write_table_values(self.ui.test_table, values)

    def predict_button_pressed(self):
        pass

    def tab_bar_clicked(self, index: int):
        if ((index not in [1, 3, 4])
            or self.ui.custom_data_table.columnCount()
                == self.inputs + self.outputs):
            return

        if index == 1:
            self.ui.custom_data_table.setColumnCount(0)
            for i in range(self.inputs + self.outputs):
                self.insert_col(self.ui.custom_data_table, i)

        elif index == 3:
            self.ui.test_table.setColumnCount(0)
            for i in range(self.inputs + 2):
                self.insert_col(self.ui.test_table, i)

            labels = ['predicted', 'true'] + [f'x{i}' for i in range(1, self.inputs+1)]
            self.ui.test_table.setHorizontalHeaderLabels(labels)

            self.ui.test_table.setRowCount(0)
            for i in range(self.X_test.shape[0]):
                self.insert_row(self.ui.test_table)

        elif index == 4:
            self.ui.predictions_table.setColumnCount(0)
            for i in range(self.inputs + self.outputs + 1):
                self.insert_col(self.ui.predictions_table, i)

    # Helper functions

    def handle_numbers(self):
        self.set_custom_data_field_availability(False)
        digits = load_digits()
        self.classes = list(range(10))
        self.X = pd.DataFrame(digits.data)
        self.y = pd.DataFrame(digits.target)
        self.inputs = len(digits.data[0])
        self.outputs = 10
        self.ui.layer_1_spin_box.setValue(self.calculate_neurons_number())

    def handle_letters(self):
        self.set_custom_data_field_availability(False)
        self.classes = np.array(list("abcdefghijklmnopqrstuvwxyz"))
        self.X, self.y = generate_letters(20, 0.1, self.classes)
        self.inputs = self.X.shape[1]
        self.outputs = len(self.classes)
        self.ui.layer_1_spin_box.setValue(self.calculate_neurons_number())

    def handle_custom(self):
        self.set_custom_data_field_availability(True)

    def set_custom_data_field_availability(self, value: bool):
        self.set_widget_availability(self.ui.inputs_spin_box, value)
        self.set_widget_availability(self.ui.outputs_spin_box, value)
        self.set_widget_availability(self.ui.custom_data_tab, value)

    def set_widget_availability(self, widget: QWidget, value: bool):
        widget.setEnabled(value)

    def change_hidden_layers(self, value: int, index: int):
        self.hidden_layers[index] = value
        self.layers[index + 1] = value

    def get_number_of_neurons(self):
        return np.sum(self.layers)

    def get_number_of_weights(self):
        return np.sum(np.multiply(
                np.array(self.layers[:-1]) + 1,
                np.array(self.layers[1:])
        ))

    def calculate_neurons_number(self):
        samples = len(self.X)
        lower = self.outputs * len(self.X) / (1 + np.log2(samples))
        upper = self.outputs \
            * (samples/self.outputs + 1) \
            * (self.inputs + self.outputs + 1) \
            + self.outputs

        io_coef = (self.inputs/self.outputs) / (self.inputs/self.outputs + 1)
        sample_coef = np.log(samples) / (np.log(samples) + 1)
        weights_number = 0.999 * lower + 0.001 * upper \
            if io_coef < sample_coef \
            else 0.8 * lower + 0.2 * upper

        return weights_number / (self.inputs + self.outputs)

    def insert_row(self, table: QTableWidget):
        row_ind = table.rowCount()
        table.setRowCount(table.rowCount() + 1)

        for i in range(table.columnCount()):
            self.set_cell_value(table, row_ind, i, None)

    def insert_col(self, table: QTableWidget, col_ind: int):
        col_ind = table.columnCount()
        table.setColumnCount(table.columnCount() + 1)

        for i in range(table.rowCount()):
            self.set_cell_value(table, i, col_ind, None)

    def set_cell_value(self, table: QTableWidget,
                       row_ind: int, col_ind: int, value):
        item = QTableWidgetItem(value)
        table.setItem(row_ind, col_ind, item)

    def read_table_values(self, table: QTableWidget):
        values = np.empty((table.rowCount(), table.columnCount()))
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                values[i, j] = table.item(i, j).text()

        return values

    def write_table_values(self, table: QTableWidget, values: np.array):
        print(self.X_test, self.y_test)
        print(values)
        print(table.rowCount(), table.columnCount())
        if values.shape != (table.rowCount(), table.columnCount()):
            print(f"Dimenstions mismatch! Table dims: ({table.rowCount()}, {
                table.columnCount()}); Values dims {values.shape}")
            return

        for i in range(table.rowCount()):
            for j in range(table.columnCount()):
                table.item(i, j).setText(str(values[i, j]))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    sys.exit(app.exec())
