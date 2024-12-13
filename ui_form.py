# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractSpinBox, QApplication, QComboBox, QDoubleSpinBox,
    QFormLayout, QGridLayout, QHeaderView, QLabel,
    QPushButton, QSizePolicy, QSpinBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(864, 567)
        Widget.setMinimumSize(QSize(864, 567))
        Widget.setMaximumSize(QSize(864, 567))
        self.tab_widget = QTabWidget(Widget)
        self.tab_widget.setObjectName(u"tab_widget")
        self.tab_widget.setGeometry(QRect(-4, -1, 871, 571))
        self.tab_widget.setMinimumSize(QSize(871, 571))
        self.tab_widget.setMaximumSize(QSize(871, 571))
        self.parameters_tab = QWidget()
        self.parameters_tab.setObjectName(u"parameters_tab")
        self.formLayoutWidget = QWidget(self.parameters_tab)
        self.formLayoutWidget.setObjectName(u"formLayoutWidget")
        self.formLayoutWidget.setGeometry(QRect(9, 9, 291, 521))
        self.parameters_form_layout = QFormLayout(self.formLayoutWidget)
        self.parameters_form_layout.setObjectName(u"parameters_form_layout")
        self.parameters_form_layout.setContentsMargins(0, 0, 0, 0)
        self.data_set_label = QLabel(self.formLayoutWidget)
        self.data_set_label.setObjectName(u"data_set_label")

        self.parameters_form_layout.setWidget(0, QFormLayout.LabelRole, self.data_set_label)

        self.data_set_combo_box = QComboBox(self.formLayoutWidget)
        self.data_set_combo_box.addItem("")
        self.data_set_combo_box.addItem("")
        self.data_set_combo_box.addItem("")
        self.data_set_combo_box.setObjectName(u"data_set_combo_box")

        self.parameters_form_layout.setWidget(0, QFormLayout.FieldRole, self.data_set_combo_box)

        self.inputs_label = QLabel(self.formLayoutWidget)
        self.inputs_label.setObjectName(u"inputs_label")

        self.parameters_form_layout.setWidget(1, QFormLayout.LabelRole, self.inputs_label)

        self.inputs_spin_box = QSpinBox(self.formLayoutWidget)
        self.inputs_spin_box.setObjectName(u"inputs_spin_box")
        self.inputs_spin_box.setMinimum(1)
        self.inputs_spin_box.setMaximum(10000000)

        self.parameters_form_layout.setWidget(1, QFormLayout.FieldRole, self.inputs_spin_box)

        self.outputs_label = QLabel(self.formLayoutWidget)
        self.outputs_label.setObjectName(u"outputs_label")

        self.parameters_form_layout.setWidget(2, QFormLayout.LabelRole, self.outputs_label)

        self.outputs_spin_box = QSpinBox(self.formLayoutWidget)
        self.outputs_spin_box.setObjectName(u"outputs_spin_box")
        self.outputs_spin_box.setMinimum(1)
        self.outputs_spin_box.setMaximum(10000)

        self.parameters_form_layout.setWidget(2, QFormLayout.FieldRole, self.outputs_spin_box)

        self.hidden_layers_label = QLabel(self.formLayoutWidget)
        self.hidden_layers_label.setObjectName(u"hidden_layers_label")

        self.parameters_form_layout.setWidget(3, QFormLayout.LabelRole, self.hidden_layers_label)

        self.hidden_layers_spin_box = QSpinBox(self.formLayoutWidget)
        self.hidden_layers_spin_box.setObjectName(u"hidden_layers_spin_box")
        self.hidden_layers_spin_box.setMaximum(8)
        self.hidden_layers_spin_box.setValue(1)

        self.parameters_form_layout.setWidget(3, QFormLayout.FieldRole, self.hidden_layers_spin_box)

        self.hidden_layers_neurons_grid_layout = QGridLayout()
        self.hidden_layers_neurons_grid_layout.setObjectName(u"hidden_layers_neurons_grid_layout")
        self.layer_5_spin_box = QSpinBox(self.formLayoutWidget)
        self.layer_5_spin_box.setObjectName(u"layer_5_spin_box")
        self.layer_5_spin_box.setMaximum(999)

        self.hidden_layers_neurons_grid_layout.addWidget(self.layer_5_spin_box, 2, 0, 1, 1)

        self.layer_3_spin_box = QSpinBox(self.formLayoutWidget)
        self.layer_3_spin_box.setObjectName(u"layer_3_spin_box")
        self.layer_3_spin_box.setMaximum(999)

        self.hidden_layers_neurons_grid_layout.addWidget(self.layer_3_spin_box, 1, 0, 1, 1)

        self.layer_7_spin_box = QSpinBox(self.formLayoutWidget)
        self.layer_7_spin_box.setObjectName(u"layer_7_spin_box")
        self.layer_7_spin_box.setMaximum(999)

        self.hidden_layers_neurons_grid_layout.addWidget(self.layer_7_spin_box, 3, 0, 1, 1)

        self.layer_8_spin_box = QSpinBox(self.formLayoutWidget)
        self.layer_8_spin_box.setObjectName(u"layer_8_spin_box")
        self.layer_8_spin_box.setMaximum(999)

        self.hidden_layers_neurons_grid_layout.addWidget(self.layer_8_spin_box, 3, 1, 1, 1)

        self.layer_4_spin_box = QSpinBox(self.formLayoutWidget)
        self.layer_4_spin_box.setObjectName(u"layer_4_spin_box")
        self.layer_4_spin_box.setMaximum(999)

        self.hidden_layers_neurons_grid_layout.addWidget(self.layer_4_spin_box, 1, 1, 1, 1)

        self.layer_6_spin_box = QSpinBox(self.formLayoutWidget)
        self.layer_6_spin_box.setObjectName(u"layer_6_spin_box")
        self.layer_6_spin_box.setMaximum(999)

        self.hidden_layers_neurons_grid_layout.addWidget(self.layer_6_spin_box, 2, 1, 1, 1)

        self.layer_1_spin_box = QSpinBox(self.formLayoutWidget)
        self.layer_1_spin_box.setObjectName(u"layer_1_spin_box")
        self.layer_1_spin_box.setMaximum(999)

        self.hidden_layers_neurons_grid_layout.addWidget(self.layer_1_spin_box, 0, 0, 1, 1)

        self.layer_2_spin_box = QSpinBox(self.formLayoutWidget)
        self.layer_2_spin_box.setObjectName(u"layer_2_spin_box")
        self.layer_2_spin_box.setMaximum(999)

        self.hidden_layers_neurons_grid_layout.addWidget(self.layer_2_spin_box, 0, 1, 1, 1)


        self.parameters_form_layout.setLayout(4, QFormLayout.FieldRole, self.hidden_layers_neurons_grid_layout)

        self.hidden_layers_neurons_label = QLabel(self.formLayoutWidget)
        self.hidden_layers_neurons_label.setObjectName(u"hidden_layers_neurons_label")
        self.hidden_layers_neurons_label.setAlignment(Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignTop)
        self.hidden_layers_neurons_label.setWordWrap(True)

        self.parameters_form_layout.setWidget(4, QFormLayout.LabelRole, self.hidden_layers_neurons_label)

        self.loss_function_label = QLabel(self.formLayoutWidget)
        self.loss_function_label.setObjectName(u"loss_function_label")

        self.parameters_form_layout.setWidget(6, QFormLayout.LabelRole, self.loss_function_label)

        self.activation_function_label = QLabel(self.formLayoutWidget)
        self.activation_function_label.setObjectName(u"activation_function_label")

        self.parameters_form_layout.setWidget(7, QFormLayout.LabelRole, self.activation_function_label)

        self.loss_function_combo_box = QComboBox(self.formLayoutWidget)
        self.loss_function_combo_box.addItem("")
        self.loss_function_combo_box.addItem("")
        self.loss_function_combo_box.setObjectName(u"loss_function_combo_box")

        self.parameters_form_layout.setWidget(6, QFormLayout.FieldRole, self.loss_function_combo_box)

        self.activation_function_combo_box = QComboBox(self.formLayoutWidget)
        self.activation_function_combo_box.addItem("")
        self.activation_function_combo_box.addItem("")
        self.activation_function_combo_box.setObjectName(u"activation_function_combo_box")

        self.parameters_form_layout.setWidget(7, QFormLayout.FieldRole, self.activation_function_combo_box)

        self.overall_neurons_label = QLabel(self.formLayoutWidget)
        self.overall_neurons_label.setObjectName(u"overall_neurons_label")

        self.parameters_form_layout.setWidget(8, QFormLayout.LabelRole, self.overall_neurons_label)

        self.overall_neurons_number_label = QLabel(self.formLayoutWidget)
        self.overall_neurons_number_label.setObjectName(u"overall_neurons_number_label")

        self.parameters_form_layout.setWidget(8, QFormLayout.FieldRole, self.overall_neurons_number_label)

        self.overall_weights_label = QLabel(self.formLayoutWidget)
        self.overall_weights_label.setObjectName(u"overall_weights_label")
        self.overall_weights_label.setWordWrap(True)

        self.parameters_form_layout.setWidget(9, QFormLayout.LabelRole, self.overall_weights_label)

        self.overall_weigts_number_label = QLabel(self.formLayoutWidget)
        self.overall_weigts_number_label.setObjectName(u"overall_weigts_number_label")

        self.parameters_form_layout.setWidget(9, QFormLayout.FieldRole, self.overall_weigts_number_label)

        self.learning_rate_label = QLabel(self.formLayoutWidget)
        self.learning_rate_label.setObjectName(u"learning_rate_label")

        self.parameters_form_layout.setWidget(10, QFormLayout.LabelRole, self.learning_rate_label)

        self.epochs_number_label = QLabel(self.formLayoutWidget)
        self.epochs_number_label.setObjectName(u"epochs_number_label")

        self.parameters_form_layout.setWidget(11, QFormLayout.LabelRole, self.epochs_number_label)

        self.learning_rate_double_spin_box = QDoubleSpinBox(self.formLayoutWidget)
        self.learning_rate_double_spin_box.setObjectName(u"learning_rate_double_spin_box")
        self.learning_rate_double_spin_box.setDecimals(6)
        self.learning_rate_double_spin_box.setMinimum(0.000001000000000)
        self.learning_rate_double_spin_box.setSingleStep(0.000001000000000)
        self.learning_rate_double_spin_box.setStepType(QAbstractSpinBox.StepType.DefaultStepType)
        self.learning_rate_double_spin_box.setValue(0.010000000000000)

        self.parameters_form_layout.setWidget(10, QFormLayout.FieldRole, self.learning_rate_double_spin_box)

        self.epochs_number_spin_box = QSpinBox(self.formLayoutWidget)
        self.epochs_number_spin_box.setObjectName(u"epochs_number_spin_box")
        self.epochs_number_spin_box.setMinimum(1)
        self.epochs_number_spin_box.setMaximum(1000000000)

        self.parameters_form_layout.setWidget(11, QFormLayout.FieldRole, self.epochs_number_spin_box)

        self.weights_init_type_lable = QLabel(self.formLayoutWidget)
        self.weights_init_type_lable.setObjectName(u"weights_init_type_lable")
        self.weights_init_type_lable.setWordWrap(True)

        self.parameters_form_layout.setWidget(5, QFormLayout.LabelRole, self.weights_init_type_lable)

        self.weights_init_type_combo_box = QComboBox(self.formLayoutWidget)
        self.weights_init_type_combo_box.addItem("")
        self.weights_init_type_combo_box.addItem("")
        self.weights_init_type_combo_box.addItem("")
        self.weights_init_type_combo_box.addItem("")
        self.weights_init_type_combo_box.setObjectName(u"weights_init_type_combo_box")

        self.parameters_form_layout.setWidget(5, QFormLayout.FieldRole, self.weights_init_type_combo_box)

        self.layoutWidget = QWidget(self.parameters_tab)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(305, 10, 551, 521))
        self.network_overview_vertical_layout = QVBoxLayout(self.layoutWidget)
        self.network_overview_vertical_layout.setObjectName(u"network_overview_vertical_layout")
        self.network_overview_vertical_layout.setContentsMargins(0, 0, 0, 0)
        self.network_structure_lable = QLabel(self.layoutWidget)
        self.network_structure_lable.setObjectName(u"network_structure_lable")

        self.network_overview_vertical_layout.addWidget(self.network_structure_lable)

        self.tab_widget.addTab(self.parameters_tab, "")
        self.custom_data_tab = QWidget()
        self.custom_data_tab.setObjectName(u"custom_data_tab")
        self.add_row_button = QPushButton(self.custom_data_tab)
        self.add_row_button.setObjectName(u"add_row_button")
        self.add_row_button.setGeometry(QRect(660, 10, 87, 26))
        self.custom_data_table = QTableWidget(self.custom_data_tab)
        self.custom_data_table.setObjectName(u"custom_data_table")
        self.custom_data_table.setGeometry(QRect(10, 51, 851, 481))
        self.remove_row_button = QPushButton(self.custom_data_tab)
        self.remove_row_button.setObjectName(u"remove_row_button")
        self.remove_row_button.setGeometry(QRect(750, 10, 111, 26))
        self.custom_data_division_ratio_label = QLabel(self.custom_data_tab)
        self.custom_data_division_ratio_label.setObjectName(u"custom_data_division_ratio_label")
        self.custom_data_division_ratio_label.setGeometry(QRect(20, 10, 591, 21))
        self.custom_data_division_ratio_label.setAlignment(Qt.AlignmentFlag.AlignBottom|Qt.AlignmentFlag.AlignLeading|Qt.AlignmentFlag.AlignLeft)
        self.custom_data_division_ratio_label.setWordWrap(True)
        self.save_data_button = QPushButton(self.custom_data_tab)
        self.save_data_button.setObjectName(u"save_data_button")
        self.save_data_button.setGeometry(QRect(610, 10, 41, 26))
        self.tab_widget.addTab(self.custom_data_tab, "")
        self.trainint_tab = QWidget()
        self.trainint_tab.setObjectName(u"trainint_tab")
        self.train_button = QPushButton(self.trainint_tab)
        self.train_button.setObjectName(u"train_button")
        self.train_button.setGeometry(QRect(10, 10, 87, 26))
        self.training_graph_label = QLabel(self.trainint_tab)
        self.training_graph_label.setObjectName(u"training_graph_label")
        self.training_graph_label.setGeometry(QRect(15, 57, 851, 481))
        self.tab_widget.addTab(self.trainint_tab, "")
        self.testing_tab = QWidget()
        self.testing_tab.setObjectName(u"testing_tab")
        self.test_button = QPushButton(self.testing_tab)
        self.test_button.setObjectName(u"test_button")
        self.test_button.setGeometry(QRect(10, 10, 87, 26))
        self.test_table = QTableWidget(self.testing_tab)
        self.test_table.setObjectName(u"test_table")
        self.test_table.setGeometry(QRect(10, 50, 851, 481))
        self.accuracy_label = QLabel(self.testing_tab)
        self.accuracy_label.setObjectName(u"accuracy_label")
        self.accuracy_label.setGeometry(QRect(620, 10, 66, 18))
        self.accuracy_value_label = QLabel(self.testing_tab)
        self.accuracy_value_label.setObjectName(u"accuracy_value_label")
        self.accuracy_value_label.setGeometry(QRect(710, 10, 111, 18))
        self.tab_widget.addTab(self.testing_tab, "")
        self.predicting_tab = QWidget()
        self.predicting_tab.setObjectName(u"predicting_tab")
        self.predict_button = QPushButton(self.predicting_tab)
        self.predict_button.setObjectName(u"predict_button")
        self.predict_button.setGeometry(QRect(10, 10, 87, 26))
        self.predictions_table = QTableWidget(self.predicting_tab)
        self.predictions_table.setObjectName(u"predictions_table")
        self.predictions_table.setGeometry(QRect(10, 50, 851, 481))
        self.tab_widget.addTab(self.predicting_tab, "")
#if QT_CONFIG(shortcut)
        self.data_set_label.setBuddy(self.data_set_combo_box)
        self.inputs_label.setBuddy(self.inputs_spin_box)
        self.outputs_label.setBuddy(self.outputs_spin_box)
        self.hidden_layers_label.setBuddy(self.hidden_layers_spin_box)
        self.loss_function_label.setBuddy(self.loss_function_combo_box)
        self.activation_function_label.setBuddy(self.activation_function_combo_box)
        self.learning_rate_label.setBuddy(self.learning_rate_double_spin_box)
        self.epochs_number_label.setBuddy(self.epochs_number_spin_box)
        self.custom_data_division_ratio_label.setBuddy(self.add_row_button)
#endif // QT_CONFIG(shortcut)

        self.retranslateUi(Widget)

        self.tab_widget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.data_set_label.setText(QCoreApplication.translate("Widget", u"Data set", None))
        self.data_set_combo_box.setItemText(0, QCoreApplication.translate("Widget", u"Numbers", None))
        self.data_set_combo_box.setItemText(1, QCoreApplication.translate("Widget", u"English letters", None))
        self.data_set_combo_box.setItemText(2, QCoreApplication.translate("Widget", u"Custom", None))

        self.inputs_label.setText(QCoreApplication.translate("Widget", u"Number of inputs", None))
        self.outputs_label.setText(QCoreApplication.translate("Widget", u"Number of outputs", None))
        self.hidden_layers_label.setText(QCoreApplication.translate("Widget", u"Hidden layers", None))
        self.hidden_layers_neurons_label.setText(QCoreApplication.translate("Widget", u"Number of neurons in hidden layers", None))
        self.loss_function_label.setText(QCoreApplication.translate("Widget", u"Loss function", None))
        self.activation_function_label.setText(QCoreApplication.translate("Widget", u"Activation function", None))
        self.loss_function_combo_box.setItemText(0, QCoreApplication.translate("Widget", u"mse", None))
        self.loss_function_combo_box.setItemText(1, QCoreApplication.translate("Widget", u"mae", None))

        self.activation_function_combo_box.setItemText(0, QCoreApplication.translate("Widget", u"sigmoid", None))
        self.activation_function_combo_box.setItemText(1, QCoreApplication.translate("Widget", u"logarithmic", None))

        self.overall_neurons_label.setText(QCoreApplication.translate("Widget", u"Number of neurons", None))
        self.overall_neurons_number_label.setText("")
        self.overall_weights_label.setText(QCoreApplication.translate("Widget", u"Number of weights (including biases)", None))
        self.overall_weigts_number_label.setText("")
        self.learning_rate_label.setText(QCoreApplication.translate("Widget", u"Learning Rate", None))
        self.epochs_number_label.setText(QCoreApplication.translate("Widget", u"Number of epochs", None))
        self.weights_init_type_lable.setText(QCoreApplication.translate("Widget", u"Weights init type", None))
        self.weights_init_type_combo_box.setItemText(0, QCoreApplication.translate("Widget", u"xavier", None))
        self.weights_init_type_combo_box.setItemText(1, QCoreApplication.translate("Widget", u"genetic", None))
        self.weights_init_type_combo_box.setItemText(2, QCoreApplication.translate("Widget", u"uniform", None))
        self.weights_init_type_combo_box.setItemText(3, QCoreApplication.translate("Widget", u"constant", None))

        self.network_structure_lable.setText(QCoreApplication.translate("Widget", u"Network structure", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.parameters_tab), QCoreApplication.translate("Widget", u"Parameters", None))
        self.add_row_button.setText(QCoreApplication.translate("Widget", u"Add row", None))
        self.remove_row_button.setText(QCoreApplication.translate("Widget", u"Remove row", None))
        self.custom_data_division_ratio_label.setText(QCoreApplication.translate("Widget", u"Ratio of input data division is the following: train - 60%, validation - 30%, testing - 10%", None))
        self.save_data_button.setText(QCoreApplication.translate("Widget", u"Save", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.custom_data_tab), QCoreApplication.translate("Widget", u"Custom data", None))
        self.train_button.setText(QCoreApplication.translate("Widget", u"Train", None))
        self.training_graph_label.setText("")
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.trainint_tab), QCoreApplication.translate("Widget", u"Training", None))
        self.test_button.setText(QCoreApplication.translate("Widget", u"Test", None))
        self.accuracy_label.setText(QCoreApplication.translate("Widget", u"Accuracy", None))
        self.accuracy_value_label.setText("")
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.testing_tab), QCoreApplication.translate("Widget", u"Testing", None))
        self.predict_button.setText(QCoreApplication.translate("Widget", u"Predict", None))
        self.tab_widget.setTabText(self.tab_widget.indexOf(self.predicting_tab), QCoreApplication.translate("Widget", u"Predicting", None))
    # retranslateUi

