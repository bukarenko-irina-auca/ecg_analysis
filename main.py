import os
import sys
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QTabWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QGroupBox, QFormLayout, QPushButton, QLineEdit, QTextEdit,
    QMessageBox, QApplication, QMainWindow
)
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavBar
from canvas_widgets import SignalCanvas, OutlierInfoCanvas
import chardet

def detect_encoding(file_path):
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)
    return chardet.detect(raw_data)['encoding']

class Tabs(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.file_path = ''
        self.file_name = ''
        self.df = None
        self.clean_df = None
        self.outlier_indexes = None
        self.delta_values = None

        self.tab = [QWidget(), QWidget()]
        self.addTab(self.tab[0], 'Time series')
        self.addTab(self.tab[1], 'Info')

        self.tab0_ui()
        self.tab1_ui()

        self.setStyleSheet('QTabBar { color:black; font-size: 9pt; font-weight: bold; }')

    def tab0_ui(self):
        self.tab[0].canvas = SignalCanvas(self.tab[0])
        navtoolbar = NavBar(self.tab[0].canvas, self.tab[0])

        self.tab[0].btn_load_df = QPushButton('Load ECG file')
        self.tab[0].btn_load_df.setStyleSheet("background-color: skyblue")
        self.tab[0].btn_load_df.clicked.connect(self.on_load_df)

        self.tab[0].edt_std_limit = QLineEdit('3')
        self.tab[0].edt_window_size = QLineEdit('5')
        self.tab[0].edt_max_delta = QLineEdit('4')

        self.tab[0].btn_filter = QPushButton('Filter Outliers')
        self.tab[0].btn_filter.setStyleSheet("background-color: skyblue")
        self.tab[0].btn_filter.clicked.connect(self.on_filter_df)

        self.tab[0].edt_outlier_indexes = QTextEdit('# Outliers')
        self.tab[0].btn_save_cleaned_df = QPushButton('Save cleaned ECG')
        self.tab[0].btn_save_cleaned_df.setStyleSheet("background-color: skyblue")
        self.tab[0].btn_save_cleaned_df.clicked.connect(self.on_save_cleaned_df)

        vbox_left = QVBoxLayout()
        vbox_left.addWidget(self.tab[0].canvas)
        vbox_left.addWidget(navtoolbar)

        vbox_right = QVBoxLayout()
        vbox_right.addWidget(self.tab[0].btn_load_df)

        edt_form = QFormLayout()
        edt_form.addRow("Std:", self.tab[0].edt_std_limit)
        edt_form.addRow("Window size:", self.tab[0].edt_window_size)
        edt_form.addRow("Outlier max value:", self.tab[0].edt_max_delta)

        filtration_params = QWidget()
        filtration_params.setLayout(edt_form)

        vbox_right.addWidget(filtration_params)
        vbox_right.addWidget(self.tab[0].btn_filter)
        vbox_right.addWidget(self.tab[0].edt_outlier_indexes)
        vbox_right.addWidget(self.tab[0].btn_save_cleaned_df)
        vbox_right.addStretch()

        groupbox = QGroupBox("Signal Filtration")
        groupbox.setLayout(vbox_right)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox_left, 3)
        hbox.addWidget(groupbox, 1)

        self.tab[0].setLayout(hbox)

    def tab1_ui(self):
        self.tab[1].canvas = OutlierInfoCanvas(self.tab[1])
        navtoolbar = NavBar(self.tab[1].canvas, self.tab[1])

        self.tab[1].btn_plot_outlier_info = QPushButton('Plot Outlier Info')
        self.tab[1].btn_plot_outlier_info.clicked.connect(self.on_plot_outlier_info)

        vbox_left = QVBoxLayout()
        vbox_left.addWidget(self.tab[1].canvas)
        vbox_left.addWidget(navtoolbar)

        vbox_right = QVBoxLayout()
        vbox_right.addWidget(self.tab[1].btn_plot_outlier_info)
        vbox_right.addStretch()

        groupbox1 = QGroupBox("Outliers")
        groupbox1.setLayout(vbox_right)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox_left, 3)
        hbox.addWidget(groupbox1, 1)

        self.tab[1].setLayout(hbox)

    def on_load_df(self):
        file_choices = "ECG Files (*.txt);;All Files (*)"
        path, _ = QFileDialog.getOpenFileName(self, 'Open ECG file', '', file_choices)

        if path:
            try:
                encoding = detect_encoding(path)
                data = pd.read_csv(path, sep=r'\s+', engine='python',
                                   dtype=np.float32, encoding=encoding, header=None)

                # Создаем DataFrame с автоматическими названиями столбцов
                self.df = pd.DataFrame(data.values, columns=[f"ECG{i + 1}" for i in range(data.shape[1])])

                # Удаляем строки с NaN (если есть)
                self.df.dropna(inplace=True)

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to load file: {str(e)}")
                return

        self.file_path, self.file_name = os.path.split(path)
        self.tab[0].canvas.plot_df(self.df)
        self.window().statusBar().showMessage(
            f"Loaded: {path} | Columns: {len(self.df.columns)} | Samples: {len(self.df)}"
        )

    def on_filter_df(self):
        """Применяет фильтр Хампеля для удаления выбросов и отображает результат"""
        if self.df is None:
            QMessageBox.warning(self, "Error", "No file loaded!")
            return

        try:
            # Параметры фильтра (можно вынести в настройки)
            window_size = 21  # Размер окна для анализа
            threshold = 3.0  # Пороговое значение для выбросов

            # Применяем фильтр Хампеля к первому каналу ЭКГ
            signal = self.df.iloc[:, 0].values
            outliers = self.hampel_filter(signal, window_size, threshold)

            # Создаем копию данных с заменой выбросов
            filtered_data = self.df.copy()
            for i in outliers:
                # Заменяем выброс медианным значением окна
                window_start = max(0, i - window_size // 2)
                window_end = min(len(signal), i + window_size // 2 + 1)
                filtered_data.iloc[i, 0] = np.median(signal[window_start:window_end])

            # Сохраняем отфильтрованные данные
            self.filtered_data = filtered_data
            self.outliers = outliers

            # Визуализируем результат
            self.tab[0].canvas.plot_ecg_with_outliers(
                original=self.df.iloc[:, 0],
                filtered=filtered_data.iloc[:, 0],
                outliers=outliers
            )

            # Показываем статистику
            QMessageBox.information(
                self,
                "Filter Applied",
                f"Found {len(outliers)} outliers ({len(outliers) / len(signal) * 100:.1f}%)\n"
                f"Window size: {window_size}\nThreshold: {threshold}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Filter failed: {str(e)}")

    def hampel_filter(self, data, window_size=21, threshold=3.0):
        """Реализация фильтра Хампеля для обнаружения выбросов"""
        half_window = window_size // 2
        outliers = []

        for i in range(len(data)):
            window_start = max(0, i - half_window)
            window_end = min(len(data), i + half_window + 1)
            window = data[window_start:window_end]

            median = np.median(window)
            mad = 1.4826 * np.median(np.abs(window - median))  # MAD с масштабированием

            if mad == 0:
                continue

            if np.abs(data[i] - median) > threshold * mad:
                outliers.append(i)

        return outliers

    def on_plot_outlier_info(self):
        if self.clean_df is None:
            QMessageBox.warning(self, "Error", "No filtered data available!")
            return
        self.tab[1].canvas.plot_outlier_info()

    def on_save_cleaned_df(self):
        if self.clean_df is None:
            QMessageBox.warning(self, "Error", "No cleaned data to save!")
            return
        fn = os.path.join(self.file_path, f"{self.file_name.split('.')[0]}_filtered.ecg")
        self.clean_df.to_csv(fn, index=False, sep=',')
        QMessageBox.about(self, "Save", f'Cleaned ECG saved as:\n {fn}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = QMainWindow()
    main_window.setWindowTitle("ECG Analysis")
    main_window.resize(1000, 600)

    tabs = Tabs()
    main_window.setCentralWidget(tabs)
    main_window.show()

    sys.exit(app.exec_())
