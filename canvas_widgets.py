import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class SignalCanvas(FigureCanvas):
    """График для отображения кардиограммы и отфильтрованных данных (метод Хампеля)."""
    def __init__(self, parent=None):
        self.fig, self.ax = Figure(figsize=(6, 4), dpi=100), None
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_ecg_with_outliers(self, original, filtered, outliers):
        """Отображает оригинальный и отфильтрованный сигнал с выбросами"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # Оригинальный сигнал
        ax.plot(original, 'b-', alpha=0.5, label='Original ECG')

        # Отфильтрованный сигнал
        ax.plot(filtered, 'g-', linewidth=1.5, label='Filtered ECG')

        # Выбросы
        ax.scatter(
            outliers,
            original[outliers],
            color='red',
            marker='x',
            s=50,
            label=f'Outliers ({len(outliers)})'
        )

        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)
        self.draw()

    def plot_df(self, df):
        """Отображает кардиограмму из DataFrame."""
        self.ax = self.fig.add_subplot(111)
        self.ax.clear()

        # Проверяем наличие стандартных названий столбцов
        if 'T' in df.columns:
            # Если есть столбец 'T' - используем его
            self.ax.plot(df.index, df['T'], label="Original ECG", color="blue")
        elif len(df.columns) >= 1:
            # Если нет специальных названий - используем первый столбец
            col_name = df.columns[0]
            self.ax.plot(df.index, df[col_name], label=f"ECG ({col_name})", color="blue")

            # Если есть дополнительные столбцы - рисуем их тоже
            if len(df.columns) > 1:
                colors = ['green', 'red', 'purple', 'orange']
                for i, col in enumerate(df.columns[1:]):
                    color = colors[i % len(colors)]
                    self.ax.plot(df.index, df[col], label=f"ECG ({col})",
                                 color=color, alpha=0.7, linewidth=0.8)

        self.ax.set_xlabel("Samples")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend()
        self.ax.grid(True)
        self.draw()

    def plot_cleaned(self, df_clean, outliers):
        """Отображает кардиограмму после фильтрации Хампеля, выделяя выбросы."""
        self.ax.clear()
        self.ax.plot(df_clean.index, df_clean['T'], label="Filtered ECG", color="green")
        self.ax.scatter(df_clean.index[outliers], df_clean['T'][outliers],
                        color="red", label="Outliers", marker="x")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend()
        self.draw()

class OutlierInfoCanvas(FigureCanvas):
    """График для отображения информации о выбросах."""
    def __init__(self, parent=None):
        self.fig, self.ax = Figure(figsize=(6, 4), dpi=100), None
        super().__init__(self.fig)
        self.setParent(parent)

    def plot_outlier_info(self, outliers):
        """Гистограмма распределения выбросов."""
        self.ax = self.fig.add_subplot(111)
        self.ax.clear()
        self.ax.hist(outliers, bins=20, color="red", alpha=0.7)
        self.ax.set_title("Outlier Distribution (Hampel Filter)")
        self.ax.set_xlabel("Deviation")
        self.ax.set_ylabel("Frequency")
        self.draw()