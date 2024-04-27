from PyQt5.QtWidgets import QGraphicsScene, QGraphicsView
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import numpy as np
import pyqtgraph as pg

class YourClass:
    def __init__(self):
        # 在介面初始化函式中初始化柱狀圖顯示
        self.init_histogram_graph()

    def init_histogram_graph(self):
        # 創建 PlotWidget
        self.histogram_plt = pg.PlotWidget()
        # 設置大小
        self.histogram_plt.resize(self.ui.x_figure_view.width(), self.ui.x_figure_view.height())
        title = "Stride Length (Average: 0.00m)"
        font = QFont()
        font.setPixelSize(15)
        # 設置 y 軸標籤
        self.histogram_plt.setLabel('left', 'Stride', font=font)
        # 設置 x 和 y 軸範圍
        self.histogram_plt.setXRange(0, 3)
        self.histogram_plt.setYRange(0, 6)
        # 設置 y 軸刻度
        y_ticks = [(i + 1, str(i + 1)) for i in range(7)]
        self.histogram_plt.getPlotItem().getAxis('left').setTicks([y_ticks])
        # 設置 x 軸刻度
        x_ticks = [(i, str(i)) for i in np.arange(0, 3.5, 0.5)]
        self.histogram_plt.getPlotItem().getAxis('bottom').setTicks([x_ticks])
        # 設置 x 軸和 y 軸標籤
        self.histogram_plt.setLabel('bottom', 'm', font=font)
        self.histogram_plt.setWindowTitle(title)
        self.histogram_plt.setTitle(title)
        # 將 PlotWidget 添加到 QGraphicsScene 中
        self.histogram_scene.addWidget(self.histogram_plt)
        # 將 QGraphicsScene 設置為 QGraphicsView 的場景
        self.ui.x_figure_view.setScene(self.histogram_scene)
        # 自動調整視圖以適應場景大小
        self.ui.x_figure_view.fitInView(self.histogram_scene.sceneRect(), Qt.KeepAspectRatio)
