import sys
import pyqtgraph as pg
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication

def main():
    # 建立應用程式
    app = QApplication(sys.argv)
    
    # 創建一個 PlotWidget
    plot_widget = pg.plot(title="PyQtGraph Example")
    
    # 生成一些示範資料
    x = [1, 2, 3, 4, 5]
    y = [1, 4, 9, 16, 25]
    
    # 添加資料到 PlotWidget
    plot_widget.plot(x, y, pen='r')  # 紅色線條
    
    # 取得 X 軸物件
    x_axis = plot_widget.getAxis('bottom')
    
    # 設定 X 軸標籤顏色
    x_axis.setLabel('Custom X Label', color=QColor(0, 0, 255))
    
    # 顯示視窗
    plot_widget.show()
    
    # 執行應用程式
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
