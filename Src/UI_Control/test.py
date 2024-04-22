import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

window = pg.plot()

x1 = [5, 5, 7, 10, 3, 8, 9, 1, 6, 2]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
bargraph = pg.BarGraphItem(x0=0, y=y, height=0.6, width=x1)
window.setXRange(0,2)
window.addItem(bargraph)
QtGui.QApplication.instance().exec_()