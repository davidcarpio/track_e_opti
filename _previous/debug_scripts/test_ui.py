import sys
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

app = QApplication(sys.argv)
win = MainWindow()
win.show()
print("UI instantiated successfully.")
