# app.py
import sys
import os

# Ensure top-level package import works even when running as a script
sys.path.insert(0, os.path.dirname(__file__))

# Try PyQt5, fallback to PySide6 (consistent with UI code)
try:
    from PyQt5.QtWidgets import QApplication
except Exception:
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:
        raise RuntimeError("Install PyQt5 or PySide6 (pip install pyqt5 OR pip install pyside6)")

from ui.main_window import PerceptronTrainerMainWindow

def main():
    app = QApplication(sys.argv)
    win = PerceptronTrainerMainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
