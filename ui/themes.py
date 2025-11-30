# ui/themes.py

LIGHT_THEME = """
QWidget {
    background-color: #f5f7fa;
    color: #0b1730;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #d7e0ee;
    border-radius: 10px;
    margin-top: 8px;
    padding: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #20508a;
    font-weight: 600;
}
QPushButton {
    background-color: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #4a7bd0, stop:1 #325ea8);
    color: white;
    border-radius: 8px;
    padding: 6px 12px;
}
QPushButton:hover { opacity: 0.95; }
QLineEdit, QTextEdit, QComboBox, QSpinBox {
    background-color: #ffffff;
    border: 1px solid #c8d6ea;
    border-radius: 6px;
    padding: 4px;
    color: #051427;
}
QTableWidget {
    background-color: #ffffff;
    border: 1px solid #d7e0ee;
    gridline-color: #e9f0fb;
}
QHeaderView::section { background-color: #eaf2fb; color: #12385d; padding: 4px; }
"""

DARK_THEME = """
QWidget {
    background-color: #0f1720;
    color: #e6eef8;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif;
    font-size: 13px;
}
QGroupBox {
    background-color: #0b1220;
    border: 1px solid #233246;
    border-radius: 8px;
    margin-top: 8px;
    padding: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: #9fb3d6;
    font-weight: 600;
}
QPushButton {
    background-color: #2b6cb0;
    color: #fff;
    border-radius: 6px;
    padding: 6px 12px;
}
QPushButton:hover { background-color: #265f9e; }
QLineEdit, QTextEdit, QComboBox, QSpinBox {
    background-color: #071122;
    border: 1px solid #233246;
    border-radius: 6px;
    padding: 4px;
    color: #e6eef8;
}
QTableWidget {
    background-color: #071122;
    border: 1px solid #233246;
    color: #e6eef8;
    gridline-color: #233246;
}
QHeaderView::section { background-color: #122233; color: #d8ecff; padding: 4px; }
"""
