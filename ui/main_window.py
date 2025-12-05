# ui/main_window.py
import math
import time
import numpy as np
from functools import partial

import matplotlib.pyplot as plt
from matplotlib import cm

# Matplotlib for plotting
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Try to import PyQt5, fallback to PySide6
try:
    from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                 QLineEdit, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                                 QMessageBox, QSpinBox, QTextEdit, QGroupBox, QSizePolicy, QCheckBox)
    from PyQt5.QtCore import Qt, QCoreApplication
    qt_binding = 'PyQt5'
except Exception:
    from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                   QLineEdit, QPushButton, QComboBox, QTableWidget, QTableWidgetItem,
                                   QMessageBox, QSpinBox, QTextEdit, QGroupBox, QSizePolicy, QCheckBox)
    from PySide6.QtCore import Qt, QCoreApplication
    qt_binding = 'PySide6'

from ml.perceptron import Perceptron
from ml.mlp import SimpleMLP
from ml.utils import parse_custom_points, is_3d_dataset
from ui.themes import DARK_THEME, LIGHT_THEME
from ui.plot3d import show_mlp_3d_surface  # your helper file


class PerceptronTrainerMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Perceptron Trainer — Modular')
        self.resize(1200, 780)

        self.dark_mode = True  # default
        self.model_mode = 'Perceptron'  # or 'SimpleMLP'

        # Layouts
        main_layout = QHBoxLayout(self)
        control_layout = QVBoxLayout()
        plot_layout = QVBoxLayout()

        # Dataset group
        dataset_group = QGroupBox('Dataset / Problem')
        ds_layout = QVBoxLayout()
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['AND', 'OR', 'XOR', 'Custom'])
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_change)
        ds_layout.addWidget(self.dataset_combo)
        ds_layout.addWidget(QLabel('Custom points (one per line): x1,x2,y  or x1,x2,x3,y for 3D'))
        self.custom_text = QTextEdit()
        self.custom_text.setPlaceholderText('0,0,0\n0,1,0\n1,0,0\n1,1,1')
        ds_layout.addWidget(self.custom_text)
        dataset_group.setLayout(ds_layout)
        control_layout.addWidget(dataset_group)

        # Model selection
        model_group = QGroupBox('Model')
        model_layout = QHBoxLayout()
        self.model_select = QComboBox()
        self.model_select.addItems(['Perceptron', 'SimpleMLP'])
        self.model_select.currentTextChanged.connect(self.on_model_change)
        model_layout.addWidget(QLabel('Mode:'))
        model_layout.addWidget(self.model_select)
        model_group.setLayout(model_layout)
        control_layout.addWidget(model_group)

        # Parameters
        params_group = QGroupBox('Parameters')
        params_layout = QVBoxLayout()

        # Number of inputs spinbox
        h_inputs = QHBoxLayout()
        h_inputs.addWidget(QLabel('Number of inputs:'))
        self.input_dim_spin = QSpinBox()
        self.input_dim_spin.setRange(1, 3)   # increase max if needed
        self.input_dim_spin.setValue(2)
        self.input_dim_spin.valueChanged.connect(self.on_input_dim_change)
        h_inputs.addWidget(self.input_dim_spin)
        params_layout.addLayout(h_inputs)

        # Dynamic weights layout
        self.weights_layout = QHBoxLayout()
        self.weight_edits = []  # list[QLineEdit]
        params_layout.addLayout(self.weights_layout)

        # Bias and eta
        h2 = QHBoxLayout()
        h2.addWidget(QLabel('b:'))
        self.b_input = QLineEdit('0')
        h2.addWidget(self.b_input)
        h2.addWidget(QLabel('eta:'))
        self.eta_input = QLineEdit('1')
        h2.addWidget(self.eta_input)
        params_layout.addLayout(h2)

        # Max epochs + threshold
        h3 = QHBoxLayout()
        h3.addWidget(QLabel('Max epochs:'))
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 10000)
        self.epochs_input.setValue(10)
        h3.addWidget(self.epochs_input)
        self.threshold_combo = QComboBox()
        self.threshold_combo.addItems(['net >= 0 -> class 1', 'net > 0 -> class 1'])
        h3.addWidget(self.threshold_combo)
        params_layout.addLayout(h3)

        # random init checkbox & restart button
        h4 = QHBoxLayout()
        self.random_init_cb = QCheckBox('Random init w/b in [-1,1]')
        h4.addWidget(self.random_init_cb)
        self.restart_btn = QPushButton('Restart Training (keep GUI)')
        self.restart_btn.clicked.connect(self.on_restart)
        h4.addWidget(self.restart_btn)
        params_layout.addLayout(h4)

        params_group.setLayout(params_layout)
        control_layout.addWidget(params_group)

        # Buttons / controls
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton('Train (animate)')
        self.train_btn.clicked.connect(self.on_train)
        btn_layout.addWidget(self.train_btn)
        self.step_btn = QPushButton('Step (single update)')
        self.step_btn.clicked.connect(self.on_step)
        btn_layout.addWidget(self.step_btn)
        self.clear_btn = QPushButton('Clear')
        self.clear_btn.clicked.connect(self.on_clear)
        btn_layout.addWidget(self.clear_btn)
        self.theme_toggle = QPushButton('Toggle Theme')
        self.theme_toggle.clicked.connect(self.toggle_theme)
        btn_layout.addWidget(self.theme_toggle)

        # New button: MLP 3D surface
        self.plot3d_btn = QPushButton('MLP 3D Surface')
        self.plot3d_btn.clicked.connect(self.on_plot_mlp_3d)
        btn_layout.addWidget(self.plot3d_btn)

        control_layout.addLayout(btn_layout)

        # Table log
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(['step', 'phase', 'x1', 'x2', 'x3', 'y', 'net_before', 'w,b'])
        control_layout.addWidget(QLabel('Training log'))
        control_layout.addWidget(self.table, stretch=1)

        # left column done
        main_layout.addLayout(control_layout, 1)

        # Right column - plots and visualizer
        # Decision plot area (2D or 3D)
        self.fig_dec = Figure(figsize=(6, 5))
        self.canvas_dec = FigureCanvas(self.fig_dec)
        plot_layout.addWidget(self.canvas_dec, stretch=3)

        # Accuracy / errors OR loss
        self.fig_acc = Figure(figsize=(6, 2))
        self.canvas_acc = FigureCanvas(self.fig_acc)
        plot_layout.addWidget(self.canvas_acc, stretch=1)

        # Activation heatmap canvas
        self.fig_act = Figure(figsize=(4, 2))
        self.canvas_act = FigureCanvas(self.fig_act)
        plot_layout.addWidget(self.canvas_act, stretch=1)

        # Weights heatmap canvas
        self.fig_w = Figure(figsize=(4, 2))
        self.canvas_w = FigureCanvas(self.fig_w)
        plot_layout.addWidget(self.canvas_w, stretch=1)

        # Simple network visualizer placeholder (small)
        self.net_vis_label = QLabel('Network Visualizer (MLP mode shows nodes & weights)')
        plot_layout.addWidget(self.net_vis_label)

        # status
        self.status_label = QLabel('Ready — using %s' % qt_binding)
        plot_layout.addWidget(self.status_label)

        main_layout.addLayout(plot_layout, 2)

        # internal holders
        self.perceptron = None
        self.mlp = None
        self.dataset_cache = None
        self.epochs = []
        self.accuracies = []
        self.errors_per_epoch = []
        self.current_step_index = 0  # for step-by-step
        self.latest_history = []

        # apply default theme
        self.apply_theme(self.dark_mode)

        # initialize dynamic weight fields
        self.on_input_dim_change(self.input_dim_spin.value())

        # ensure custom field enabled only when needed
        self.on_dataset_change(self.dataset_combo.currentText())

    # -------------------- dynamic weights ---------------------
    def on_input_dim_change(self, dim: int):
        # clear existing widgets in weights_layout
        while self.weights_layout.count():
            item = self.weights_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        self.weight_edits = []

        # create new labels + line edits
        for i in range(dim):
            label = QLabel(f"w{i+1}:")
            edit = QLineEdit('0')
            self.weights_layout.addWidget(label)
            self.weights_layout.addWidget(edit)
            self.weight_edits.append(edit)

    def get_current_weights(self):
        try:
            return [float(e.text()) for e in self.weight_edits]
        except ValueError:
            raise ValueError("Weights must be numeric")

    # -------------------- theme ---------------------
    def apply_theme(self, dark: bool):
        if dark:
            self.setStyleSheet(DARK_THEME)
        else:
            self.setStyleSheet(LIGHT_THEME)
        self.dark_mode = dark

    def toggle_theme(self):
        self.apply_theme(not self.dark_mode)

    # -------------------- dataset & model ----------------
    def on_dataset_change(self, text):
        self.custom_text.setEnabled(text == 'Custom')

    def on_model_change(self, text):
        self.model_mode = text

    def parse_dataset(self):
        name = self.dataset_combo.currentText()
        if name == 'AND':
            pts = [(np.array([0.0, 0.0]), 0), (np.array([0.0, 1.0]), 0),
                   (np.array([1.0, 0.0]), 0), (np.array([1.0, 1.0]), 1)]
        elif name == 'OR':
            pts = [(np.array([0.0, 0.0]), 0), (np.array([0.0, 1.0]), 1),
                   (np.array([1.0, 0.0]), 1), (np.array([1.0, 1.0]), 1)]
        elif name == 'XOR':
            pts = [(np.array([0.0, 0.0]), 0), (np.array([0.0, 1.0]), 1),
                   (np.array([1.0, 0.0]), 1), (np.array([1.0, 1.0]), 0)]
        else:
            txt = self.custom_text.toPlainText().strip()
            pts = parse_custom_points(txt)
        self.dataset_cache = pts
        return pts

    # -------------------- UI actions ----------------
    def on_clear(self):
        self.table.setRowCount(0)
        self.fig_dec.clear()
        self.canvas_dec.draw()
        self.fig_acc.clear()
        self.canvas_acc.draw()
        self.fig_act.clear()
        self.canvas_act.draw()
        self.fig_w.clear()
        self.canvas_w.draw()
        self.status_label.setText('Cleared')
        self.epochs.clear()
        self.accuracies.clear()
        self.errors_per_epoch.clear()

    def on_restart(self):
        # Reset model state without changing GUI fields or dataset
        try:
            dataset = self.parse_dataset()
        except Exception as e:
            QMessageBox.critical(self, "Dataset error", str(e))
            return

        try:
            w_list = self.get_current_weights()
            b = float(self.b_input.text())
            eta = float(self.eta_input.text())
        except Exception as e:
            QMessageBox.critical(self, "Param error", str(e))
            return

        input_dim = len(w_list)

        # random init (respect input_dim)
        if self.random_init_cb.isChecked():
            w_list = [np.random.uniform(-1, 1) for _ in range(input_dim)]
            b = np.random.uniform(-1, 1)

        if self.model_mode == 'Perceptron':
            self.perceptron = Perceptron(w=w_list, b=b, eta=eta)
        else:
            self.mlp = SimpleMLP(input_dim=input_dim, hidden_units=2, eta=eta)
            # optional seeding from GUI weights
            try:
                self.mlp.set_weights_from_flat(w_list + [b])
            except Exception:
                pass

        self.table.setRowCount(0)
        self.fig_dec.clear()
        self.canvas_dec.draw()
        self.fig_acc.clear()
        self.canvas_acc.draw()
        self.fig_act.clear()
        self.canvas_act.draw()
        self.fig_w.clear()
        self.canvas_w.draw()
        self.status_label.setText('Restarted model with new parameters')

    # single-step mode (per sample)
    def on_step(self):
        if (self.perceptron is None and self.model_mode == 'Perceptron') or \
           (self.mlp is None and self.model_mode == 'SimpleMLP'):
            self.on_restart()
        try:
            dataset = self.parse_dataset()
        except Exception as e:
            QMessageBox.critical(self, "Dataset", str(e))
            return

        if self.current_step_index == 0:
            self.table.setRowCount(0)
            self.epochs.clear()
            self.accuracies.clear()
            self.errors_per_epoch.clear()

        if self.model_mode == 'Perceptron':
            per = self.perceptron
            idx = (self.current_step_index // 2) % len(dataset)
            x, y = dataset[idx]
            y_pred, net = per.predict(x, threshold_ge=(self.threshold_combo.currentIndex() == 0))
            err = y - y_pred
            per.history = getattr(per, 'history', [])
            per.history.append({'sample_x': tuple(x.tolist()), 'y': int(y),
                                'net_before': net, 'err': int(err),
                                'w_before': tuple(per.w.tolist()),
                                'b_before': float(per.b), 'phase': 'before'})
            if err != 0:
                per.w = per.w + per.eta * err * x
                per.b = per.b + per.eta * err
            per.history.append({'sample_x': tuple(x.tolist()), 'y': int(y),
                                'net_before': None, 'err': int(err),
                                'w_before': tuple(per.w.tolist()),
                                'b_before': float(per.b), 'phase': 'after'})
            rec = per.history[-2]
            self.append_record_to_table(rec)
            rec2 = per.history[-1]
            self.append_record_to_table(rec2)
            self.draw_current_model(per.w, per.b)
        else:
            QMessageBox.information(self, "Step", "Step for MLP not implemented (use Train animate).")

        self.current_step_index += 2

    def append_record_to_table(self, rec):
        i = self.table.rowCount()
        self.table.insertRow(i)
        self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
        self.table.setItem(i, 1, QTableWidgetItem(rec.get('phase', '')))
        x1, x2 = (rec['sample_x'][0], rec['sample_x'][1]) if len(rec['sample_x']) >= 2 else (rec['sample_x'][0], 0)
        x3 = rec['sample_x'][2] if len(rec['sample_x']) == 3 else ''
        self.table.setItem(i, 2, QTableWidgetItem(f"{x1:.3f}"))
        self.table.setItem(i, 3, QTableWidgetItem(f"{x2:.3f}"))
        self.table.setItem(i, 4, QTableWidgetItem(f"{x3}" if x3 != '' else ""))
        self.table.setItem(i, 5, QTableWidgetItem(str(rec['y'])))
        netb = rec.get('net_before', '')
        self.table.setItem(i, 6, QTableWidgetItem('' if netb is None else f"{netb:.3f}"))
        wb = f"w={tuple(rec['w_before'])}, b={rec['b_before']:.3f}"
        self.table.setItem(i, 7, QTableWidgetItem(wb))
        self.table.scrollToBottom()

    # main training (animated)
    def on_train(self):
        try:
            dataset = self.parse_dataset()
            if len(dataset) == 0:
                raise ValueError("Dataset empty")
        except Exception as e:
            QMessageBox.critical(self, "Dataset", str(e))
            return

        if self.model_mode == 'Perceptron':
            if self.perceptron is None:
                self.on_restart()
            model = self.perceptron
        else:
            if self.mlp is None:
                self.on_restart()
            model = self.mlp

        try:
            max_epochs = int(self.epochs_input.value())
            threshold_ge = (self.threshold_combo.currentIndex() == 0)
            delay_ms = 80
        except Exception as e:
            QMessageBox.critical(self, "Param error", str(e))
            return

        self.table.setRowCount(0)
        self.epochs.clear()
        self.accuracies.clear()
        self.errors_per_epoch.clear()

        self.fig_dec.clear()
        self.canvas_dec.draw()
        self.fig_acc.clear()
        self.canvas_acc.draw()
        self.fig_act.clear()
        self.canvas_act.draw()
        self.fig_w.clear()
        self.canvas_w.draw()

        if self.model_mode == 'Perceptron':
            per = model
            per.history = []
            for epoch in range(1, max_epochs + 1):
                errors = 0
                for x, y in dataset:
                    y_pred, net = per.predict(x, threshold_ge)
                    err = y - y_pred
                    rec_before = {'sample_x': tuple(x.tolist()), 'y': int(y),
                                  'net_before': net, 'err': int(err),
                                  'w_before': tuple(per.w.tolist()),
                                  'b_before': float(per.b), 'phase': 'before'}
                    per.history.append(rec_before)
                    self.append_record_to_table(rec_before)
                    if err != 0:
                        per.w = per.w + per.eta * err * x
                        per.b = per.b + per.eta * err
                        errors += 1
                    rec_after = {'sample_x': tuple(x.tolist()), 'y': int(y),
                                 'net_before': None, 'err': int(err),
                                 'w_before': tuple(per.w.tolist()),
                                 'b_before': float(per.b), 'phase': 'after'}
                    per.history.append(rec_after)
                    self.append_record_to_table(rec_after)
                    self.draw_current_model(per.w, per.b)
                    QCoreApplication.processEvents()
                    time.sleep(delay_ms / 1000.0)
                correct = sum(1 for x, y in dataset if per.predict(x, threshold_ge)[0] == y)
                acc = correct / len(dataset)
                self.epochs.append(epoch)
                self.accuracies.append(acc)
                self.errors_per_epoch.append(errors)
                self.draw_accuracy_plot()
                self.status_label.setText(f"Epoch {epoch} done — acc {acc*100:.1f}% errors={errors}")
                if errors == 0:
                    break
            self.status_label.setText(f"Training finished. final w={tuple(per.w.tolist())}, b={per.b:.3f}")
        else:
            mlp = model
            mlp.train(dataset, max_epochs=max_epochs,
                      callback=self.mlp_epoch_callback, report_every=1)
            try:
                layers = mlp.get_layers()
                self.draw_decision_region_heatmap(layers, grid_res=140)
            except Exception:
                pass
            self.status_label.setText("MLP training finished")

            # optionally auto-show 3D surface for 2D inputs
            try:
                X = np.array([x for x, y in dataset])
                if X.shape[1] == 2:
                    show_mlp_3d_surface(mlp, X, resolution=60)
            except Exception:
                pass

    # ------------- extra UI action: 3D surface button --------------
    def on_plot_mlp_3d(self):
        if self.mlp is None:
            QMessageBox.information(self, "MLP 3D surface",
                                    "Train a SimpleMLP model first.")
            return
        try:
            dataset = self.dataset_cache if self.dataset_cache is not None else self.parse_dataset()
            X = np.array([x for x, y in dataset])
            if X.shape[1] != 2:
                QMessageBox.information(self, "MLP 3D surface",
                                        "3D surface is only available for 2D inputs (x1, x2).")
                return
        except Exception as e:
            QMessageBox.critical(self, "Dataset error", str(e))
            return

        show_mlp_3d_surface(self.mlp, X, resolution=60)

    # ------------- drawing helpers --------------
    def mlp_epoch_callback(self, epoch, activations, losses, layers):
        self.epochs.append(epoch)
        self.accuracies.append(None)
        self.draw_loss_curve(losses)

        if len(activations) > 1:
            A1 = activations[1]
            self.fig_act.clf()
            ax = self.fig_act.add_subplot(111)
            im = ax.imshow(A1.T, aspect='auto', cmap='viridis')
            ax.set_title(f'Hidden activations (epoch {epoch})')
            ax.set_ylabel('hidden neuron')
            ax.set_xlabel('sample index')
            self.fig_act.colorbar(im, ax=ax, fraction=0.05)
            self.canvas_act.draw()

        self.fig_w.clf()
        W1 = layers[0]['W']
        W2 = layers[1]['W']
        ax1 = self.fig_w.add_subplot(1, 2, 1)
        im1 = ax1.imshow(W1, aspect='auto', cmap='coolwarm')
        ax1.set_title('W1 (hidden weights)')
        ax1.set_xlabel('input dim')
        ax1.set_ylabel('hidden unit')
        self.fig_w.colorbar(im1, ax=ax1, fraction=0.05)
        ax2 = self.fig_w.add_subplot(1, 2, 2)
        im2 = ax2.imshow(W2, aspect='auto', cmap='coolwarm')
        ax2.set_title('W2 (output weights)')
        ax2.set_xlabel('hidden unit')
        self.fig_w.colorbar(im2, ax=ax2, fraction=0.05)
        self.canvas_w.draw()

        try:
            dataset = self.dataset_cache if self.dataset_cache is not None else self.parse_dataset()
            X = np.array([x for x, y in dataset])
            if X.shape[1] == 2:
                self.draw_decision_region_heatmap(layers)
        except Exception:
            pass

        self.status_label.setText(f"MLP epoch {epoch} — loss {losses[-1]:.6f}")
        QCoreApplication.processEvents()

    def draw_current_model(self, w, b):
        self.fig_dec.clf()
        if len(w) == 3:
            ax = self.fig_dec.add_subplot(111, projection='3d')
        else:
            ax = self.fig_dec.add_subplot(111)

        dataset = self.dataset_cache if self.dataset_cache is not None else self.parse_dataset()
        pts = np.array([x for x, y in dataset])
        ys = np.array([y for x, y in dataset])

        if len(w) == 2:
            class0 = pts[ys == 0]
            class1 = pts[ys == 1]
            if len(class0) > 0:
                ax.scatter(class0[:, 0], class0[:, 1], marker='o', s=80, edgecolors='k')
            if len(class1) > 0:
                ax.scatter(class1[:, 0], class1[:, 1], marker='s', s=80, edgecolors='k')
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
            if abs(w[1]) > 1e-8:
                xs = np.linspace(-0.5, 1.5, 200)
                ys_line = (-w[0] * xs - b) / w[1]
                ax.plot(xs, ys_line, '-', linewidth=2, color='#ffcc66')
            else:
                if abs(w[0]) > 1e-8:
                    x0 = -b / w[0]
                    ax.axvline(x0, linestyle='--', linewidth=2, color='#ffcc66')
            ax.set_title('2D decision boundary')
        elif len(w) == 3:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=ys, cmap='bwr', s=60)
            xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 10),
                                 np.linspace(-0.5, 1.5, 10))
            if abs(w[2]) > 1e-8:
                zz = (-w[0] * xx - w[1] * yy - b) / w[2]
                ax.plot_surface(xx, yy, zz, alpha=0.3, color=(0.8, 0.6, 0.2))
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 1.5)
            ax.set_zlim(-0.5, 1.5)
            ax.set_title('3D decision plane')
        else:
            ax.text(0.5, 0.5,
                    "Visualization only implemented for 2 or 3 inputs.",
                    ha='center', va='center', transform=ax.transAxes)

        self.canvas_dec.draw()

    def draw_current_model_from_flat(self, wflat):
        self.fig_dec.clf()
        ax = self.fig_dec.add_subplot(111)
        ax.text(0.5, 0.5,
                "MLP visual: see network visualizer.\nDecision regions are nonlinear.",
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
        self.canvas_dec.draw()

    def draw_accuracy_plot(self):
        self.fig_acc.clf()
        ax = self.fig_acc.add_subplot(111)
        ax.plot(self.epochs, self.accuracies, marker='o', linestyle='-', label='accuracy')
        ax.bar(self.epochs, self.errors_per_epoch, alpha=0.3, label='errors')
        ax.set_ylim(0, 1)
        ax.set_xlabel('Epoch')
        ax.legend()
        self.canvas_acc.draw()

    def draw_loss_curve(self, losses):
        self.fig_acc.clf()
        ax = self.fig_acc.add_subplot(111)
        ax.plot(range(1, len(losses) + 1), losses, marker='o')
        ax.set_title('Loss per epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        self.canvas_acc.draw()

    def draw_decision_region_heatmap(self, layers, grid_res=120):
        W1 = layers[0]['W']
        b1 = layers[0]['b']
        W2 = layers[1]['W']
        b2 = layers[1]['b']

        dataset = self.dataset_cache if self.dataset_cache is not None else self.parse_dataset()
        X = np.array([x for x, y in dataset])
        ys = np.array([y for x, y in dataset])
        if X.shape[1] != 2:
            return

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                             np.linspace(y_min, y_max, grid_res))
        grid = np.c_[xx.ravel(), yy.ravel()]

        Z1 = sigmoid(grid.dot(W1.T) + b1)
        Z2 = sigmoid(Z1.dot(W2.T) + b2)
        Z = Z2.reshape(xx.shape)

        self.fig_dec.clf()
        ax = self.fig_dec.add_subplot(111)
        ax.contourf(xx, yy, Z, levels=30, cmap='coolwarm', alpha=0.8)
        ax.scatter(X[ys == 0, 0], X[ys == 0, 1], marker='o', edgecolors='k', label='class 0')
        ax.scatter(X[ys == 1, 0], X[ys == 1, 1], marker='s', edgecolors='k', label='class 1')
        ax.legend()
        ax.set_title('Decision region (MLP)')
        self.canvas_dec.draw()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
