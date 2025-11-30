# ml/perceptron.py
import numpy as np

class Perceptron:
    """
    Simple perceptron supporting 2D or 3D features.
    """
    def __init__(self, w=None, b=0.0, eta=1.0):
        if w is None:
            self.w = np.zeros(2)
        else:
            self.w = np.array(w, dtype=float)
        self.b = float(b)
        self.eta = float(eta)
        self.history = []

    def predict(self, x, threshold_ge=True):
        net = float(np.dot(self.w, x) + self.b)
        if threshold_ge:
            return 1 if net >= 0 else 0, net
        else:
            return 1 if net > 0 else 0, net

    def train_one_epoch(self, dataset, threshold_ge=True, record=False):
        updates = 0
        for x, y in dataset:
            y_pred, net = self.predict(x, threshold_ge)
            err = y - y_pred
            if record:
                self.history.append({'sample_x': tuple(x.tolist()), 'y': int(y), 'net_before': net, 'err': int(err), 'w_before': tuple(self.w.tolist()), 'b_before': float(self.b), 'phase':'before'})
            if err != 0:
                self.w = self.w + self.eta * err * x
                self.b = self.b + self.eta * err
                updates += 1
            if record:
                self.history.append({'sample_x': tuple(x.tolist()), 'y': int(y), 'net_before': None, 'err': int(err), 'w_before': tuple(self.w.tolist()), 'b_before': float(self.b), 'phase':'after'})
        return updates

    def train(self, dataset, max_epochs=10, threshold_ge=True, record=False):
        self.history = []
        for epoch in range(1, int(max_epochs)+1):
            updates = self.train_one_epoch(dataset, threshold_ge, record)
            if updates == 0:
                return epoch, True
        return max_epochs, False
