# ml/mlp.py
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(a):
    # expects a = sigmoid(z)
    return a * (1.0 - a)

class SimpleMLP:
    """
    Simple MLP with one hidden layer.
    - input_dim : number of input features
    - hidden_units: list or int (if int, single hidden layer with that many units)
    - eta: learning rate
    This implementation performs batch gradient descent with sigmoid activation.
    The train(...) method accepts a callback(epoch, activations, losses, layers)
    where:
      - activations is a list: [A0 (X), A1 (hidden activations), A2 (output)]
      - losses is a python list of loss values so far
      - layers is a list of dicts [{'W':W1,'b':b1}, {'W':W2,'b':b2}]
    """
    def __init__(self, input_dim=2, hidden_units=2, eta=0.1, seed=None):
        if seed is not None:
            np.random.seed(int(seed))
        self.input_dim = int(input_dim)
        if isinstance(hidden_units, (list, tuple)):
            self.hidden_units = int(hidden_units[0])
        else:
            self.hidden_units = int(hidden_units)
        self.eta = float(eta)

        # initialize weights (Xavier-ish)
        self.W1 = np.random.randn(self.hidden_units, self.input_dim) * np.sqrt(1.0 / max(1, self.input_dim))
        self.b1 = np.zeros((self.hidden_units,))
        self.W2 = np.random.randn(1, self.hidden_units) * np.sqrt(1.0 / max(1, self.hidden_units))
        self.b2 = np.zeros((1,))

        # history
        self.losses = []

    def forward(self, X):
        """
        X: shape (N, input_dim)
        returns activations [A0, A1, A2] where:
          A0 = X
          A1 = hidden activations (N, hidden_units)
          A2 = outputs (N, 1)
        """
        Z1 = X.dot(self.W1.T) + self.b1  # (N, hidden_units)
        A1 = sigmoid(Z1)
        Z2 = A1.dot(self.W2.T) + self.b2  # (N, 1)
        A2 = sigmoid(Z2)
        return [X, A1, A2]

    def predict_prob(self, X):
        return self.forward(X)[-1]  # (N,1) probabilities

    def predict(self, X, threshold=0.5):
        probs = self.predict_prob(X)
        return (probs >= threshold).astype(int)

    def get_layers(self):
        return [
            {'W': self.W1.copy(), 'b': self.b1.copy()},
            {'W': self.W2.copy(), 'b': self.b2.copy()}
        ]

    def train(self, dataset, max_epochs=100, callback=None, report_every=1):
        """
        dataset: list of (x_numpy_array, y_int) pairs, x shape (d,) or (1,d)
        callback: function(epoch, activations, losses, layers) - called after each epoch
        report_every: call callback every N epochs (set to 1 to call every epoch)
        """
        # prepare batch arrays
        X = np.array([x for x,y in dataset], dtype=float)  # (N, d)
        Y = np.array([y for x,y in dataset], dtype=float).reshape(-1,1)  # (N,1)

        self.losses = []
        N = len(X)
        for epoch in range(1, int(max_epochs)+1):
            # forward
            A0, A1, A2 = self.forward(X)
            # loss (MSE)
            loss = np.mean((A2 - Y)**2)
            self.losses.append(loss)

            # backpropagation (batch)
            dA2 = (A2 - Y)                         # (N,1)
            dZ2 = dA2 * dsigmoid(A2)              # (N,1)
            dW2 = (dZ2.T @ A1) / N                # (1, hidden_units)
            db2 = np.mean(dZ2, axis=0)            # (1,)

            dA1 = dZ2 @ self.W2                   # (N, hidden_units)
            dZ1 = dA1 * dsigmoid(A1)              # (N, hidden_units)
            dW1 = (dZ1.T @ A0) / N                # (hidden_units, input_dim)
            db1 = np.mean(dZ1, axis=0)            # (hidden_units,)

            # gradient descent update
            self.W2 -= self.eta * dW2
            self.b2 -= self.eta * db2
            self.W1 -= self.eta * dW1
            self.b1 -= self.eta * db1

            # call callback for UI update
            if callback is not None and (epoch % report_every == 0):
                activations = [A0, A1, A2]
                layers = self.get_layers()
                try:
                    callback(epoch, activations, list(self.losses), layers)
                except Exception:
                    # don't crash training if callback fails
                    pass

        return list(self.losses)
