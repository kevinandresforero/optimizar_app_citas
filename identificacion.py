import numpy as np
from sklearn.neural_network import MLPRegressor


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def metricas(y_true, y_pred):
    return {'MAE': mae(y_true, y_pred),
            'MSE': mse(y_true, y_pred),
            'RMSE': rmse(y_true, y_pred)}


# =====================================================================
# Neural Network for system identification
# =====================================================================

class IdentificadorNN:
    def __init__(self, hidden_layer_sizes=(16, 16), max_iter=1000, random_state=42):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=random_state,
            tol=1e-6,
            verbose=False,
        )
        self.history = None

    def entrenar(self, X, y):
        X_train, y_train = X, y
        self.model.fit(X_train, y_train)
        self.history = self.model.loss_curve_
        return self

    def predecir(self, X):
        return self.model.predict(X)

    def predecir_trayectoria(self, x0, y0, n_steps=200):
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        x[0], y[0] = x0, y0
        for k in range(n_steps - 1):
            pred = self.model.predict([[x[k], y[k]]])[0]
            x[k + 1] = max(pred[0], 0)
            y[k + 1] = max(pred[1], 0)
        return x, y


# =====================================================================
# ANFIS for system identification
# =====================================================================

class IdentificadorANFIS:
    def __init__(self, n_mf=3, lr=0.01, n_epochs=200, seed=42):
        self.n_mf = n_mf
        self.lr = lr
        self.n_epochs = n_epochs
        np.random.seed(seed)

        self.mu = np.array([
            [30, 50, 70],
            [30, 50, 70],
        ])
        self.sigma = np.full((2, n_mf), 10.0)

        n_input = 2 * n_mf
        self.W1 = np.random.randn(n_input, 12) * 0.1
        self.b1 = np.zeros(12)
        self.W2 = np.random.randn(12, 2) * 0.1
        self.b2 = np.zeros(2)

        self.loss_history = []

    def _fuzzify(self, x):
        phi = []
        for i in range(2):
            for j in range(self.n_mf):
                val = np.exp(-((x[i] - self.mu[i, j]) ** 2) / (2 * self.sigma[i, j] ** 2 + 1e-8))
                phi.append(val)
        return np.array(phi)

    def _forward(self, phi):
        h = np.tanh(self.W1.T @ phi + self.b1)
        out = self.W2.T @ h + self.b2
        return out, h

    def _backward(self, phi, h, out, target, lr):
        error = out - target
        d_W2 = np.outer(h, error)
        d_b2 = error
        d_h = self.W2 @ error * (1 - h ** 2)
        d_W1 = np.outer(phi, d_h)
        d_b1 = d_h
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1

    def entrenar(self, X, y):
        n_samples = X.shape[0]
        for epoch in range(self.n_epochs):
            lr_cur = self.lr * (1 - epoch / self.n_epochs)
            epoch_loss = 0.0
            for i in range(n_samples):
                phi = self._fuzzify(X[i])
                out, h = self._forward(phi)
                loss = np.sum((out - y[i]) ** 2)
                epoch_loss += loss
                self._backward(phi, h, out, y[i], lr_cur)
            self.loss_history.append(epoch_loss / n_samples)
        return self

    def predecir(self, X):
        preds = []
        for i in range(X.shape[0]):
            phi = self._fuzzify(X[i])
            out, _ = self._forward(phi)
            preds.append(out)
        return np.array(preds)

    def predecir_trayectoria(self, x0, y0, n_steps=200):
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        x[0], y[0] = x0, y0
        for k in range(n_steps - 1):
            phi = self._fuzzify([x[k], y[k]])
            out, _ = self._forward(phi)
            x[k + 1] = max(out[0], 0)
            y[k + 1] = max(out[1], 0)
        return x, y
