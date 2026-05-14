import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer


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
    def __init__(self, hidden_layer_sizes=(32, 16), max_iter=3000,
                 random_state=42, alpha=0.001):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            tol=1e-6,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            verbose=False,
        )
        self.history = None

    def entrenar(self, X, y):
        rng = np.random.RandomState(42)
        X_aug = [X]
        y_aug = [y]
        for _ in range(3):
            X_aug.append(X + rng.randn(*X.shape) * 1.0)
            y_aug.append(y)
        X_all = np.vstack(X_aug)
        y_all = np.vstack(y_aug)
        self.model.fit(X_all, y_all)
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
# ANFIS for system identification (RBF grid + Ridge regression)
# =====================================================================

class IdentificadorANFIS:
    def __init__(self, n_mf=5, alpha=1.0, noise_std=0.5):
        self.n_mf = n_mf
        self.n_inputs = 2
        self.n_outputs = 2
        self.n_rules = n_mf ** self.n_inputs
        self.alpha = alpha
        self.noise_std = noise_std

        # Fixed Gaussian centres evenly covering [0, 100]
        self.centros = np.zeros((self.n_inputs, n_mf))
        for i in range(self.n_inputs):
            self.centros[i] = np.linspace(10, 90, n_mf)
        self.sigma = 12.0

        # Rule index mapping
        self.rule_idx = []
        for r in range(self.n_rules):
            idx = []
            tmp = r
            for _ in range(self.n_inputs):
                idx.append(tmp % self.n_mf)
                tmp //= self.n_mf
            self.rule_idx.append(idx)

        self.model = Ridge(alpha=self.alpha, random_state=42)
        self._entrenado = False
        self.loss_history = []

    def _transformar(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        Phi = np.zeros((n, self.n_rules))
        for i in range(n):
            mu = np.zeros((self.n_inputs, self.n_mf))
            for j in range(self.n_inputs):
                d = X[i, j] - self.centros[j]
                mu[j] = np.exp(-(d ** 2) / (2 * self.sigma ** 2 + 1e-8))
            w = np.ones(self.n_rules)
            for r in range(self.n_rules):
                for j in range(self.n_inputs):
                    w[r] *= mu[j, self.rule_idx[r][j]]
            Phi[i] = w / (np.sum(w) + 1e-8)
        return Phi

    def entrenar(self, X, y):
        rng = np.random.RandomState(42)
        X_aug = [X]
        y_aug = [y]
        for _ in range(2):
            X_aug.append(X + rng.randn(*X.shape) * self.noise_std)
            y_aug.append(y)
        X_all = np.vstack(X_aug)
        y_all = np.vstack(y_aug)

        Phi = self._transformar(X_all)
        self.model.fit(Phi, y_all)
        self._entrenado = True
        return self

    def predecir(self, X):
        if not self._entrenado:
            raise RuntimeError("Entrenar primero")
        Phi = self._transformar(X)
        return self.model.predict(Phi)

    def predecir_trayectoria(self, x0, y0, n_steps=200):
        if not self._entrenado:
            raise RuntimeError("Entrenar primero")
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        x[0], y[0] = x0, y0
        for k in range(n_steps - 1):
            Phi = self._transformar(np.array([[x[k], y[k]]]))
            pred = self.model.predict(Phi)[0]
            x[k + 1] = max(pred[0], 0)
            y[k + 1] = max(pred[1], 0)
        return x, y
