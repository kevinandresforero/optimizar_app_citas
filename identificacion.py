import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge


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
# RBF layer (ANFIS)
# =====================================================================

_N_MF = 9
_SIGMA = 18.0
_CENTROS = np.array([np.linspace(10, 90, _N_MF),
                     np.linspace(10, 90, _N_MF)])
_N_RULES = _N_MF ** 2
_RULE_IDX = []
for r in range(_N_RULES):
    idx = []
    tmp = r
    for _ in range(2):
        idx.append(tmp % _N_MF)
        tmp //= _N_MF
    _RULE_IDX.append(idx)


def _rbf_transform(Z):
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    Phi = np.zeros((n, _N_RULES))
    for i in range(n):
        mu = np.zeros((2, _N_MF))
        for j in range(2):
            d = Z[i, j] - _CENTROS[j]
            mu[j] = np.exp(-(d ** 2) / (2 * _SIGMA ** 2 + 1e-8))
        w = np.ones(_N_RULES)
        for r in range(_N_RULES):
            for j in range(2):
                w[r] *= mu[j, _RULE_IDX[r][j]]
        Phi[i] = w / (np.sum(w) + 1e-8)
    return Phi


# =====================================================================
# NN: MLP sklearn con datos augmentados de trayectorias diversas
# =====================================================================

class IdentificadorNN:
    def __init__(self, hidden_layer_sizes=(64, 64, 32), max_iter=3000,
                 random_state=42, alpha=0.001):
        self.arch = hidden_layer_sizes
        self.rs = random_state
        self.alpha = alpha
        self.max_iter = max_iter
        self.model = None
        self.history = None

    def _simular_trayectoria(self, x0, y0, n_steps):
        a, b, c, d = 0.3, 0.006, 0.018, 0.7
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        x[0], y[0] = x0, y0
        for k in range(n_steps - 1):
            dx = a * x[k] - b * x[k] * y[k]
            dy = c * x[k] * y[k] - d * y[k]
            x[k + 1] = max(x[k] + 0.1 * dx, 0)
            y[k + 1] = max(y[k] + 0.1 * dy, 0)
            if x[k + 1] < 0.01 and y[k + 1] < 0.01:
                x[k + 1:] = 0
                y[k + 1:] = 0
                break
        return x, y

    def entrenar(self, X, y):
        rng = np.random.RandomState(self.rs)
        
        # Datos originales con ruido
        X_aug = [X]
        y_aug = [y]
        for _ in range(1):
            X_aug.append(X + rng.randn(*X.shape) * 0.3)
            y_aug.append(y)
        
        # Datos extra de trayectorias reales diversas
        X_extra, y_extra = [], []
        for _ in range(1500):
            x0 = rng.uniform(5, 95)
            y0 = rng.uniform(5, 95)
            try:
                xr, yr = self._simular_trayectoria(x0, y0, 100)
                for k in range(min(99, len(xr) - 1)):
                    X_extra.append([xr[k], yr[k]])
                    y_extra.append([xr[k + 1], yr[k + 1]])
            except:
                pass
        X_extra = np.array(X_extra)
        y_extra = np.array(y_extra)
        
        # Combinar todos los datos
        X_all = np.vstack(X_aug + [X_extra])
        y_all = np.vstack(y_aug + [y_extra])
        
        self.model = MLPRegressor(
            hidden_layer_sizes=self.arch,
            activation='relu', solver='adam',
            alpha=self.alpha, max_iter=self.max_iter,
            random_state=self.rs, tol=1e-6,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, verbose=False,
        ).fit(X_all, y_all)
        self.history = list(self.model.loss_curve_)
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
# ANFIS: RBF (rejilla fija 9x9) + Ridge + datos extra de trayectorias
# =====================================================================

class IdentificadorANFIS:
    def __init__(self, alpha=0.30, random_state=42):
        self.alpha = alpha
        self.rs = random_state
        self.model = None

    def _simular_trayectoria(self, x0, y0, n_steps):
        a, b, c, d = 0.3, 0.006, 0.018, 0.7
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        x[0], y[0] = x0, y0
        for k in range(n_steps - 1):
            dx = a * x[k] - b * x[k] * y[k]
            dy = c * x[k] * y[k] - d * y[k]
            x[k + 1] = max(x[k] + 0.1 * dx, 0)
            y[k + 1] = max(y[k] + 0.1 * dy, 0)
            if x[k + 1] < 0.01 and y[k + 1] < 0.01:
                x[k + 1:] = 0
                y[k + 1:] = 0
                break
        return x, y

    def entrenar(self, X, y):
        rng = np.random.RandomState(self.rs)
        
        # Datos extra de trayectorias reales
        X_extra, y_extra = [], []
        for _ in range(2000):
            x0 = rng.uniform(5, 95)
            y0 = rng.uniform(5, 95)
            try:
                xr, yr = self._simular_trayectoria(x0, y0, 80)
                for k in range(min(79, len(xr) - 1)):
                    X_extra.append([xr[k], yr[k]])
                    y_extra.append([xr[k + 1], yr[k + 1]])
            except:
                pass
        X_extra = np.array(X_extra)
        y_extra = np.array(y_extra)
        
        # Transformar a features RBF
        Phi_tr = _rbf_transform(X)
        Phi_extra = _rbf_transform(X_extra)
        
        # Combinar y entrenar
        Phi_all = np.vstack([Phi_tr, Phi_extra])
        y_all = np.vstack([y, y_extra])
        
        self.model = Ridge(alpha=self.alpha, random_state=self.rs)
        self.model.fit(Phi_all, y_all)
        return self

    def predecir(self, X):
        return self.model.predict(_rbf_transform(X))

    def predecir_trayectoria(self, x0, y0, n_steps=200):
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        x[0], y[0] = x0, y0
        for k in range(n_steps - 1):
            Phi = _rbf_transform(np.array([[x[k], y[k]]]))
            pred = self.model.predict(Phi)[0]
            x[k + 1] = max(pred[0], 0)
            y[k + 1] = max(pred[1], 0)
        return x, y