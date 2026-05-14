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
    def __init__(self, hidden_layer_sizes=(64, 64, 32), max_iter=5000,
                 random_state=42, alpha=0.0005, early_stopping=True,
                 validation_fraction=0.1, n_iter_no_change=20,
                 noise_std=1.0, n_augment=4):
        self.noise_std = noise_std
        self.n_augment = n_augment
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            tol=1e-6,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            verbose=False,
        )
        self.history = None
        self.best_loss_ = None

    def entrenar(self, X, y):
        X_aug = [X]
        y_aug = [y]
        rng = np.random.RandomState(42)
        for _ in range(self.n_augment):
            noise = rng.randn(*X.shape) * self.noise_std
            X_aug.append(X + noise)
            y_aug.append(y)

        X_all = np.vstack(X_aug)
        y_all = np.vstack(y_aug)

        self.model.fit(X_all, y_all)
        self.history = self.model.loss_curve_
        self.best_loss_ = self.model.best_loss_
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
# ANFIS for system identification (5-layer, hybrid learning)
# =====================================================================

class IdentificadorANFIS:
    def __init__(self, n_mf=4, lr=0.001, n_epochs=500, seed=42,
                 reg_lambda=0.01, patience=30, val_fraction=0.1,
                 noise_std=0.3, batch_size=3000):
        self.n_mf = n_mf
        self.lr = lr
        self.n_epochs = n_epochs
        self.reg_lambda = reg_lambda
        self.patience = patience
        self.val_fraction = val_fraction
        self.n_inputs = 2
        self.n_outputs = 2
        self.n_rules = n_mf ** self.n_inputs
        self.parado_temprano = False

        np.random.seed(seed)

        self.c = np.zeros((self.n_inputs, n_mf))
        for i in range(self.n_inputs):
            self.c[i] = np.linspace(20, 80, n_mf)
        self.sigma = np.full((self.n_inputs, n_mf), 15.0)

        self.conseq = np.zeros((self.n_rules, self.n_outputs, 3))

        self.rule_idx = []
        for r in range(self.n_rules):
            idx = []
            tmp = r
            for _ in range(self.n_inputs):
                idx.append(tmp % self.n_mf)
                tmp //= self.n_mf
            self.rule_idx.append(idx)

        self.loss_history = []
        self.val_loss_history = []
        self.noise_std = noise_std
        self.batch_size = batch_size

    def _compute_firing(self, x):
        x = np.asarray(x, dtype=float)
        mu = np.zeros((self.n_inputs, self.n_mf))
        for i in range(self.n_inputs):
            d = x[i] - self.c[i]
            mu[i] = np.exp(-(d ** 2) / (2 * self.sigma[i] ** 2 + 1e-8))
        w = np.ones(self.n_rules)
        for r in range(self.n_rules):
            for i in range(self.n_inputs):
                w[r] *= mu[i, self.rule_idx[r][i]]
        return w / (np.sum(w) + 1e-8)

    def _forward(self, x):
        x = np.asarray(x, dtype=float)
        w_norm = self._compute_firing(x)
        out = np.zeros(self.n_outputs)
        for d in range(self.n_outputs):
            for r in range(self.n_rules):
                p, q, r0 = self.conseq[r, d]
                out[d] += w_norm[r] * (p * x[0] + q * x[1] + r0)
        return out, w_norm

    def _backward(self, x, target, out, w_norm, lr):
        x = np.asarray(x, dtype=float)
        dL_dout = 2 * (out - target)

        dL_dwn = np.zeros(self.n_rules)
        for r in range(self.n_rules):
            for d in range(self.n_outputs):
                p, q, r0 = self.conseq[r, d]
                dL_dwn[r] += dL_dout[d] * (p * x[0] + q * x[1] + r0)

        mu = np.zeros((self.n_inputs, self.n_mf))
        for i in range(self.n_inputs):
            d = x[i] - self.c[i]
            mu[i] = np.exp(-(d ** 2) / (2 * self.sigma[i] ** 2 + 1e-8))
        w = np.ones(self.n_rules)
        for r in range(self.n_rules):
            for i in range(self.n_inputs):
                w[r] *= mu[i, self.rule_idx[r][i]]
        w_sum = np.sum(w) + 1e-8

        dL_dw = np.zeros(self.n_rules)
        for r in range(self.n_rules):
            for s in range(self.n_rules):
                kr = 1.0 if r == s else 0.0
                dL_dw[r] += dL_dwn[s] * (kr * w_sum - w[s]) / (w_sum ** 2)

        dL_dmu = np.zeros((self.n_inputs, self.n_mf))
        for r in range(self.n_rules):
            for i in range(self.n_inputs):
                j = self.rule_idx[r][i]
                prod = 1.0
                for k in range(self.n_inputs):
                    if k != i:
                        prod *= mu[k, self.rule_idx[r][k]]
                dL_dmu[i, j] += dL_dw[r] * prod

        dL_dc = np.zeros((self.n_inputs, self.n_mf))
        dL_dsig = np.zeros((self.n_inputs, self.n_mf))
        for i in range(self.n_inputs):
            for j in range(self.n_mf):
                m = mu[i, j]
                ss = self.sigma[i, j] ** 2 + 1e-8
                dmu_dc = m * (x[i] - self.c[i, j]) / ss
                dmu_ds = m * (x[i] - self.c[i, j]) ** 2 / (self.sigma[i, j] * ss + 1e-8)
                dL_dc[i, j] = dL_dmu[i, j] * dmu_dc
                dL_dsig[i, j] = dL_dmu[i, j] * dmu_ds

        self.c -= lr * dL_dc
        self.sigma -= lr * dL_dsig
        self.sigma = np.clip(self.sigma, 1.0, 50.0)

    def _ajustar_consecuentes(self, X, y):
        n = X.shape[0]
        A = np.zeros((n, self.n_rules * 3))
        for i in range(n):
            w_norm = self._compute_firing(X[i])
            for r in range(self.n_rules):
                A[i, 3 * r]     = w_norm[r] * X[i, 0]
                A[i, 3 * r + 1] = w_norm[r] * X[i, 1]
                A[i, 3 * r + 2] = w_norm[r]

        I = np.eye(A.shape[1])
        ATA = A.T @ A + self.reg_lambda * I
        for d in range(self.n_outputs):
            theta = np.linalg.solve(ATA, A.T @ y[:, d])
            for r in range(self.n_rules):
                self.conseq[r, d, 0] = theta[3 * r]
                self.conseq[r, d, 1] = theta[3 * r + 1]
                self.conseq[r, d, 2] = theta[3 * r + 2]

    def entrenar(self, X, y):
        n = X.shape[0]
        n_val = max(1, int(n * self.val_fraction))
        n_train = n - n_val

        X_tr, y_tr = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        self._ajustar_consecuentes(X_tr, y_tr)

        best_val_loss = float('inf')
        best_c = self.c.copy()
        best_sigma = self.sigma.copy()
        best_conseq = self.conseq.copy()
        wait = 0

        for epoch in range(self.n_epochs):
            lr_cur = self.lr * (1 - epoch / self.n_epochs)
            train_loss = 0.0

            idx = np.random.permutation(n_train)
            bs = min(self.batch_size, n_train)
            batch_idx = np.random.choice(n_train, bs, replace=False)
            for s in range(bs):
                i = batch_idx[s]
                xi = X_tr[i]
                if np.random.random() < 0.4:
                    xi = xi + np.random.randn(2) * self.noise_std
                out, w_norm = self._forward(xi)
                train_loss += np.sum((out - y_tr[i]) ** 2)
                self._backward(xi, y_tr[i], out, w_norm, lr_cur)

            if (epoch + 1) % 20 == 0:
                self._ajustar_consecuentes(X_tr, y_tr)

            self.loss_history.append(train_loss / bs)

            n_vs = min(500, n_val)
            val_loss = 0.0
            for i in range(n_vs):
                out, _ = self._forward(X_val[i])
                val_loss += np.sum((out - y_val[i]) ** 2)
            val_loss /= n_val
            self.val_loss_history.append(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_c = self.c.copy()
                best_sigma = self.sigma.copy()
                best_conseq = self.conseq.copy()
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    self.c = best_c
                    self.sigma = best_sigma
                    self.conseq = best_conseq
                    self.parado_temprano = True
                    break

        return self

    def predecir(self, X):
        preds = np.empty((X.shape[0], self.n_outputs))
        for i in range(X.shape[0]):
            out, _ = self._forward(X[i])
            preds[i] = out
        return preds

    def predecir_trayectoria(self, x0, y0, n_steps=200):
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        x[0], y[0] = x0, y0
        for k in range(n_steps - 1):
            out, _ = self._forward(np.array([x[k], y[k]]))
            x[k + 1] = max(out[0], 0)
            y[k + 1] = max(out[1], 0)
        return x, y
