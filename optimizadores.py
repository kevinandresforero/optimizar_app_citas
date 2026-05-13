"""Three optimizer implementations for the Dating App Lotka-Volterra model.

Classes:
    DifferentialEvolutionOptimizer — global search via Differential Evolution + L-BFGS-B.
    SGDOptimizer                 — stochastic gradient descent with momentum and restarts.
    ANFISOptimizer               — neuro-fuzzy optimizer (simplified ANFIS).

All share the same interface:
    __init__(a, b, c, d) -> set initial/default parameters.
    optimize(n_steps)     -> run optimization, store results in self.
    get_params()          -> return (a, b, c, d).
    get_results()         -> dict of all metrics.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize

# Shared simulation engine (same as DatingAppOptimizer)
DT = 0.1
X0, Y0 = 40, 50

ALPHA_BOUNDS = (0.2, 1.0)
BETA_BOUNDS = (0.005, 0.04)
DELTA_BOUNDS = (0.002, 0.04)
GAMMA_BOUNDS = (0.1, 0.7)

PENALTY = 1e6


def simulate(a, b, c, d, n_steps=800, x0=X0, y0=Y0):
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    x[0], y[0] = x0, y0
    for k in range(n_steps - 1):
        x[k + 1] = max(x[k] + DT * (a * x[k] - b * x[k] * y[k]), 0)
        y[k + 1] = max(y[k] + DT * (c * x[k] * y[k] - d * y[k]), 0)
    return x, y


def cost_function(params, n_steps=800, x0=X0, y0=Y0):
    a, b, c, d = params
    x, y = simulate(a, b, c, d, n_steps, x0, y0)
    ys, xs = y[n_steps // 2:], x[n_steps // 2:]
    y_prom = float(np.mean(ys))
    x_prom = float(np.mean(xs))

    if y_prom < 20 or x_prom < 5:
        return PENALTY + (20 - y_prom) * 100 + (5 - x_prom) * 100

    x_eq = d / max(c, 1e-6)
    y_eq = a / max(b, 1e-6)
    proporcion = x_prom / max(y_prom, 1)
    cv = np.std(ys) / max(y_prom, 1)
    retencion = np.exp(-d)
    ganancia = y_prom * 0.5 + y_prom * 0.05 * 9.99 * (c / 0.01) - y_prom * 0.15
    sat = (c / max(d, 0.01)) * 5

    j = ((proporcion - 0.6) ** 2 * 50
         + ((y_prom - 55) / 55) ** 2 * 40
         + cv * 30
         + ((x_eq - x_prom) / max(x_prom, 1)) ** 2 * 10
         + ((y_eq - y_prom) / max(y_prom, 1)) ** 2 * 10
         - retencion * 20
         - ganancia * 0.3
         - min(sat, 10) * 2)
    return j


def compute_metrics(params, n_steps=800, x0=X0, y0=Y0):
    a, b, c, d = params
    x, y = simulate(a, b, c, d, n_steps, x0, y0)
    ys, xs = y[n_steps // 2:], x[n_steps // 2:]
    y_prom = float(np.mean(ys))
    x_prom = float(np.mean(xs))
    cv = float(np.std(ys) / max(y_prom, 1))
    ret = float(np.exp(-d))
    gan = float(y_prom * 0.5 + y_prom * 0.05 * 9.99 * (c / 0.01) - y_prom * 0.15)
    x_eq = d / max(c, 1e-6)
    y_eq = a / max(b, 1e-6)
    osc = sum(1 for i in range(1, len(ys) - 1)
              if (ys[i - 1] < ys[i] > ys[i + 1]) or (ys[i - 1] > ys[i] < ys[i + 1]))
    return {
        'a': a, 'b': b, 'c': c, 'd': d,
        'x_eq': x_eq, 'y_eq': y_eq,
        'x_prom': x_prom, 'y_prom': y_prom,
        'cv': cv, 'retencion': ret, 'ganancia': gan,
        'oscilaciones': osc, 'cost': cost_function(params, n_steps, x0, y0),
    }


# =====================================================================
# Optimizer 1: Differential Evolution + L-BFGS-B (reference)
# =====================================================================

class DifferentialEvolutionOptimizer:
    """Differential Evolution + L-BFGS-B (reference optimizer)."""

    def __init__(self, a=0.3, b=0.006, c=0.018, d=0.7):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.x_sim = self.y_sim = None
        self.history = []

    def optimize(self, n_steps=800, x0=X0, y0=Y0):
        bounds = [ALPHA_BOUNDS, BETA_BOUNDS, DELTA_BOUNDS, GAMMA_BOUNDS]
        res = differential_evolution(
            lambda p: cost_function(p, n_steps, x0, y0), bounds,
            strategy='best1bin', maxiter=1500, popsize=50,
            tol=1e-8, seed=42,
        )
        res2 = minimize(lambda p: cost_function(p, n_steps, x0, y0),
                        res.x, bounds=bounds, method='L-BFGS-B')
        xopt = res2.x if res2.fun < res.fun else res.x
        self.a, self.b, self.c, self.d = xopt
        self.x_sim, self.y_sim = simulate(self.a, self.b, self.c, self.d, n_steps, x0, y0)
        self.history = [('final', cost_function(xopt, n_steps, x0, y0))]
        return self

    def get_params(self):
        return self.a, self.b, self.c, self.d

    def get_results(self, n_steps=800, x0=X0, y0=Y0):
        return compute_metrics(self.get_params(), n_steps, x0, y0)


# =====================================================================
# Optimizer 2: Stochastic Gradient Descent with momentum + restarts
# =====================================================================

class SGDOptimizer:
    """Stochastic gradient descent with momentum + grid-based restarts.

    Uses finite-difference gradient approximation with gradient clipping,
    momentum, adaptive learning rate, and a grid+random restart strategy
    to avoid the penalty boundaries and find competitive solutions.
    """

    def __init__(self, a=0.3, b=0.006, c=0.018, d=0.7):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.x_sim = self.y_sim = None
        self.history = []

    def _clip(self, p):
        return np.clip(p,
                       [ALPHA_BOUNDS[0], BETA_BOUNDS[0],
                        DELTA_BOUNDS[0], GAMMA_BOUNDS[0]],
                       [ALPHA_BOUNDS[1], BETA_BOUNDS[1],
                        DELTA_BOUNDS[1], GAMMA_BOUNDS[1]])

    def _finite_grad(self, p, n_steps, x0, y0, eps=1e-4):
        base = cost_function(p, n_steps, x0, y0)
        grad = np.zeros(4)
        for i in range(4):
            p_up = p.copy()
            p_up[i] += eps
            p_up = self._clip(p_up)
            grad[i] = (cost_function(p_up, n_steps, x0, y0) - base) / eps
        # Clip gradient to avoid explosive updates near penalty boundaries
        norm = np.linalg.norm(grad)
        if norm > 1000:
            grad = grad / norm * 1000
        return grad, base

    def _restart_from(self, p0, n_steps, x0, y0, max_iter=150, lr=0.005):
        p = p0.copy()
        v = np.zeros(4)
        best_p = p.copy()
        best_c = cost_function(p, n_steps, x0, y0)
        lr_cur = lr

        for step in range(max_iter):
            grad, cur_cost = self._finite_grad(p, n_steps, x0, y0)
            # Momentum + gradient descent
            v = 0.85 * v + lr_cur * grad
            p = self._clip(p - v)
            lr_cur *= 0.99

            cur_cost = cost_function(p, n_steps, x0, y0)
            if cur_cost < best_c:
                best_c = cur_cost
                best_p = p.copy()

        return best_p, best_c

    def optimize(self, n_steps=800, x0=X0, y0=Y0):
        rng = np.random.default_rng(42)

        # Grid of diverse starting points
        alphas = np.linspace(ALPHA_BOUNDS[0], ALPHA_BOUNDS[1], 4)
        betas = np.linspace(BETA_BOUNDS[0], BETA_BOUNDS[1], 3)
        deltas = np.linspace(DELTA_BOUNDS[0], DELTA_BOUNDS[1], 3)
        gammas = np.linspace(GAMMA_BOUNDS[0], GAMMA_BOUNDS[1], 3)

        starts = [(self.a, self.b, self.c, self.d)]
        for a in alphas:
            for b in betas:
                for c in deltas:
                    for d in gammas:
                        if rng.random() < 0.3:
                            starts.append((a, b, c, d))
        for _ in range(20):
            starts.append((
                rng.uniform(*ALPHA_BOUNDS),
                rng.uniform(*BETA_BOUNDS),
                rng.uniform(*DELTA_BOUNDS),
                rng.uniform(*GAMMA_BOUNDS),
            ))

        best_p = None
        best_c = float('inf')

        for i, p0 in enumerate(starts):
            p, c = self._restart_from(np.array(p0), n_steps, x0, y0)
            self.history.append((f'start_{i}', c, tuple(p)))
            if c < best_c:
                best_c = c
                best_p = p

        self.a, self.b, self.c, self.d = best_p
        self.x_sim, self.y_sim = simulate(*best_p, n_steps, x0, y0)
        return self

    def get_params(self):
        return self.a, self.b, self.c, self.d

    def get_results(self, n_steps=800, x0=X0, y0=Y0):
        return compute_metrics(self.get_params(), n_steps, x0, y0)


# =====================================================================
# Optimizer 3: ANFIS-style (Neuro-Fuzzy)
# =====================================================================

class ANFISOptimizer:
    """Neuro-Fuzzy optimizer (simplified ANFIS).

    Defines Gaussian membership functions (Low, Medium, High) for each
    parameter. A small feedforward NN learns the optimal MF centres and
    widths by sampling parameter combinations and backpropagating the
    cost signal. The inference process maps cost components to parameter
    adjustments through fuzzy rules.
    """

    def __init__(self, a=0.3, b=0.006, c=0.018, d=0.7):
        self.a, self.b, self.c, self.d = a, b, c, d
        self.x_sim = self.y_sim = None
        self.history = []

        n_mf = 3
        self.n_mf = n_mf

        self.mu = np.array([
            [0.3, 0.6, 0.9],
            [0.01, 0.02, 0.035],
            [0.008, 0.02, 0.035],
            [0.2, 0.4, 0.6],
        ])
        self.sigma = np.full((4, n_mf), 0.15)

        np.random.seed(42)
        n_input = 4 * n_mf
        self.W1 = np.random.randn(n_input, 16) * 0.1
        self.b1 = np.zeros(16)
        self.W2 = np.random.randn(16, 4) * 0.1
        self.b2 = np.zeros(4)

    def _fuzzify(self, params):
        a, b, c, d = params
        x = np.array([a, b, c, d])
        phi = []
        for i in range(4):
            for j in range(self.n_mf):
                val = np.exp(-((x[i] - self.mu[i, j]) ** 2) / (2 * self.sigma[i, j] ** 2))
                phi.append(val)
        return np.array(phi)

    def _nn_predict(self, phi):
        h = np.tanh(self.W1.T @ phi + self.b1)
        out = self.W2.T @ h + self.b2
        return np.tanh(out) * 0.1

    def _nn_train(self, phi, target, lr=0.01):
        h = np.tanh(self.W1.T @ phi + self.b1)
        out = self.W2.T @ h + self.b2
        pred = np.tanh(out) * 0.1
        error = pred - target
        d_out = error * (1 - np.tanh(out) ** 2) * 0.1
        d_W2 = np.outer(h, d_out)
        d_b2 = d_out
        d_h = self.W2 @ d_out * (1 - h ** 2)
        d_W1 = np.outer(phi, d_h)
        d_b1 = d_h
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1

    def _infer_adjustment(self, current_cost, prev_cost_):
        delta = current_cost - prev_cost_
        scale = max(abs(current_cost), 1.0)
        if delta > 0:
            d_a = -0.02 * (current_cost / scale)
            d_b = 0.001 * (current_cost / scale)
            d_c = -0.002 * (current_cost / scale)
            d_d = -0.03 * (current_cost / scale)
        else:
            d_a = 0.01 * (1 - current_cost / scale)
            d_b = -0.0005 * (1 - current_cost / scale)
            d_c = 0.001 * (1 - current_cost / scale)
            d_d = 0.02 * (1 - current_cost / scale)
        return d_a, d_b, d_c, d_d

    def optimize(self, n_steps=800, x0=X0, y0=Y0,
                 n_epochs=50, n_samples=20, nn_lr=0.05):
        best_cost = float('inf')
        best_params = np.array([self.a, self.b, self.c, self.d])
        params = best_params.copy()
        cur_cost = cost_function(params, n_steps, x0, y0)
        self.history.append((0, cur_cost, tuple(params)))

        # Pre-populate with random candidates for diversity
        rng = np.random.default_rng(42)
        for _ in range(30):
            rp = np.array([
                rng.uniform(*ALPHA_BOUNDS),
                rng.uniform(*BETA_BOUNDS),
                rng.uniform(*DELTA_BOUNDS),
                rng.uniform(*GAMMA_BOUNDS),
            ])
            rc = cost_function(rp, n_steps, x0, y0)
            if rc < best_cost:
                best_cost = rc
                best_params = rp.copy()
                params = rp.copy()
                cur_cost = rc

        for epoch in range(n_epochs):
            lr = nn_lr * (1 - epoch / n_epochs)
            for _ in range(n_samples):
                phi = self._fuzzify(params)
                nn_delta = self._nn_predict(phi)
                fuzzy_da, fuzzy_db, fuzzy_dc, fuzzy_dd = \
                    self._infer_adjustment(cur_cost, cur_cost)

                da = nn_delta[0] + fuzzy_da * 0.5
                db = nn_delta[1] + fuzzy_db * 0.5
                dc = nn_delta[2] + fuzzy_dc * 0.5
                dd = nn_delta[3] + fuzzy_dd * 0.5

                new_params = np.clip(
                    [params[0] + da, params[1] + db,
                     params[2] + dc, params[3] + dd],
                    [ALPHA_BOUNDS[0], BETA_BOUNDS[0],
                     DELTA_BOUNDS[0], GAMMA_BOUNDS[0]],
                    [ALPHA_BOUNDS[1], BETA_BOUNDS[1],
                     DELTA_BOUNDS[1], GAMMA_BOUNDS[1]],
                )
                new_cost = cost_function(new_params, n_steps, x0, y0)

                phi_new = self._fuzzify(new_params)
                target_delta = np.array([
                    new_params[0] - params[0],
                    new_params[1] - params[1],
                    new_params[2] - params[2],
                    new_params[3] - params[3],
                ])
                self._nn_train(phi_new, target_delta, lr)

                if new_cost < cur_cost or np.random.random() < 0.2:
                    params = new_params
                    cur_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_params = params.copy()

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                self.history.append((epoch + 1, best_cost, tuple(best_params)))

        self.a, self.b, self.c, self.d = best_params
        self.x_sim, self.y_sim = simulate(*best_params, n_steps, x0, y0)
        return self

    def get_params(self):
        return self.a, self.b, self.c, self.d

    def get_results(self, n_steps=800, x0=X0, y0=Y0):
        return compute_metrics(self.get_params(), n_steps, x0, y0)
