import numpy as np

DT = 0.1
X0, Y0 = 40, 50

RNG = np.random.default_rng(42)


def simulate(a, b, c, d, n_steps=800, x0=X0, y0=Y0):
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    x[0], y[0] = x0, y0
    for k in range(n_steps - 1):
        x[k + 1] = max(x[k] + DT * (a * x[k] - b * x[k] * y[k]), 0)
        y[k + 1] = max(y[k] + DT * (c * x[k] * y[k] - d * y[k]), 0)
    return x, y


def generar_datos(n_secuencias=20, n_steps=200, variar_params=True):
    X_list, y_list = [], []
    params_usados = []

    for i in range(n_secuencias):
        np.random.seed(42 + i)
        if variar_params:
            a = np.random.uniform(0.2, 0.5)
            b = np.random.uniform(0.005, 0.02)
            c = np.random.uniform(0.005, 0.03)
            d = np.random.uniform(0.3, 0.7)
        else:
            a, b, c, d = 0.3, 0.006, 0.018, 0.7

        x0 = np.random.uniform(20, 80)
        y0 = np.random.uniform(20, 80)

        x, y = simulate(a, b, c, d, n_steps, x0, y0)
        params_usados.append((a, b, c, d, x0, y0))

        for k in range(n_steps - 1):
            X_list.append([x[k], y[k]])
            y_list.append([x[k + 1], y[k + 1]])

    return np.array(X_list), np.array(y_list), params_usados
