"""Nonlinear MPC controller for Lotka-Volterra dating app model.

Solo c (eficiencia del matching) es variable de control.
a (crecimiento de perfiles) y d (abandono de usuarios) son parametros
fijos del escenario que el MPC conoce pero no optimiza.
b (tasa de match) es constante global.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DT = 0.1
X0, Y0 = 40.0, 50.0
B_OPT = 0.005488

C_BOUNDS = (0.002, 0.04)
C_REF = 0.017875

# Parametros estocasticos default
SIGMA_X = 0.5
SIGMA_Y = 0.3
PULSE_RATE = 0.001
PULSE_SCALE_X = 15.0
PULSE_SCALE_Y = 5.0


# ---------------------------------------------------------------------------
# Simulacion determinista
# ---------------------------------------------------------------------------
def simulate_step(x, y, a, b, c, d):
    x_next = max(x + DT * (a * x - b * x * y), 0)
    y_next = max(y + DT * (c * x * y - d * y), 0)
    return x_next, y_next


def simulate_trajectory(x0, y0, a, d, c_seq, b_fixed):
    N = len(c_seq)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    x[0], y[0] = x0, y0
    for k in range(N):
        x[k + 1], y[k + 1] = simulate_step(
            x[k], y[k], a, b_fixed, c_seq[k], d
        )
    return x, y


# ---------------------------------------------------------------------------
# Simulacion estocastica (ruido blanco + pulsos Poisson)
# ---------------------------------------------------------------------------
def simulate_step_stochastic(x, y, a, b, c, d, rng,
                              sigma_x=0.5, sigma_y=0.3,
                              pulse_rate=0.001,
                              pulse_scale_x=15.0, pulse_scale_y=5.0):
    ruido_x = sigma_x * rng.normal()
    ruido_y = sigma_y * rng.normal()

    pulso_x = rng.exponential(pulse_scale_x) if rng.random() < pulse_rate else 0.0
    pulso_y = rng.exponential(pulse_scale_y) if rng.random() < pulse_rate else 0.0

    x_next = max(x + DT * (a * x - b * x * y) + ruido_x + pulso_x, 0)
    y_next = max(y + DT * (c * x * y - d * y) + ruido_y + pulso_y, 0)

    return x_next, y_next, ruido_x, ruido_y, pulso_x, pulso_y


# ---------------------------------------------------------------------------
# Trayectorias dinamicas (Markov) para el Escenario 3
# ---------------------------------------------------------------------------
def generar_trayectoria_dinamica(n_steps, seed=42):
    """Proceso Markoviano de 3 modos para a y d.

    Modos:
      0 -> Crecimiento  (a=0.6, d=0.15)
      1 -> Estable       (a=0.3, d=0.4)
      2 -> Crisis        (a=0.08, d=0.7)

    Transiciones abruptas cada ~20-60 pasos con probabilidad.
    """
    rng = np.random.default_rng(seed)

    modos_a = np.array([0.6, 0.3, 0.08])
    modos_d = np.array([0.15, 0.4, 0.7])
    # Matriz de transicion: alta probabilidad de quedarse, baja de cambiar
    # Cada fila: [quedarse, cambiar a otro modo]
    P = np.array([
        [0.97, 0.015, 0.015],
        [0.015, 0.97, 0.015],
        [0.015, 0.015, 0.97],
    ])

    a_seq = np.zeros(n_steps)
    d_seq = np.zeros(n_steps)

    modo = rng.integers(0, 3)
    for k in range(n_steps):
        a_seq[k] = modos_a[modo] + rng.uniform(-0.05, 0.05)
        d_seq[k] = modos_d[modo] + rng.uniform(-0.05, 0.05)
        a_seq[k] = np.clip(a_seq[k], 0.05, 0.8)
        d_seq[k] = np.clip(d_seq[k], 0.08, 0.75)

        if rng.random() < 0.03:
            modo = rng.choice(3, p=P[modo])

    return a_seq, d_seq


# ---------------------------------------------------------------------------
# Controlador MPC
# ---------------------------------------------------------------------------
class MPCController:
    def __init__(self, b_fixed=B_OPT, N=15, M=10,
                 q_x=1.0, q_y=1.0, r_c=0.1):
        self.b = b_fixed
        self.N = N
        self.M = M
        self.q_x = q_x
        self.q_y = q_y
        self.r_c = r_c

        self.bounds = [C_BOUNDS] * M
        self.c_prev = None

    def steady_state_u(self, x_ref, y_ref):
        return C_REF

    def _cost(self, c_flat, x0, y0, x_ref, y_ref, c_ref, a_fijo, d_fijo):
        c_seq = np.concatenate([c_flat, [c_flat[-1]] * (self.N - self.M)])

        x, y = simulate_trajectory(x0, y0, a_fijo, d_fijo, c_seq, self.b)

        J = 0.0
        for k in range(self.N):
            J += self.q_x * (x[k + 1] - x_ref) ** 2
            J += self.q_y * (y[k + 1] - y_ref) ** 2

        for k in range(self.N):
            if k < self.M:
                dc = c_seq[k] - c_ref
            else:
                dc = c_seq[-1] - c_ref
            J += self.r_c * dc ** 2

        return J

    def solve(self, x0, y0, x_ref, y_ref, a_fijo, d_fijo):
        c_ref = self.steady_state_u(x_ref, y_ref)

        if self.c_prev is not None:
            c0 = np.concatenate([self.c_prev[1:], [self.c_prev[-1]]])
        else:
            c0 = np.full(self.M, c_ref)

        res = minimize(
            self._cost, c0,
            args=(x0, y0, x_ref, y_ref, c_ref, a_fijo, d_fijo),
            method='SLSQP',
            bounds=self.bounds,
            options={'maxiter': 300, 'ftol': 1e-8, 'disp': False}
        )

        c_opt = res.x
        self.c_prev = c_opt
        return c_opt[0], res.success, res.fun

    def run_simulation(self, n_steps, x_ref, y_ref, a_fijo, d_fijo,
                       x0=X0, y0=Y0, disturbance=None,
                       stochastic=True, seed=42):
        self.c_prev = None

        x_traj = np.zeros(n_steps + 1)
        y_traj = np.zeros(n_steps + 1)
        c_traj = np.zeros(n_steps)
        cost_seq = np.zeros(n_steps)
        success_seq = np.zeros(n_steps, dtype=bool)
        rx_t = np.zeros(n_steps)
        ry_t = np.zeros(n_steps)
        px_t = np.zeros(n_steps)
        py_t = np.zeros(n_steps)

        x_traj[0], y_traj[0] = x0, y0
        rng = np.random.default_rng(seed) if stochastic else None

        for k in range(n_steps):
            x_cur, y_cur = x_traj[k], y_traj[k]

            if disturbance is not None and k == disturbance['step']:
                x_cur += disturbance.get('dx', 0.0)
                y_cur += disturbance.get('dy', 0.0)

            c_opt, success, cost_val = self.solve(
                x_cur, y_cur, x_ref, y_ref, a_fijo, d_fijo
            )

            c_traj[k] = c_opt
            cost_seq[k] = cost_val
            success_seq[k] = success

            if stochastic:
                x_traj[k+1], y_traj[k+1], rx_t[k], ry_t[k], px_t[k], py_t[k] = \
                    simulate_step_stochastic(
                        x_cur, y_cur, a_fijo, self.b, c_traj[k], d_fijo,
                        rng, SIGMA_X, SIGMA_Y, PULSE_RATE,
                        PULSE_SCALE_X, PULSE_SCALE_Y
                    )
            else:
                x_traj[k + 1], y_traj[k + 1] = simulate_step(
                    x_cur, y_cur, a_fijo, self.b, c_traj[k], d_fijo
                )

        result = {
            'x': x_traj, 'y': y_traj,
            'c': c_traj,
            'cost': cost_seq, 'success': success_seq,
            'n_steps': n_steps,
            'x_ref': x_ref, 'y_ref': y_ref,
            'a_fijo': a_fijo, 'd_fijo': d_fijo,
            'ruido_x': rx_t, 'ruido_y': ry_t,
            'pulso_x': px_t, 'pulso_y': py_t,
        }
        return result

    def plot_results(self, results, title='', save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        n = results['n_steps']
        t = np.arange(n + 1) * DT
        tc = np.arange(n) * DT
        x_ref = results['x_ref']
        y_ref = results['y_ref']

        ax = axes[0, 0]
        ax.plot(t, results['x'], 'b-', lw=1.5, label='$x(t)$: perfiles disponibles')
        ax.plot(t, results['y'], 'r-', lw=1.5, label='$y(t)$: usuarios activos')
        ax.axhline(x_ref, color='b', ls='--', alpha=0.5,
                   label=f'$x_{{ref}}={x_ref}$')
        ax.axhline(y_ref, color='r', ls='--', alpha=0.5,
                   label=f'$y_{{ref}}={y_ref}$')
        ax.set_xlabel('Tiempo')
        ax.set_ylabel('Estado')
        ax.set_title('Evolucion de estados')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax2 = axes[0, 1]
        ax2.plot(results['x'], results['y'], 'g-', lw=1.5)
        ax2.plot(results['x'][0], results['y'][0], 'go', label='Inicio',
                 markersize=8)
        ax2.plot(results['x'][-1], results['y'][-1], 'rs', label='Final',
                 markersize=8)
        ax2.plot(x_ref, y_ref, 'k*', label='Referencia', markersize=12)
        ax2.set_xlabel('$x$: perfiles disponibles')
        ax2.set_ylabel('$y$: usuarios activos')
        ax2.set_title('Diagrama de fase')
        ax2.legend()
        ax2.grid(alpha=0.3)

        ax3 = axes[1, 0]
        ax3.plot(tc, results['c'], 'g-', lw=1.5,
                 label='$c$: eficiencia matching (control)')
        ax3.axhline(C_BOUNDS[0], color='g', ls=':', alpha=0.3)
        ax3.axhline(C_BOUNDS[1], color='g', ls=':', alpha=0.3)
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Control')
        ax3.set_title('Senal de control $c(t)$ (lineas = cotas)')
        ax3.legend(fontsize=8)
        ax3.grid(alpha=0.3)

        ax4 = axes[1, 1]
        ax4.plot(tc, results['cost'], 'k-', lw=1, label='Costo MPC')
        ax4.set_xlabel('Tiempo')
        ax4.set_ylabel('Costo')
        ax4.set_title('Costo de optimizacion')
        ax4.grid(alpha=0.3)

        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  Grafica guardada: {save_path}")

        plt.close()
        return fig


# ---------------------------------------------------------------------------
# Funcion comparativa: sin control vs con MPC (un mismo plot 2 paneles)
# ---------------------------------------------------------------------------
def grafica_comparativa(n_steps, x_no_ctrl, y_no_ctrl,
                         x_ctrl, y_ctrl, c_trayectoria,
                         x_ref, y_ref, x_eq, y_eq,
                         a_fijo, d_fijo,
                         titulo, nombre_archivo):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    t = np.arange(n_steps + 1) * DT
    tc = np.arange(n_steps) * DT

    ax1.plot(t, x_no_ctrl, 'b-', lw=0.8, alpha=0.5,
             label='$x(t)$ perfiles (sin control)')
    ax1.plot(t, y_no_ctrl, 'r-', lw=0.8, alpha=0.5,
             label='$y(t)$ usuarios (sin control)')
    ax1.plot(t, x_ctrl, 'b-', lw=1.5,
             label='$x(t)$ perfiles (con MPC)')
    ax1.plot(t, y_ctrl, 'r-', lw=1.5,
             label='$y(t)$ usuarios (con MPC)')
    ax1.axhline(x_ref, color='b', ls='--', alpha=0.4,
                label=f'$x_{{ref}}={x_ref}$')
    ax1.axhline(y_ref, color='r', ls='--', alpha=0.4,
                label=f'$y_{{ref}}={y_ref}$')
    ax1.axhline(x_eq, color='b', ls=':', alpha=0.2,
                label=f'$x^*$={x_eq:.1f}')
    ax1.axhline(y_eq, color='r', ls=':', alpha=0.2,
                label=f'$y^*$={y_eq:.1f}')
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Estado')
    ax1.set_title(titulo, fontsize=11, fontweight='bold')
    ax1.legend(fontsize=7, loc='upper left')
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(tc, c_trayectoria, 'g-', lw=1.0, alpha=0.2,
             label='$c(t)$ eficiencia matching (MPC)')
    ax2.axhline(C_BOUNDS[0], color='g', ls=':', alpha=0.2,
                label=f'cota inf={C_BOUNDS[0]}')
    ax2.axhline(C_BOUNDS[1], color='g', ls=':', alpha=0.2,
                label=f'cota sup={C_BOUNDS[1]}')
    ax2.axhline(C_REF, color='g', ls='--', alpha=0.2,
                label=f'$c_{{ref}}={C_REF}$')
    ax2.set_ylabel('Control $c$')
    ax2.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Grafica guardada: {nombre_archivo}")


# ---------------------------------------------------------------------------
# Simulacion sin control (referencia, estocastica)
# ---------------------------------------------------------------------------
def simular_sin_control(x0, y0, a_fijo, d_fijo, n_steps, seed=42):
    """Simula el sistema con a,d fijos y c=C_REF (sin MPC)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    x[0], y[0] = x0, y0
    for k in range(n_steps):
        x[k+1], y[k+1], _, _, _, _ = simulate_step_stochastic(
            x[k], y[k], a_fijo, B_OPT, C_REF, d_fijo,
            rng, SIGMA_X, SIGMA_Y, PULSE_RATE,
            PULSE_SCALE_X, PULSE_SCALE_Y
        )
    return x, y


def simular_sin_control_con_disturbio(x0, y0, a_fijo, d_fijo,
                                       n_steps, dist_step, dx, seed=42):
    rng = np.random.default_rng(seed)
    x = np.zeros(n_steps + 1)
    y = np.zeros(n_steps + 1)
    x[0], y[0] = x0, y0
    for k in range(n_steps):
        xk = x[k] + (dx if k == dist_step else 0.0)
        yk = y[k]
        x[k+1], y[k+1], _, _, _, _ = simulate_step_stochastic(
            xk, yk, a_fijo, B_OPT, C_REF, d_fijo,
            rng, SIGMA_X, SIGMA_Y, PULSE_RATE,
            PULSE_SCALE_X, PULSE_SCALE_Y
        )
    return x, y


# ---------------------------------------------------------------------------
# Escenarios
# ---------------------------------------------------------------------------
N_STEPS = 10000


def escenario_crecimiento(seed=42):
    print("\n" + "=" * 60)
    print("  ESCENARIO 1: CRECIMIENTO")
    print("  a=0.5, d=0.2 | x0=10, y0=10 | ref=(50, 70)")
    print("=" * 60)

    a_fijo, d_fijo = 0.5, 0.2
    x0, y0 = 10.0, 10.0
    x_ref, y_ref = 50.0, 70.0
    eq_x, eq_y = d_fijo / C_REF, a_fijo / B_OPT

    x_nc, y_nc = simular_sin_control(x0, y0, a_fijo, d_fijo, N_STEPS, seed)
    mpc = MPCController()
    res = mpc.run_simulation(N_STEPS, x_ref, y_ref, a_fijo, d_fijo,
                              x0=x0, y0=y0, stochastic=True, seed=seed)

    sr = res['success'].mean() * 100
    print(f"  Tasa de exito solver: {sr:.1f}%")
    print(f"  Sin control: x_final={x_nc[-1]:.2f}, y_final={y_nc[-1]:.2f}")
    print(f"  Con MPC:     x_final={res['x'][-1]:.2f}, y_final={res['y'][-1]:.2f}")
    print(f"  ¿Con MPC x llega a 0? {any(res['x'] <= 0)}")
    print(f"  ¿Con MPC y llega a 0? {any(res['y'] <= 0)}")
    print(f"  ¿Sin control x llega a 0? {any(x_nc <= 0)}")
    print(f"  ¿Sin control y llega a 0? {any(y_nc <= 0)}")

    grafica_comparativa(
        N_STEPS, x_nc, y_nc, res['x'], res['y'], res['c'],
        x_ref, y_ref, eq_x, eq_y, a_fijo, d_fijo,
        "Escenario 1: Crecimiento — Sin control vs MPC",
        "mpc_escenario1.png"
    )
    return res, x_nc, y_nc


def escenario_mantener(seed=43):
    # y* = a/b = 0.1/0.005488 ≈ 18.2 (fijo, independiente de c)
    # x* = d/c_ref = 0.6/0.017875 ≈ 33.6
    print("\n" + "=" * 60)
    print("  ESCENARIO 2A: RETENCION — MANTENER")
    print("  a=0.1, d=0.6 | x0=60, y0=60 | ref=(35, 20)")
    print("  y* = a/b = 18.2 (fijo) — el MPC mantiene y vivo")
    print("=" * 60)

    a_fijo, d_fijo = 0.1, 0.6
    x0, y0 = 60.0, 60.0
    x_ref, y_ref = 35.0, 20.0
    eq_x, eq_y = d_fijo / C_REF, a_fijo / B_OPT

    x_nc, y_nc = simular_sin_control(x0, y0, a_fijo, d_fijo, N_STEPS, seed)
    mpc = MPCController()
    res = mpc.run_simulation(N_STEPS, x_ref, y_ref, a_fijo, d_fijo,
                              x0=x0, y0=y0, stochastic=True, seed=seed)

    sr = res['success'].mean() * 100
    print(f"  Tasa de exito solver: {sr:.1f}%")
    print(f"  Sin control: x_final={x_nc[-1]:.2f}, y_final={y_nc[-1]:.2f}")
    print(f"  Con MPC:     x_final={res['x'][-1]:.2f}, y_final={res['y'][-1]:.2f}")
    print(f"  ¿Con MPC x llega a 0? {any(res['x'] <= 0)}")
    print(f"  ¿Con MPC y llega a 0? {any(res['y'] <= 0)}")
    print(f"  ¿Sin control x llega a 0? {any(x_nc <= 0)}")
    print(f"  ¿Sin control y llega a 0? {any(y_nc <= 0)}")

    grafica_comparativa(
        N_STEPS, x_nc, y_nc, res['x'], res['y'], res['c'],
        x_ref, y_ref, eq_x, eq_y, a_fijo, d_fijo,
        "Escenario 2a: Retencion — Mantener (Sin control vs MPC)",
        "mpc_escenario2a.png"
    )
    return res, x_nc, y_nc


def escenario_crecer(seed=44):
    # y* = a/b = 18.2 fijo. Con c menor, x* sube.
    # Para x_ref=60: c = d / x_ref = 0.6/60 = 0.01 (dentro de cotas)
    print("\n" + "=" * 60)
    print("  ESCENARIO 2B: RETENCION — CRECER (en x)")
    print("  a=0.1, d=0.6 | x0=60, y0=60 | ref=(60, 25)")
    print("  c se reduce para elevar x* = d/c, y se mantiene via transitorios")
    print("=" * 60)

    a_fijo, d_fijo = 0.1, 0.6
    x0, y0 = 60.0, 60.0
    x_ref, y_ref = 60.0, 25.0
    eq_x, eq_y = d_fijo / C_REF, a_fijo / B_OPT

    x_nc, y_nc = simular_sin_control(x0, y0, a_fijo, d_fijo, N_STEPS, seed)
    mpc = MPCController()
    res = mpc.run_simulation(N_STEPS, x_ref, y_ref, a_fijo, d_fijo,
                              x0=x0, y0=y0, stochastic=True, seed=seed)

    sr = res['success'].mean() * 100
    print(f"  Tasa de exito solver: {sr:.1f}%")
    print(f"  Sin control: x_final={x_nc[-1]:.2f}, y_final={y_nc[-1]:.2f}")
    print(f"  Con MPC:     x_final={res['x'][-1]:.2f}, y_final={res['y'][-1]:.2f}")
    print(f"  ¿Con MPC x llega a 0? {any(res['x'] <= 0)}")
    print(f"  ¿Con MPC y llega a 0? {any(res['y'] <= 0)}")
    print(f"  ¿Sin control x llega a 0? {any(x_nc <= 0)}")
    print(f"  ¿Sin control y llega a 0? {any(y_nc <= 0)}")

    grafica_comparativa(
        N_STEPS, x_nc, y_nc, res['x'], res['y'], res['c'],
        x_ref, y_ref, eq_x, eq_y, a_fijo, d_fijo,
        "Escenario 2b: Retencion — Crecer en x (Sin control vs MPC)",
        "mpc_escenario2b.png"
    )
    return res, x_nc, y_nc


def escenario_dinamico(seed=45):
    print("\n" + "=" * 60)
    print("  ESCENARIO 3: DINAMICO (a(t), d(t) markovianos)")
    print("  x0=30, y0=30 | ref=(40, 40)")
    print("=" * 60)

    x0, y0 = 30.0, 30.0
    x_ref, y_ref = 40.0, 40.0
    eq_x, eq_y = 39.2, 53.0

    a_seq, d_seq = generar_trayectoria_dinamica(N_STEPS, seed=seed)

    # Sin control: c=C_REF fijo, a(t), d(t) variables
    rng_nc = np.random.default_rng(seed)
    x_nc = np.zeros(N_STEPS + 1)
    y_nc = np.zeros(N_STEPS + 1)
    x_nc[0], y_nc[0] = x0, y0
    for k in range(N_STEPS):
        x_nc[k+1], y_nc[k+1], _, _, _, _ = simulate_step_stochastic(
            x_nc[k], y_nc[k], a_seq[k], B_OPT, C_REF, d_seq[k],
            rng_nc, SIGMA_X, SIGMA_Y, PULSE_RATE,
            PULSE_SCALE_X, PULSE_SCALE_Y
        )

    # Con MPC: en cada paso el MPC recibe a(k), d(k) actuales
    mpc = MPCController()
    mpc.c_prev = None
    x_ctrl = np.zeros(N_STEPS + 1)
    y_ctrl = np.zeros(N_STEPS + 1)
    c_traj = np.zeros(N_STEPS)
    x_ctrl[0], y_ctrl[0] = x0, y0
    rng_mpc = np.random.default_rng(seed)

    for k in range(N_STEPS):
        x_cur, y_cur = x_ctrl[k], y_ctrl[k]
        a_cur, d_cur = a_seq[k], d_seq[k]
        c_opt, success, _ = mpc.solve(x_cur, y_cur, x_ref, y_ref, a_cur, d_cur)
        c_traj[k] = c_opt
        x_ctrl[k+1], y_ctrl[k+1], _, _, _, _ = \
            simulate_step_stochastic(
                x_cur, y_cur, a_cur, mpc.b, c_opt, d_cur,
                rng_mpc, SIGMA_X, SIGMA_Y, PULSE_RATE,
                PULSE_SCALE_X, PULSE_SCALE_Y
            )

    print(f"  Sin control: x_final={x_nc[-1]:.2f}, y_final={y_nc[-1]:.2f}")
    print(f"  Con MPC:     x_final={x_ctrl[-1]:.2f}, y_final={y_ctrl[-1]:.2f}")
    print(f"  ¿Con MPC x llega a 0? {any(x_ctrl <= 0)}")
    print(f"  ¿Con MPC y llega a 0? {any(y_ctrl <= 0)}")
    print(f"  ¿Sin control x llega a 0? {any(x_nc <= 0)}")
    print(f"  ¿Sin control y llega a 0? {any(y_nc <= 0)}")

    grafica_comparativa(
        N_STEPS, x_nc, y_nc, x_ctrl, y_ctrl, c_traj,
        x_ref, y_ref, eq_x, eq_y, np.nan, np.nan,
        "Escenario 3: Dinamico — a(t), d(t) variables (Sin control vs MPC)",
        "mpc_escenario3.png"
    )
    return (x_ctrl, y_ctrl, c_traj), (x_nc, y_nc), (a_seq, d_seq)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    r1, _, _ = escenario_crecimiento()
    r2a, _, _ = escenario_mantener()
    r2b, _, _ = escenario_crecer()
    r3_data, _, _ = escenario_dinamico()

    print("\n" + "=" * 60)
    print("  RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"  Esc.1 Crecimiento:    x_final={r1['x'][-1]:.2f}, y_final={r1['y'][-1]:.2f}")
    print(f"  Esc.2a Mantener:      x_final={r2a['x'][-1]:.2f}, y_final={r2a['y'][-1]:.2f}")
    print(f"  Esc.2b Crecer:        x_final={r2b['x'][-1]:.2f}, y_final={r2b['y'][-1]:.2f}")
    print(f"  Esc.3 Dinamico:       x_final={r3_data[0][-1]:.2f}, y_final={r3_data[1][-1]:.2f}")
    print()
    print("  Graficas generadas:")
    print("    - mpc_escenario1.png")
    print("    - mpc_escenario2a.png")
    print("    - mpc_escenario2b.png")
    print("    - mpc_escenario3.png")
