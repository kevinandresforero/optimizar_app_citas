"""Nonlinear MPC controller for Lotka-Volterra dating app model.

Controla perfiles (x) y usuarios (y) manipulando:
  - a: tasa de crecimiento de perfiles (marketing)
  - c: eficiencia del algoritmo de matching
  - d: tasa de abandono de usuarios (retencion)

Parametro fijo: b (tasa de match)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

DT = 0.1
X0, Y0 = 40.0, 50.0
B_OPT = 0.005488

A_BOUNDS = (0.2, 1.0)
C_BOUNDS = (0.002, 0.04)
D_BOUNDS = (0.1, 0.7)

C_REF = 0.017875


def simulate_step(x, y, a, b, c, d):
    x_next = max(x + DT * (a * x - b * x * y), 0)
    y_next = max(y + DT * (c * x * y - d * y), 0)
    return x_next, y_next


def simulate_trajectory(x0, y0, a_seq, c_seq, d_seq, b_fixed):
    N = len(a_seq)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    x[0], y[0] = x0, y0
    for k in range(N):
        x[k + 1], y[k + 1] = simulate_step(
            x[k], y[k], a_seq[k], b_fixed, c_seq[k], d_seq[k]
        )
    return x, y


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


def simulate_stochastic(x0, y0, a_seq, c_seq, d_seq, b_fixed, seed=42,
                        sigma_x=0.5, sigma_y=0.3, pulse_rate=0.001,
                        pulse_scale_x=15.0, pulse_scale_y=5.0):
    N = len(a_seq)
    rng = np.random.default_rng(seed)
    x = np.zeros(N + 1)
    y = np.zeros(N + 1)
    rx_t = np.zeros(N)
    ry_t = np.zeros(N)
    px_t = np.zeros(N)
    py_t = np.zeros(N)
    x[0], y[0] = x0, y0
    for k in range(N):
        x[k+1], y[k+1], rx_t[k], ry_t[k], px_t[k], py_t[k] = \
            simulate_step_stochastic(x[k], y[k], a_seq[k], b_fixed, c_seq[k], d_seq[k],
                                      rng, sigma_x, sigma_y, pulse_rate,
                                      pulse_scale_x, pulse_scale_y)
    return x, y, rx_t, ry_t, px_t, py_t


class MPCController:
    def __init__(self, b_fixed=B_OPT, N=15, M=10,
                 q_x=1.0, q_y=1.0, r_a=0.1, r_c=0.1, r_d=0.1):
        self.b = b_fixed
        self.N = N
        self.M = M
        self.q_x = q_x
        self.q_y = q_y
        self.r_a = r_a
        self.r_c = r_c
        self.r_d = r_d
        self.u_prev = None

        self.bounds = []
        for _ in range(M):
            self.bounds += [A_BOUNDS, C_BOUNDS, D_BOUNDS]

    def steady_state_u(self, x_ref, y_ref):
        a_ref = self.b * y_ref
        c_ref = C_REF
        d_ref = c_ref * x_ref
        d_ref = np.clip(d_ref, D_BOUNDS[0], D_BOUNDS[1])
        return np.array([a_ref, c_ref, d_ref])

    def _cost(self, u_flat, x0, y0, x_ref, y_ref, u_ref):
        u = u_flat.reshape(self.M, 3)
        a_seq = np.concatenate([u[:, 0], [u[-1, 0]] * (self.N - self.M)])
        c_seq = np.concatenate([u[:, 1], [u[-1, 1]] * (self.N - self.M)])
        d_seq = np.concatenate([u[:, 2], [u[-1, 2]] * (self.N - self.M)])

        x, y = simulate_trajectory(x0, y0, a_seq, c_seq, d_seq, self.b)

        J = 0.0
        for k in range(self.N):
            J += self.q_x * (x[k + 1] - x_ref) ** 2
            J += self.q_y * (y[k + 1] - y_ref) ** 2

        for k in range(self.N):
            if k < self.M:
                du = u[k] - u_ref
            else:
                du = u[-1] - u_ref
            J += self.r_a * du[0] ** 2
            J += self.r_c * du[1] ** 2
            J += self.r_d * du[2] ** 2

        return J

    def solve(self, x0, y0, x_ref, y_ref):
        u_ref = self.steady_state_u(x_ref, y_ref)

        if self.u_prev is not None:
            u0 = np.vstack([self.u_prev[1:], self.u_prev[-1]])
            x0_guess = u0.flatten()
        else:
            x0_guess = np.tile(u_ref, self.M)

        res = minimize(
            self._cost, x0_guess,
            args=(x0, y0, x_ref, y_ref, u_ref),
            method='SLSQP',
            bounds=self.bounds,
            options={'maxiter': 300, 'ftol': 1e-8, 'disp': False}
        )

        u_opt = res.x.reshape(self.M, 3)
        self.u_prev = u_opt
        return u_opt[0], res.success, res.fun

    def run_simulation(self, n_steps, x_ref, y_ref, x0=X0, y0=Y0,
                       disturbance=None,
                       stochastic=False, seed=42,
                       sigma_x=0.5, sigma_y=0.3,
                       pulse_rate=0.001, pulse_scale_x=15.0,
                       pulse_scale_y=5.0):
        self.u_prev = None

        x_traj = np.zeros(n_steps + 1)
        y_traj = np.zeros(n_steps + 1)
        a_traj = np.zeros(n_steps)
        c_traj = np.zeros(n_steps)
        d_traj = np.zeros(n_steps)
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

            u_opt, success, cost_val = self.solve(x_cur, y_cur, x_ref, y_ref)

            a_traj[k] = u_opt[0]
            c_traj[k] = u_opt[1]
            d_traj[k] = u_opt[2]
            cost_seq[k] = cost_val
            success_seq[k] = success

            if stochastic:
                x_traj[k+1], y_traj[k+1], rx_t[k], ry_t[k], px_t[k], py_t[k] = \
                    simulate_step_stochastic(
                        x_cur, y_cur, a_traj[k], self.b, c_traj[k], d_traj[k],
                        rng, sigma_x, sigma_y, pulse_rate,
                        pulse_scale_x, pulse_scale_y
                    )
            else:
                x_traj[k + 1], y_traj[k + 1] = simulate_step(
                    x_cur, y_cur, a_traj[k], self.b, c_traj[k], d_traj[k]
                )

        result = {
            'x': x_traj, 'y': y_traj,
            'a': a_traj, 'c': c_traj, 'd': d_traj,
            'cost': cost_seq, 'success': success_seq,
            'n_steps': n_steps,
            'x_ref': x_ref, 'y_ref': y_ref,
        }
        if stochastic:
            result['ruido_x'] = rx_t
            result['ruido_y'] = ry_t
            result['pulso_x'] = px_t
            result['pulso_y'] = py_t
        return result

    def run_simulation_stochastic(self, n_steps, x_ref, y_ref, x0=X0, y0=Y0,
                                   seed=42, sigma_x=0.5, sigma_y=0.3,
                                   pulse_rate=0.001, pulse_scale_x=15.0,
                                   pulse_scale_y=5.0):
        self.u_prev = None

        x_traj = np.zeros(n_steps + 1)
        y_traj = np.zeros(n_steps + 1)
        a_traj = np.zeros(n_steps)
        c_traj = np.zeros(n_steps)
        d_traj = np.zeros(n_steps)
        cost_seq = np.zeros(n_steps)
        success_seq = np.zeros(n_steps, dtype=bool)
        rx_t = np.zeros(n_steps)
        ry_t = np.zeros(n_steps)
        px_t = np.zeros(n_steps)
        py_t = np.zeros(n_steps)

        x_traj[0], y_traj[0] = x0, y0
        rng = np.random.default_rng(seed)

        for k in range(n_steps):
            x_cur, y_cur = x_traj[k], y_traj[k]
            u_opt, success, cost_val = self.solve(x_cur, y_cur, x_ref, y_ref)
            a_traj[k] = u_opt[0]
            c_traj[k] = u_opt[1]
            d_traj[k] = u_opt[2]
            cost_seq[k] = cost_val
            success_seq[k] = success

            x_traj[k+1], y_traj[k+1], rx_t[k], ry_t[k], px_t[k], py_t[k] = \
                simulate_step_stochastic(
                    x_cur, y_cur, a_traj[k], self.b, c_traj[k], d_traj[k],
                    rng, sigma_x, sigma_y, pulse_rate, pulse_scale_x, pulse_scale_y
                )

        return {
            'x': x_traj, 'y': y_traj,
            'a': a_traj, 'c': c_traj, 'd': d_traj,
            'cost': cost_seq, 'success': success_seq,
            'n_steps': n_steps,
            'x_ref': x_ref, 'y_ref': y_ref,
            'ruido_x': rx_t, 'ruido_y': ry_t,
            'pulso_x': px_t, 'pulso_y': py_t,
        }

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
        ax3.plot(tc, results['a'], 'b-', lw=1.5, label='$a$: crecimiento perfiles')
        ax3.plot(tc, results['c'], 'g-', lw=1.5, label='$c$: eficiencia matching')
        ax3.plot(tc, results['d'], 'r-', lw=1.5, label='$d$: abandono usuarios')
        ax3.axhline(A_BOUNDS[0], color='b', ls=':', alpha=0.3)
        ax3.axhline(A_BOUNDS[1], color='b', ls=':', alpha=0.3)
        ax3.axhline(C_BOUNDS[0], color='g', ls=':', alpha=0.3)
        ax3.axhline(C_BOUNDS[1], color='g', ls=':', alpha=0.3)
        ax3.axhline(D_BOUNDS[0], color='r', ls=':', alpha=0.3)
        ax3.axhline(D_BOUNDS[1], color='r', ls=':', alpha=0.3)
        ax3.set_xlabel('Tiempo')
        ax3.set_ylabel('Control')
        ax3.set_title('Senales de control (lineas punteadas = cotas)')
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


def escenario_referencia_constante(stochastic=True, seed=42):
    print("\n" + "=" * 60)
    print("  ESCENARIO 1: SEGUIMIENTO DE REFERENCIA CONSTANTE")
    print("=" * 60)

    mpc = MPCController()
    res = mpc.run_simulation(n_steps=200, x_ref=35.0, y_ref=55.0,
                             stochastic=stochastic, seed=seed)

    sr = res['success'].mean() * 100
    print(f"  Tasa de exito del solver: {sr:.1f}%")
    print(f"  Estado final: x={res['x'][-1]:.2f}, y={res['y'][-1]:.2f}")
    print(f"  Referencia:   x_ref=35.0, y_ref=55.0")

    mpc.plot_results(res, "MPC: Seguimiento de Referencia Constante",
                     save_path="mpc_escenario1.png")
    return res


def escenario_cambio_escalon(stochastic=True, seed=42):
    print("\n" + "=" * 60)
    print("  ESCENARIO 2: CAMBIO ESCALON DE REFERENCIA")
    print("=" * 60)

    mpc = MPCController()
    n_steps = 250
    switch = 100
    xr1, yr1 = 35.0, 55.0
    xr2, yr2 = 30.0, 58.0
    rng = np.random.default_rng(seed) if stochastic else None

    x_traj = np.zeros(n_steps + 1)
    y_traj = np.zeros(n_steps + 1)
    a_traj = np.zeros(n_steps)
    c_traj = np.zeros(n_steps)
    d_traj = np.zeros(n_steps)
    cost_seq = np.zeros(n_steps)
    success_seq = np.zeros(n_steps, dtype=bool)
    x_ref_seq = np.full(n_steps + 1, xr1)
    y_ref_seq = np.full(n_steps + 1, yr1)
    x_ref_seq[switch:] = xr2
    y_ref_seq[switch:] = yr2

    x_traj[0], y_traj[0] = 40.0, 50.0

    for k in range(n_steps):
        x_ref = xr1 if k < switch else xr2
        y_ref = yr1 if k < switch else yr2

        x_cur, y_cur = x_traj[k], y_traj[k]
        u_opt, success, cost_val = mpc.solve(x_cur, y_cur, x_ref, y_ref)

        a_traj[k] = u_opt[0]
        c_traj[k] = u_opt[1]
        d_traj[k] = u_opt[2]
        cost_seq[k] = cost_val
        success_seq[k] = success

        if stochastic:
            x_traj[k+1], y_traj[k+1], _, _, _, _ = simulate_step_stochastic(
                x_cur, y_cur, a_traj[k], mpc.b, c_traj[k], d_traj[k], rng
            )
        else:
            x_traj[k+1], y_traj[k+1] = simulate_step(
                x_cur, y_cur, a_traj[k], mpc.b, c_traj[k], d_traj[k]
            )

    results = {
        'x': x_traj, 'y': y_traj,
        'a': a_traj, 'c': c_traj, 'd': d_traj,
        'cost': cost_seq, 'success': success_seq,
        'n_steps': n_steps,
        'x_ref_seq': x_ref_seq, 'y_ref_seq': y_ref_seq,
        'switch': switch,
    }

    sr = success_seq.mean() * 100
    print(f"  Tasa de exito del solver: {sr:.1f}%")
    print(f"  Cambio de ref. en t={switch * DT}")
    print(f"  Ref 1: x=35.0, y=55.0")
    print(f"  Ref 2: x=30.0, y=58.0")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t = np.arange(n_steps + 1) * DT
    tc = np.arange(n_steps) * DT

    axes[0, 0].plot(t, results['x'], 'b-', lw=1.5, label='$x(t)$: perfiles disponibles')
    axes[0, 0].plot(t, results['y'], 'r-', lw=1.5, label='$y(t)$: usuarios activos')
    axes[0, 0].plot(t, x_ref_seq, 'b--', alpha=0.5, label='$x_{ref}$')
    axes[0, 0].plot(t, y_ref_seq, 'r--', alpha=0.5, label='$y_{ref}$')
    axes[0, 0].axvline(switch * DT, color='gray', ls=':', alpha=0.5)
    axes[0, 0].set_xlabel('Tiempo')
    axes[0, 0].set_ylabel('Estado')
    axes[0, 0].set_title('Evolucion de estados')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(results['x'], results['y'], 'g-', lw=1.5)
    axes[0, 1].plot(results['x'][0], results['y'][0], 'go', label='Inicio',
                    markersize=8)
    axes[0, 1].plot(results['x'][-1], results['y'][-1], 'rs', label='Final',
                    markersize=8)
    axes[0, 1].plot(xr1, yr1, 'k*', label='Ref 1', markersize=12)
    axes[0, 1].plot(xr2, yr2, 'm*', label='Ref 2', markersize=12)
    axes[0, 1].set_xlabel('$x$: perfiles disponibles')
    axes[0, 1].set_ylabel('$y$: usuarios activos')
    axes[0, 1].set_title('Diagrama de fase')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(tc, results['a'], 'b-', lw=1.5, label='$a$: crecimiento perfiles')
    axes[1, 0].plot(tc, results['c'], 'g-', lw=1.5, label='$c$: eficiencia matching')
    axes[1, 0].plot(tc, results['d'], 'r-', lw=1.5, label='$d$: abandono usuarios')
    axes[1, 0].axhline(A_BOUNDS[0], color='b', ls=':', alpha=0.3)
    axes[1, 0].axhline(A_BOUNDS[1], color='b', ls=':', alpha=0.3)
    axes[1, 0].axhline(C_BOUNDS[0], color='g', ls=':', alpha=0.3)
    axes[1, 0].axhline(C_BOUNDS[1], color='g', ls=':', alpha=0.3)
    axes[1, 0].axhline(D_BOUNDS[0], color='r', ls=':', alpha=0.3)
    axes[1, 0].axhline(D_BOUNDS[1], color='r', ls=':', alpha=0.3)
    axes[1, 0].axvline(switch * DT, color='gray', ls=':', alpha=0.5)
    axes[1, 0].set_xlabel('Tiempo')
    axes[1, 0].set_ylabel('Control')
    axes[1, 0].set_title('Senales de control')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(tc, results['cost'], 'k-', lw=1, label='Costo MPC')
    axes[1, 1].set_xlabel('Tiempo')
    axes[1, 1].set_ylabel('Costo')
    axes[1, 1].set_title('Costo de optimizacion')
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle("MPC: Cambio Escalon de Referencia", fontsize=13,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig("mpc_escenario2.png", dpi=150, bbox_inches='tight')
    print("  Grafica guardada: mpc_escenario2.png")
    plt.close()

    return results


def escenario_perturbacion(stochastic=True, seed=42):
    print("\n" + "=" * 60)
    print("  ESCENARIO 3: RECHAZO A PERTURBACIONES")
    print("=" * 60)

    mpc = MPCController()
    dist = {'step': 60, 'dx': 15.0, 'dy': 0.0}
    res = mpc.run_simulation(n_steps=200, x_ref=35.0, y_ref=55.0,
                             x0=35.0, y0=55.0, disturbance=dist,
                             stochastic=stochastic, seed=seed)

    sr = res['success'].mean() * 100
    print(f"  Tasa de exito del solver: {sr:.1f}%")
    print(f"  Perturbacion en t={dist['step'] * DT}: Dx = {dist['dx']}")
    print(f"  Estado inicial: x=35.0, y=55.0 (en referencia)")

    mpc.plot_results(res, "MPC: Rechazo a Perturbacion",
                     save_path="mpc_escenario3.png")
    return res


if __name__ == '__main__':
    r1 = escenario_referencia_constante()
    r2 = escenario_cambio_escalon()
    r3 = escenario_perturbacion()

    print("\n" + "=" * 60)
    print("  RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(f"  Escenario 1 (Referencia constante):")
    print(f"    x_final={r1['x'][-1]:.2f}, y_final={r1['y'][-1]:.2f}")
    print(f"  Escenario 2 (Cambio escalon):")
    print(f"    x_final={r2['x'][-1]:.2f}, y_final={r2['y'][-1]:.2f}")
    print(f"  Escenario 3 (Rechazo perturbacion):")
    print(f"    x_final={r3['x'][-1]:.2f}, y_final={r3['y'][-1]:.2f}")
    print()
    print("  Graficas generadas:")
    print("    - mpc_escenario1.png")
    print("    - mpc_escenario2.png")
    print("    - mpc_escenario3.png")
