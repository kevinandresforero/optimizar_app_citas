"""Nonlinear optimization of a dating app using the Lotka-Volterra model.

The algorithm balances matching efficiency (c), user retention (d),
and profile growth (a) to maximize profitability while maintaining a
stable equilibrium.

System equations:
    x_dot = a * x - b * x * y   (profiles / potential matches)
    y_dot = c * x * y - d * y   (active users)

Equilibrium point:
    x* = d / c    y* = a / b

Typical usage::

    from optimizar_app_citas import DatingAppOptimizer

    opt = DatingAppOptimizer(a=0.3, b=0.006, c=0.018, d=0.7)
    opt.optimize()
    opt.analyse_scenarios()
    opt.plot_results()
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize


class DatingAppOptimizer:
    """Optimize a dating-app business model via Lotka-Volterra dynamics.

    The state vector is (x, y) where:
        x = number of profiles / potential matches ("prey")
        y = number of active users ("predators")

    The four parameters control the coupled dynamics and can be tuned to
    maximise profitability while keeping the system stable.

    Attributes:
        a: Profile growth rate (alpha).
        b: Match rate (beta).
        c: Algorithm efficiency (delta).
        d: User abandonment rate (gamma).
        x_sim: Full simulated profile trajectory (after optimisation).
        y_sim: Full simulated user trajectory (after optimisation).
    """

    # Sampling time for Euler discretisation
    DT = 0.1
    # Default initial conditions
    X0 = 40
    Y0 = 50
    # Parameter bounds for optimisation
    ALPHA_BOUNDS = (0.2, 1.0)
    BETA_BOUNDS = (0.005, 0.04)
    DELTA_BOUNDS = (0.002, 0.04)
    GAMMA_BOUNDS = (0.1, 0.7)
    # Penalty for invalid configurations
    PENALTY = 1e6

    def __init__(self, a=0.3, b=0.006, c=0.018, d=0.7):
        """Initialise the optimiser with the Lotka-Volterra parameters.

        Args:
            a: Profile growth rate (alpha).
            b: Match rate (beta).
            c: Algorithm efficiency (delta).
            d: User abandonment rate (gamma).
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x_sim = None
        self.y_sim = None
        self._scenario_rows = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_initial_conditions(self, x0, y0):
        """Override the default initial conditions for simulation.

        Args:
            x0: Initial number of profiles.
            y0: Initial number of active users.
        """
        DatingAppOptimizer.X0 = x0
        DatingAppOptimizer.Y0 = y0

    def set_bounds(self, alpha=None, beta=None, delta=None, gamma=None):
        """Override the default parameter bounds for optimisation.

        Args:
            alpha: Tuple (min, max) for alpha, or None to keep default.
            beta: Tuple (min, max) for beta, or None to keep default.
            delta: Tuple (min, max) for delta, or None to keep default.
            gamma: Tuple (min, max) for gamma, or None to keep default.
        """
        if alpha is not None:
            DatingAppOptimizer.ALPHA_BOUNDS = alpha
        if beta is not None:
            DatingAppOptimizer.BETA_BOUNDS = beta
        if delta is not None:
            DatingAppOptimizer.DELTA_BOUNDS = delta
        if gamma is not None:
            DatingAppOptimizer.GAMMA_BOUNDS = gamma

    def simulate(self, a=None, b=None, c=None, d=None, n_steps=800):
        """Run a forward-Euler simulation of the Lotka-Volterra system.

        Args:
            a: Alpha override (defaults to self.a).
            b: Beta override (defaults to self.b).
            c: Delta override (defaults to self.c).
            d: Gamma override (defaults to self.d).
            n_steps: Number of discrete time steps.

        Returns:
            Tuple (x, y) of numpy arrays of length n_steps.
        """
        a = self.a if a is None else a
        b = self.b if b is None else b
        c = self.c if c is None else c
        d = self.d if d is None else d

        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        x[0], y[0] = self.X0, self.Y0
        dt = self.DT

        for k in range(n_steps - 1):
            x[k + 1] = max(x[k] + dt * (a * x[k] - b * x[k] * y[k]), 0)
            y[k + 1] = max(y[k] + dt * (c * x[k] * y[k] - d * y[k]), 0)

        return x, y

    def optimize(self, n_steps=800):
        """Run global + local optimisation to find the best parameters.

        Uses Differential Evolution for global search followed by
        L-BFGS-B for local refinement. Updates self.a, self.b, self.c,
        self.d and stores trajectories in self.x_sim, self.y_sim.

        Args:
            n_steps: Number of simulation steps for cost evaluation.

        Returns:
            self (for method chaining).
        """
        print("=" * 62)
        print("     OPTIMIZACION DE APP DE CITAS (MODELO LOTKA-VOLTERRA)")
        print("=" * 62)

        def _cost(params):
            return self._cost_function(params, n_steps)

        bounds = [
            self.ALPHA_BOUNDS,
            self.BETA_BOUNDS,
            self.DELTA_BOUNDS,
            self.GAMMA_BOUNDS,
        ]

        res = differential_evolution(
            _cost, bounds,
            strategy='best1bin', maxiter=1500, popsize=50,
            tol=1e-8, seed=42,
        )

        res2 = minimize(_cost, res.x, bounds=bounds, method='L-BFGS-B')
        xopt = res2.x if res2.fun < res.fun else res.x

        self.a, self.b, self.c, self.d = xopt
        self.x_sim, self.y_sim = self.simulate(n_steps=n_steps)

        self._print_optimization_results(res.fun, n_steps)
        return self

    def analyse_scenarios(self, n_steps=800):
        """Compare the optimal configuration against counterfactual scenarios.

        Evaluates aggressive algorithm, slow algorithm, high / low churn,
        and high match rate to validate that the optimum is the most stable
        configuration.

        Args:
            n_steps: Number of simulation steps.

        Returns:
            List of tuples (name, y_mean, x_mean, cv, profit, status,
            x_traj, y_traj) for each scenario.
        """
        print("\n" + "-" * 62)
        print("  ANALISIS DE ESCENARIOS")
        print("-" * 62)

        a, b, c, d = self.a, self.b, self.c, self.d
        scenarios = [
            ("OPTIMO", a, b, c, d),
            ("Algoritmo agresivo (delta x3)", a, b, min(c * 3, 0.04), d),
            ("Algoritmo lento (delta/4)", a, b, c / 4, d),
            ("Alta desercion (gamma x1.8)", a, b, c, min(d * 1.8, 0.7)),
            ("Baja desercion (gamma/2)", a, b, c, d / 2),
            ("Alta tasa match (beta x3)", a, min(b * 3, 0.04), c, d),
        ]

        rows = []
        for name, aa, bb, cc, dd in scenarios:
            x, y = self.simulate(aa, bb, cc, dd, n_steps)
            y_est = float(np.mean(y[n_steps // 2:]))
            x_est = float(np.mean(x[n_steps // 2:]))
            cv = float(np.std(y[n_steps // 2:]) / max(y_est, 1))
            profit = (
                y_est * 0.5
                + y_est * 0.05 * 9.99 * (cc / 0.01)
                - y_est * 0.15
            )
            stable = cv < 0.5
            status = (
                "Rentable"
                if (y_est >= 20 and x_est >= 8 and stable)
                else "Inestable" if not stable
                else "Poca base"
            )
            rows.append((name, y_est, x_est, cv, profit, status, x, y))
            print(f"\n  {name}")
            print(f"    Users={y_est:.0f}  Perfiles={x_est:.0f}  "
                  f"CV={cv:.2f}  Ganancia=${profit:.1f}  [{status}]")

        self._scenario_rows = rows
        return rows

    def plot_results(self, save_path=None):
        """Generate a 2x2 figure summarising the optimisation results.

        Panels:
            1. Time evolution of active users for all scenarios.
            2. Phase diagram (x vs y) for all scenarios.
            3. Bar chart of optimal parameters and equilibrium point.
            4. Info box explaining the equilibrium and trade-offs.

        Args:
            save_path: File path for the PNG. Defaults to
                       ``optimizacion_app_citas.png`` in CWD.
        """
        if self._scenario_rows is None:
            raise RuntimeError("Call analyse_scenarios() before plot_results().")

        rows = self._scenario_rows
        a, b, c, d = self.a, self.b, self.c, self.d
        save_path = save_path or "optimizacion_app_citas.png"

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Optimizacion de App de Citas — Modelo Depredador-Presa",
            fontsize=14,
            fontweight="bold",
        )

        # Top-left: time evolution of users
        ax = axes[0, 0]
        for r in rows:
            ax.plot(r[6], label=f"{r[0]} (y={r[1]:.0f})", lw=1.5)
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Usuarios y(t)")
        ax.set_title("Evolucion temporal de usuarios")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        # Top-right: phase diagram
        ax2 = axes[0, 1]
        for r in rows:
            ax2.plot(r[7], r[6], label=r[0], lw=1.5)
        ax2.set_xlabel("Perfiles x(t)")
        ax2.set_ylabel("Usuarios y(t)")
        ax2.set_title("Diagrama de fase (x vs y)")
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.3)

        # Bottom-left: bar chart
        ax3 = axes[1, 0]
        x_eq = d / max(c, 1e-6)
        y_eq = a / max(b, 1e-6)
        names = ["a (alpha)", "b (beta)", "c (delta)", "d (gamma)", "x*", "y*"]
        vals = [a, b, c, d, x_eq, y_eq]
        colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c"]
        bars = ax3.bar(names, vals, color=colors, alpha=0.7)
        ax3.set_title("Parametros y punto de equilibrio")
        for bar, val in zip(bars, vals):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Bottom-right: info box
        ax4 = axes[1, 1]
        ax4.axis("off")
        proporcion = x_eq / max(y_eq, 1)
        info = (
            "PUNTO DE EQUILIBRIO\n"
            "--------------------\n"
            f"x* = d/c = {x_eq:.1f}\n"
            f"  (perfiles potenciales)\n\n"
            f"y* = a/b  = {y_eq:.1f}\n"
            f"  (usuarios activos)\n\n"
            f"Proporcion x*/y* = {proporcion:.2f}\n"
            f"  (ideal: 0.5-1.0)\n\n"
            "TRADE-OFF CENTRAL:\n"
            "c (delta) alto -> pocos perfiles\n"
            "c (delta) bajo -> frustracion\n"
            "d (gamma) alto -> abandono\n"
            "d (gamma) bajo -> retencion"
        )
        ax4.text(
            0.1, 0.95, info,
            transform=ax4.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Grafico guardado en: {save_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cost_function(self, params, n_steps):
        """Multi-objective cost function for the optimiser.

        Aggregates penalties / rewards for profile/user ratio, user count,
        stability, retention, profit, and equilibrium coherence.

        Args:
            params: (alpha, beta, delta, gamma) tuple.
            n_steps: Simulation length.

        Returns:
            Scalar cost (lower is better).
        """
        a, b, c, d = params
        x, y = self.simulate(a, b, c, d, n_steps)

        ys, xs = y[n_steps // 2:], x[n_steps // 2:]
        y_prom = float(np.mean(ys))
        x_prom = float(np.mean(xs))

        if y_prom < 20 or x_prom < 5:
            return self.PENALTY + (20 - y_prom) * 100 + (5 - x_prom) * 100

        x_eq = d / max(c, 1e-6)
        y_eq = a / max(b, 1e-6)

        proporcion = x_prom / max(y_prom, 1)
        j_prop = (proporcion - 0.6) ** 2 * 50
        j_users = ((y_prom - 55) / 55) ** 2 * 40

        cv = np.std(ys) / max(y_prom, 1)
        j_stab = cv * 30

        j_eq_x = ((x_eq - x_prom) / max(x_prom, 1)) ** 2 * 10
        j_eq_y = ((y_eq - y_prom) / max(y_prom, 1)) ** 2 * 10

        retencion = np.exp(-d)
        j_ret = -retencion * 20

        ganancia = y_prom * 0.5 + y_prom * 0.05 * 9.99 * (c / 0.01) - y_prom * 0.15
        j_gain = -ganancia * 0.3

        sat = (c / max(d, 0.01)) * 5
        j_sat = -min(sat, 10) * 2

        return j_prop + j_users + j_stab + j_eq_x + j_eq_y + j_ret + j_gain + j_sat

    def _print_optimization_results(self, final_cost, n_steps):
        """Print the optimal parameters and equilibrium summary."""
        a, b, c, d = self.a, self.b, self.c, self.d
        xs, ys = self.x_sim[n_steps // 2:], self.y_sim[n_steps // 2:]
        yp = float(np.mean(ys))
        xp = float(np.mean(xs))

        x_eq = d / max(c, 1e-6)
        y_eq = a / max(b, 1e-6)
        proporcion = xp / max(yp, 1)
        retencion = np.exp(-d)
        cv = np.std(ys) / max(yp, 1)
        ganancia = yp * 0.5 + yp * 0.05 * 9.99 * (c / 0.01) - yp * 0.15

        print(f"\n  Costo optimo: {final_cost:.2f}\n")
        print(f"  PARAMETROS DEL SISTEMA")
        print(f"  a (alpha, crecimiento perfiles):  {a:.4f}")
        print(f"  b (beta,  tasa de match):         {b:.4f}")
        print(f"  c (delta, eficiencia algoritmo):  {c:.4f}")
        print(f"  d (gamma, abandono usuarios):     {d:.4f}")
        print(f"\n  PUNTO DE EQUILIBRIO (x*, y*)")
        print(f"  x* = d/c = {x_eq:.1f}   (perfiles)")
        print(f"  y* = a/b = {y_eq:.1f}   (usuarios)")
        print(f"  Simulados:  x_prom={xp:.1f}, y_prom={yp:.1f}")
        print(f"  Proporcion perfiles/usuario:     {proporcion:.2f}")
        print(f"  Variabilidad (CV):               {cv:.3f}")
        print(f"  Tasa de retencion:               {retencion:.1%}")
        print(f"  Ganancia neta estimada/mes:      ${ganancia:.2f}")


def main():
    """Entry point: create an optimiser, run it, and show results."""
    opt = DatingAppOptimizer()
    opt.optimize()
    opt.analyse_scenarios()
    opt.plot_results()

    print("\n" + "=" * 62)
    print("     CONCLUSIONES ESTRATEGICAS")
    print("=" * 62)
    print("""
  El modelo Lotka-Volterra aplicado a la app de citas revela:

  1. El algoritmo de matching (c) debe calibrarse cuidadosamente:
     - Demasiado eficiente: usuarios consiguen pareja y se van
     - Muy ineficiente: usuarios se frustran y abandonan
     - Punto optimo: mantener esperanza sin exito inmediato

  2. La retencion (d) depende de la experiencia percibida:
     - La proporcion perfiles/usuario (x/y) es clave
     - Si hay pocos perfiles para explorar -> aburrimiento
     - Si hay muchos perfiles pero sin matches -> frustracion

  3. Para maximizar ganancia:
     - Anuncios: depende del numero de usuarios activos
     - Premium: depende de la eficiencia percibida del algoritmo
     - El equilibrio optimo maximiza el ingreso total

  4. El punto fijo del sistema es:
       x* = d/c    y* = a/b
     La estabilidad del sistema depende de mantener estos
     valores dentro de rangos sostenibles.
""")


if __name__ == "__main__":
    main()
