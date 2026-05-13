import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize

T = 0.1
X0, Y0 = 40, 50

ALPHA_BOUNDS = (0.2, 1.0)
BETA_BOUNDS = (0.005, 0.04)
DELTA_BOUNDS = (0.002, 0.04)
GAMMA_BOUNDS = (0.1, 0.7)

PENALTY = 1e6

def simular(alpha, beta, delta, gamma, N=800):
    x = np.zeros(N); y = np.zeros(N)
    x[0], y[0] = X0, Y0
    for k in range(N-1):
        x[k+1] = max(x[k] + T * (alpha * x[k] - beta * x[k] * y[k]), 0)
        y[k+1] = max(y[k] + T * (delta * x[k] * y[k] - gamma * y[k]), 0)
    return x, y

def costo(params):
    a, b, d, g = params
    x, y = simular(a, b, d, g)
    ys, xs = y[400:], x[400:]
    y_prom, x_prom = np.mean(ys), np.mean(xs)

    if y_prom < 20 or x_prom < 5:
        return PENALTY + (20 - y_prom)*100 + (5 - x_prom)*100

    x_eq = g / max(d, 1e-6)
    y_eq = a / max(b, 1e-6)

    proporcion = x_prom / max(y_prom, 1)
    proporcion_ideal = 0.6
    J_proporcion = (proporcion - proporcion_ideal)**2 * 50

    y_objetivo = 55
    J_usuarios = ((y_prom - y_objetivo) / y_objetivo)**2 * 40

    cv = np.std(ys) / max(y_prom, 1)
    J_estabilidad = cv * 30

    J_equilibrio_x = ((x_eq - x_prom) / max(x_prom, 1))**2 * 10
    J_equilibrio_y = ((y_eq - y_prom) / max(y_prom, 1))**2 * 10

    retencion = np.exp(-g)
    J_retencion = -retencion * 20

    ganancia = y_prom * 0.5 + y_prom * 0.05 * 9.99 * (d / 0.01) - y_prom * 0.15
    J_ganancia = -ganancia * 0.3

    sat = (d / max(g, 0.01)) * 5
    J_sat = -min(sat, 10) * 2

    return J_proporcion + J_usuarios + J_estabilidad + J_equilibrio_x + J_equilibrio_y + J_retencion + J_ganancia + J_sat


def optimizar():
    print("=" * 62)
    print("     OPTIMIZACION DE APP DE CITAS (MODELO LOTKA-VOLTERRA)")
    print("=" * 62)

    res = differential_evolution(
        costo, [ALPHA_BOUNDS, BETA_BOUNDS, DELTA_BOUNDS, GAMMA_BOUNDS],
        strategy='best1bin', maxiter=1500, popsize=50,
        tol=1e-8, seed=42,
    )

    res2 = minimize(costo, res.x, bounds=[ALPHA_BOUNDS, BETA_BOUNDS, DELTA_BOUNDS, GAMMA_BOUNDS],
                    method='L-BFGS-B')
    xopt = res2.x if res2.fun < res.fun else res.x

    a, b, d, g = xopt
    x_sim, y_sim = simular(a, b, d, g)
    xs, ys = x_sim[400:], y_sim[400:]
    yp, xp = np.mean(ys), np.mean(xs)

    x_eq = g / max(d, 1e-6)
    y_eq = a / max(b, 1e-6)
    proporcion = xp / max(yp, 1)
    retencion = np.exp(-g)
    cv = np.std(ys) / max(yp, 1)
    ganancia = yp * 0.5 + yp * 0.05 * 9.99 * (d / 0.01) - yp * 0.15

    print(f"\n  Costo optimo: {res.fun:.2f}\n")
    print(f"  PARAMETROS DEL SISTEMA")
    print(f"  alpha (crecimiento perfiles):   {a:.4f}")
    print(f"  beta  (tasa de match):           {b:.4f}")
    print(f"  delta (eficiencia algoritmo):    {d:.4f}")
    print(f"  gamma (abandono usuarios):       {g:.4f}")
    print(f"\n  PUNTO DE EQUILIBRIO (x*, y*)")
    print(f"  Perfiles (x* = gamma/delta):     {x_eq:.1f}")
    print(f"  Usuarios (y* = alpha/beta):      {y_eq:.1f}")
    print(f"  Simulados:  x_prom={xp:.1f}, y_prom={yp:.1f}")
    print(f"  Proporcion perfiles/usuario:     {proporcion:.2f}")
    print(f"  Variabilidad (CV):               {cv:.3f}")
    print(f"  Tasa de retencion:               {retencion:.1%}")
    print(f"  Ganancia neta estimada/mes:      ${ganancia:.2f}")

    return a, b, d, g, x_sim, y_sim


def analizar(a, b, d, g):
    print("\n" + "-" * 62)
    print("  ANALISIS DE ESCENARIOS")
    print("-" * 62)

    escenarios = [
        ("OPTIMO", a, b, d, g),
        ("Algoritmo agresivo (delta x3)", a, b, min(d*3, 0.04), g),
        ("Algoritmo lento (delta/4)", a, b, d/4, g),
        ("Alta desercion (gamma x1.8)", a, b, d, min(g*1.8, 0.7)),
        ("Baja desercion (gamma/2)", a, b, d, g/2),
        ("Alta tasa match (beta x3)", a, min(b*3, 0.04), d, g),
    ]

    rows = []
    for nom, aa, bb, dd, gg in escenarios:
        x, y = simular(aa, bb, dd, gg)
        y_est = np.mean(y[400:])
        x_est = np.mean(x[400:])
        cv = np.std(y[400:]) / max(y_est, 1)
        g_ = y_est * 0.5 + y_est * 0.05 * 9.99 * (dd/0.01) - y_est * 0.15
        stable = cv < 0.5
        viable = "Rentable" if (y_est >= 20 and x_est >= 8 and stable) else \
                 "Inestable" if not stable else "Poca base"
        rows.append((nom, y_est, x_est, cv, g_, viable, x, y))
        print(f"\n  {nom}")
        print(f"    Users={y_est:.0f}  Perfiles={x_est:.0f}  CV={cv:.2f}  Ganancia=${g_:.1f}  [{viable}]")

    return rows


def graficar(rows, a, b, d, g):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Optimizacion de App de Citas — Modelo Depredador-Presa", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for r in rows:
        ax.plot(r[6], label=f"{r[0]} (y={r[1]:.0f})", lw=1.5)
    ax.set_xlabel("Tiempo"); ax.set_ylabel("Usuarios y(t)")
    ax.set_title("Evolucion temporal de usuarios")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax2 = axes[0, 1]
    for r in rows:
        ax2.plot(r[7], r[6], label=r[0], lw=1.5)
    ax2.set_xlabel("Perfiles x(t)"); ax2.set_ylabel("Usuarios y(t)")
    ax2.set_title("Diagrama de fase (x vs y)")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    x_eq = g/max(d,1e-6); y_eq = a/max(b,1e-6)
    names = ["alpha", "beta", "delta", "gamma", "x*", "y*"]
    vals = [a, b, d, g, x_eq, y_eq]
    colors = ["#3498db", "#e67e22", "#2ecc71", "#e74c3c", "#9b59b6", "#1abc9c"]
    bars = ax3.bar(names, vals, color=colors, alpha=0.7)
    ax3.set_title("Parametros y punto de equilibrio")
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax4 = axes[1, 1]
    ax4.axis("off")
    proporcion = x_eq/max(y_eq,1)
    text = (
        "PUNTO DE EQUILIBRIO\n"
        "--------------------\n"
        f"x* = gamma/delta = {x_eq:.1f}\n"
        f"  (perfiles potenciales)\n\n"
        f"y* = alpha/beta  = {y_eq:.1f}\n"
        f"  (usuarios activos)\n\n"
        f"Proporcion x*/y* = {proporcion:.2f}\n"
        f"  (ideal: 0.5-1.0)\n\n"
        "TRADE-OFF CENTRAL:\n"
        "delta alto  -> pocos perfiles\n"
        "delta bajo  -> frustracion\n"
        "gamma alto  -> abandono\n"
        "gamma bajo  -> retencion"
    )
    ax4.text(0.1, 0.95, text, transform=ax4.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = "/home/kaforerog/Documentos/U/ciber-III/fin/optimizacion_app_citas.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Grafico guardado en: optimizacion_app_citas.png")


def main():
    a, b, d, g, xs, ys = optimizar()
    rows = analizar(a, b, d, g)
    graficar(rows, a, b, d, g)

    print("\n" + "=" * 62)
    print("     CONCLUSIONES ESTRATEGICAS")
    print("=" * 62)
    print("""
  El modelo Lotka-Volterra aplicado a la app de citas revela:

  1. El algoritmo de matching (delta) debe calibrarse cuidadosamente:
     - Demasiado eficiente: usuarios consiguen pareja y se van
     - Muy ineficiente: usuarios se frustran y abandonan
     - Punto optimo: mantener esperanza sin exito inmediato

  2. La retencion (gamma) depende de la experiencia percibida:
     - La proporcion perfiles/usuario (x/y) es clave
     - Si hay pocos perfiles para explorar -> aburrimiento
     - Si hay muchos perfiles pero sin matches -> frustracion

  3. Para maximizar ganancia:
     - Anuncios: depende del numero de usuarios activos
     - Premium: depende de la eficiencia percibida del algoritmo
     - El equilibrio optimo maximiza el ingreso total

  4. El punto fijo del sistema es:
       x* = gamma/delta    y* = alpha/beta
     La estabilidad del sistema depende de mantener estos
     valores dentro de rangos sostenibles.
""")


if __name__ == "__main__":
    main()
