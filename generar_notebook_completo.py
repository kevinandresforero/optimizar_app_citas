"""Genera un notebook completo que usa mpc_controlador.py + explicaciones."""
import json

cells = []

def md(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [source]
    })

def code(source):
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [source],
        "outputs": [],
        "execution_count": None
    })

# ============================================================
# TITLE
# ============================================================
md(r"""# Proyecto MPC — Sistema Depredador-Presa (App de Citas)

**Asignatura:** Cibernetica III — 2026-1
**Autores:** Emmanuel Guerrero Piza, Kevin Andres Forero Guaitero
**Universidad Distrital Francisco Jose de Caldas**

---

## Contenido

1. **Modelo matematico** del sistema (Lotka-Volterra para app de citas)
2. **Import y configuracion** del controlador MPC
3. **Escenario 1:** Crecimiento — escalar la app de (10,10) a (50,70)
4. **Escenario 2a:** Retencion — mantener viva la app con alta desercion
5. **Escenario 2b:** Retencion — crecer en perfiles en entorno adverso
6. **Escenario 3:** Dinamico — a(t), d(t) markovianos (a(t), d(t) markovianos)
7. **Comparativa 2x2** de los 4 escenarios
8. **Metricas de error RMS**
9. **Conclusiones**

> **Nota clave:** Solo la eficiencia del matching ($c$) es variable de control.
> $a$ (crecimiento de perfiles) y $d$ (abandono de usuarios) son parametros
> del escenario que el MPC conoce pero no optimiza.
""")

# ============================================================
# SECTION 1: MODELO
# ============================================================
md(r"""---
## 1. Modelo del sistema

El sistema dinamico corresponde a un modelo **Lotka-Volterra** (depredador-presa)
aplicado a la dinamica de una aplicacion de citas:

### Ecuaciones de evolucion

$$
\begin{aligned}
\dot{x} &= a\,x - b\,x\,y \quad &\text{(perfiles / potenciales matches)} \\[4pt]
\dot{y} &= c\,x\,y - d\,y \quad &\text{(usuarios activos)}
\end{aligned}
$$

### Variables de estado

| Simbolo | Variable | Unidades |
|---------|----------|----------|
| $x(t)$ | Numero de perfiles disponibles (presa) | Adimensional |
| $y(t)$ | Numero de usuarios activos (depredador) | Adimensional |

### Parametros del modelo

| Parametro | Nombre | Significado fisico |
|-----------|--------|--------------------|
| $a$ | $\alpha$ (alpha) | Tasa de crecimiento de perfiles (nuevos registros) |
| $b$ | $\beta$ (beta) | Tasa de match (interaccion usuario-perfil) |
| $c$ | $\delta$ (delta) | Eficiencia del algoritmo de matching |
| $d$ | $\gamma$ (gamma) | Tasa de abandono de usuarios (churn) |

### Punto de equilibrio

$$
x^* = \frac{d}{c}, \qquad y^* = \frac{a}{b}
$$

**Observacion crucial:** $y^*$ depende **unicamente** de $a$ y $b$. El
controlador solo manipula $c$, que afecta a $x^*$ pero **no** al equilibrio
de usuarios activos. Esto impone un limite fundamental cuando $a$ es bajo.

### Parametros optimos (obtenidos en Parte 2)

| Parametro | Valor optimo |
|-----------|-------------|
| $a$ | 0.2909 |
| $b$ | 0.0055 |
| $c$ | 0.0179 |
| $d$ | 0.7000 |

**Equilibrio natural:** $x^* = 39.2$ perfiles, $y^* = 53.0$ usuarios.
""")

# ============================================================
# SECTION 2: IMPORTS
# ============================================================
md(r"""---
## 2. Import del controlador y configuracion global

Todo el codigo del MPC y las funciones de simulacion estan en
`mpc_controlador.py`. Aqui solo importamos lo necesario y
configuramos los parametros generales.
""")

code(r"""import numpy as np
import matplotlib.pyplot as plt
from mpc_controlador import (MPCController, grafica_comparativa,
                              simular_sin_control, generar_trayectoria_dinamica,
                              simulate_step_stochastic,
                              DT, B_OPT, C_REF, C_BOUNDS,
                              SIGMA_X, SIGMA_Y, PULSE_RATE,
                              PULSE_SCALE_X, PULSE_SCALE_Y,
                              N_STEPS)

print(f"Configuracion lista. N_STEPS = {N_STEPS}")
print(f"Tiempo total de simulacion = {N_STEPS * DT:.0f} uds.")
print(f"b (tasa de match) = {B_OPT}")
print(f"c_ref (referencia de control) = {C_REF}")
print(f"c_bounds = {C_BOUNDS}")
""")

# ============================================================
# NOTA: MISMA ARQUITECTURA DEL CONTROLADOR
# ============================================================
md(r"""---
> **Nota:** Todos los escenarios usan la **misma** instancia de
> `MPCController` con los mismos hiperparámetros ($N=15$, $M=10$,
> $Q=\operatorname{diag}(1,1)$, $r_c=0.1$, $c\in[0.002,0.04]$,
> solver SLSQP). Solo cambian los parámetros $a$, $d$, el estado
> inicial $(x_0,y_0)$ y la referencia $(x_{\text{ref}},y_{\text{ref}})$.
> Esto demuestra que un **único controlador bien sintonizado**
> puede operar bajo condiciones de mercado muy distintas.
""")

# ============================================================
# SECTION 3: ESCENARIO 1 — CRECIMIENTO
# ============================================================
md(r"""---
## 3. Escenario 1: Crecimiento

**Objetivo:** Escalar la app desde un estado inicial bajo hasta una
referencia saludable.

| Parametro | Valor |
|-----------|-------|
| $a$ (crecimiento perfiles) | 0.5 |
| $d$ (abandono usuarios) | 0.2 |
| Estado inicial $(x_0, y_0)$ | (10, 10) |
| Referencia $(x_{\text{ref}}, y_{\text{ref}})$ | (50, 70) |
| Equilibrio natural $(x^*, y^*)$ | (11.2, 91.1) |

**Sin control:** $c$ fijo en $c_{\text{ref}}$ → $x$ colapsa mientras
$y$ se dispara hacia su equilibrio (91).

**Con MPC:** $c$ se ajusta para mantener ambos estados cerca de la
referencia.
""")

code(r"""print("=" * 60)
print("  ESCENARIO 1: CRECIMIENTO (a=0.5, d=0.2)")
print("  x0=10, y0=10  |  ref=(50, 70)")
print("=" * 60)

seed1 = 42
a1, d1 = 0.5, 0.2
x01, y01 = 10.0, 10.0
xr1, yr1 = 50.0, 70.0
eq_x1, eq_y1 = d1 / C_REF, a1 / B_OPT
print(f"  Equilibrio natural: x*={eq_x1:.1f}, y*={eq_y1:.1f}")
print("")

# --- Sin control ---
print("  Simulando sin control...")
x_nc1, y_nc1 = simular_sin_control(x01, y01, a1, d1, N_STEPS, seed1)

# --- Con MPC ---
print("  Simulando con MPC...")
mpc1 = MPCController()
res1 = mpc1.run_simulation(N_STEPS, xr1, yr1, a1, d1,
                            x0=x01, y0=y01, stochastic=True, seed=seed1)

# --- Resultados ---
sr1 = res1['success'].mean() * 100
print(f"  Tasa exito solver: {sr1:.1f}%")
print("")
print(f"  Sin control:")
print(f"    x_final={x_nc1[-1]:.1f}, y_final={y_nc1[-1]:.1f}")
print(f"  Con MPC:")
print(f"    x_final={res1['x'][-1]:.1f}, y_final={res1['y'][-1]:.1f}")
print(f"  Con MPC: x<=0? {any(res1['x']<=0)}  y<=0? {any(res1['y']<=0)}")
print(f"  Sin control: x<=0? {any(x_nc1<=0)}  y<=0? {any(y_nc1<=0)}")

grafica_comparativa(
    N_STEPS, x_nc1, y_nc1, res1['x'], res1['y'], res1['c'],
    xr1, yr1, eq_x1, eq_y1, a1, d1,
    "Escenario 1: Crecimiento — Sin control vs MPC",
    "comparativa_escenario1.png"
)
""")

# ============================================================
# SECTION 4: ESCENARIO 2A — MANTENER
# ============================================================
md(r"""---
## 4. Escenario 2a: Retencion — Mantener

**Objetivo:** Mantener la app con vida a pesar de alta desercion y
bajo crecimiento de perfiles.

| Parametro | Valor |
|-----------|-------|
| $a$ (crecimiento perfiles) | 0.1 |
| $d$ (abandono usuarios) | 0.6 |
| Estado inicial $(x_0, y_0)$ | (60, 60) |
| Referencia $(x_{\text{ref}}, y_{\text{ref}})$ | (35, 20) |
| Equilibrio natural $(x^*, y^*)$ | (33.6, 18.2) |

Aqui $y^* = a/b = 18.2$ esta **fijo** — el MPC no puede cambiar el
equilibrio de usuarios. Sin control, $y$ colapsa a 0 (app muere).
Con MPC, $c$ ajusta para mantener $y$ cerca de su equilibrio natural.
""")

code(r"""print("=" * 60)
print("  ESCENARIO 2A: RETENCION — MANTENER (a=0.1, d=0.6)")
print("  x0=60, y0=60  |  ref=(35, 20)")
print("  y* = a/b = 18.2 (fijo, no controlable)")
print("=" * 60)

seed2a = 43
a2a, d2a = 0.1, 0.6
x02a, y02a = 60.0, 60.0
xr2a, yr2a = 35.0, 20.0
eq_x2a, eq_y2a = d2a / C_REF, a2a / B_OPT
print(f"  Equilibrio natural: x*={eq_x2a:.1f}, y*={eq_y2a:.1f}")
print("")

# --- Sin control ---
print("  Simulando sin control...")
x_nc2a, y_nc2a = simular_sin_control(x02a, y02a, a2a, d2a, N_STEPS, seed2a)

# --- Con MPC ---
print("  Simulando con MPC...")
mpc2a = MPCController()
res2a = mpc2a.run_simulation(N_STEPS, xr2a, yr2a, a2a, d2a,
                              x0=x02a, y0=y02a, stochastic=True, seed=seed2a)

# --- Resultados ---
sr2a = res2a['success'].mean() * 100
print(f"  Tasa exito solver: {sr2a:.1f}%")
print("")
print(f"  Sin control:")
print(f"    x_final={x_nc2a[-1]:.1f}, y_final={y_nc2a[-1]:.1f}")
print(f"  Con MPC:")
print(f"    x_final={res2a['x'][-1]:.1f}, y_final={res2a['y'][-1]:.1f}")
print(f"  Con MPC: x<=0? {any(res2a['x']<=0)}  y<=0? {any(res2a['y']<=0)}")
print(f"  Sin control: x<=0? {any(x_nc2a<=0)}  y<=0? {any(y_nc2a<=0)}")

grafica_comparativa(
    N_STEPS, x_nc2a, y_nc2a, res2a['x'], res2a['y'], res2a['c'],
    xr2a, yr2a, eq_x2a, eq_y2a, a2a, d2a,
    "Escenario 2a: Retencion — Mantener (Sin control vs MPC)",
    "comparativa_escenario2a.png"
)
""")

# ============================================================
# SECTION 5: ESCENARIO 2B — CRECER EN X
# ============================================================
md(r"""---
## 5. Escenario 2b: Retencion — Crecer en $x$

**Objetivo:** Aumentar el numero de perfiles incluso en entorno adverso.

| Parametro | Valor |
|-----------|-------|
| $a$ (crecimiento perfiles) | 0.1 |
| $d$ (abandono usuarios) | 0.6 |
| Estado inicial $(x_0, y_0)$ | (60, 60) |
| Referencia $(x_{\text{ref}}, y_{\text{ref}})$ | (60, 25) |
| Equilibrio natural $(x^*, y^*)$ | (33.6, 18.2) |

Para $x^*=60$ el MPC debe reducir $c$ (asi $x^*=d/c$ sube).
$y$ sigue limitado por $y^* = a/b$. Sin control, $y\to0$.
""")

code(r"""print("=" * 60)
print("  ESCENARIO 2B: RETENCION — CRECER EN x (a=0.1, d=0.6)")
print("  x0=60, y0=60  |  ref=(60, 25)")
print("=" * 60)

seed2b = 44
a2b, d2b = 0.1, 0.6
x02b, y02b = 60.0, 60.0
xr2b, yr2b = 60.0, 25.0
eq_x2b, eq_y2b = d2b / C_REF, a2b / B_OPT
print(f"  Equilibrio natural: x*={eq_x2b:.1f}, y*={eq_y2b:.1f}")
print("")

print("  Simulando sin control...")
x_nc2b, y_nc2b = simular_sin_control(x02b, y02b, a2b, d2b, N_STEPS, seed2b)

print("  Simulando con MPC...")
mpc2b = MPCController()
res2b = mpc2b.run_simulation(N_STEPS, xr2b, yr2b, a2b, d2b,
                              x0=x02b, y0=y02b, stochastic=True, seed=seed2b)

sr2b = res2b['success'].mean() * 100
print(f"  Tasa exito solver: {sr2b:.1f}%")
print("")
print(f"  Sin control:")
print(f"    x_final={x_nc2b[-1]:.1f}, y_final={y_nc2b[-1]:.1f}")
print(f"  Con MPC:")
print(f"    x_final={res2b['x'][-1]:.1f}, y_final={res2b['y'][-1]:.1f}")
print(f"  Con MPC: x<=0? {any(res2b['x']<=0)}  y<=0? {any(res2b['y']<=0)}")
print(f"  Sin control: x<=0? {any(x_nc2b<=0)}  y<=0? {any(y_nc2b<=0)}")

grafica_comparativa(
    N_STEPS, x_nc2b, y_nc2b, res2b['x'], res2b['y'], res2b['c'],
    xr2b, yr2b, eq_x2b, eq_y2b, a2b, d2b,
    "Escenario 2b: Retencion — Crecer en x (Sin control vs MPC)",
    "comparativa_escenario2b.png"
)
""")

# ============================================================
# SECTION 6: ESCENARIO 3 — DINAMICO
# ============================================================
md(r"""---
## 6. Escenario 3: Dinamico (mercado impredecible)

**Objetivo:** Probar la robustez del MPC cuando el entorno cambia
abruptamente.

| Parametro | Valor |
|-----------|-------|
| $a(t), d(t)$ | Proceso Markoviano 3 modos |
| Inicio $(x_0, y_0)$ | (30, 30) |
| Referencia $(x_{\text{ref}}, y_{\text{ref}})$ | (40, 40) |

Los modos: **Crecimiento** ($a=0.6, d=0.15$), **Estable** ($a=0.3, d=0.4$),
**Crisis** ($a=0.08, d=0.7$) con transiciones abruptas.

El MPC recibe en cada paso $a(k)$ y $d(k)$ actuales y adapta $c(k)$
en tiempo real. Sin control, la app colapsa durante las crisis.
""")

code(r"""print("=" * 60)
print("  ESCENARIO 3: DINAMICO (a(t), d(t) markovianos)")
print("  x0=30, y0=30  |  ref=(40, 40)")
print("=" * 60)

seed3 = 45
x03, y03 = 30.0, 30.0
xr3, yr3 = 40.0, 40.0

# Escenario 3: a(t), d(t) markovianos (dinamicos)
# ---------------------------------------------------------------------------
print("  Generando trayectorias dinamicas de a(t) y d(t)...")
a_seq3, d_seq3 = generar_trayectoria_dinamica(N_STEPS, seed=seed3)

# --- Sin control ---
print("  Simulando sin control...")
rng_nc3 = np.random.default_rng(seed3)
x_nc3 = np.zeros(N_STEPS + 1)
y_nc3 = np.zeros(N_STEPS + 1)
x_nc3[0], y_nc3[0] = x03, y03
for k in range(N_STEPS):
    x_nc3[k+1], y_nc3[k+1], _, _, _, _ = simulate_step_stochastic(
        x_nc3[k], y_nc3[k], a_seq3[k], B_OPT, C_REF, d_seq3[k],
        rng_nc3, SIGMA_X, SIGMA_Y, PULSE_RATE,
        PULSE_SCALE_X, PULSE_SCALE_Y
    )

# --- Con MPC ---
print("  Simulando con MPC (adaptacion en tiempo real)...")
mpc3 = MPCController()
mpc3.c_prev = None
x_ctrl3 = np.zeros(N_STEPS + 1)
y_ctrl3 = np.zeros(N_STEPS + 1)
c_traj3 = np.zeros(N_STEPS)
x_ctrl3[0], y_ctrl3[0] = x03, y03
rng_mpc3 = np.random.default_rng(seed3)

for k in range(N_STEPS):
    x_cur, y_cur = x_ctrl3[k], y_ctrl3[k]
    a_cur, d_cur = a_seq3[k], d_seq3[k]
    c_opt, success, _ = mpc3.solve(x_cur, y_cur, xr3, yr3, a_cur, d_cur)
    c_traj3[k] = c_opt
    x_ctrl3[k+1], y_ctrl3[k+1], _, _, _, _ = \
        simulate_step_stochastic(
            x_cur, y_cur, a_cur, mpc3.b, c_opt, d_cur,
            rng_mpc3, SIGMA_X, SIGMA_Y, PULSE_RATE,
            PULSE_SCALE_X, PULSE_SCALE_Y
        )

print("")
print(f"  Sin control:")
print(f"    x_final={x_nc3[-1]:.1f}, y_final={y_nc3[-1]:.1f}")
print(f"  Con MPC:")
print(f"    x_final={x_ctrl3[-1]:.1f}, y_final={y_ctrl3[-1]:.1f}")
print(f"  Con MPC: x<=0? {any(x_ctrl3<=0)}  y<=0? {any(y_ctrl3<=0)}")
print(f"  Sin control: x<=0? {any(x_nc3<=0)}  y<=0? {any(y_nc3<=0)}")

grafica_comparativa(
    N_STEPS, x_nc3, y_nc3, x_ctrl3, y_ctrl3, c_traj3,
    xr3, yr3, 39.2, 53.0, np.nan, np.nan,
    "Escenario 3: Dinamico — a(t), d(t) variables (Sin control vs MPC)",
    "comparativa_escenario3.png"
)
""")

# ============================================================
# SECTION 7: COMPARATIVA 2x2
# ============================================================
md(r"""---
## 7. Comparativa de los 4 escenarios (2x2)

Vista conjunta. Sin control (transparente) vs con MPC (intenso).
""")

code(r"""print("Generando figura comparativa 2x2...")

fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.5))
t = np.arange(N_STEPS + 1) * DT

def panel(ax, x_nc, y_nc, x_ctrl, y_ctrl, xr, yr, titulo):
    ax.plot(t, x_nc, 'b-', lw=0.5, alpha=0.45, label='$x$ s/ control')
    ax.plot(t, y_nc, 'r-', lw=0.5, alpha=0.45, label='$y$ s/ control')
    ax.plot(t, x_ctrl, 'b-', lw=1.1, label='$x$ con MPC')
    ax.plot(t, y_ctrl, 'r-', lw=1.1, label='$y$ con MPC')
    ax.axhline(xr, color='b', ls='--', alpha=0.25)
    ax.axhline(yr, color='r', ls='--', alpha=0.25)
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Estado')
    ax.set_title(titulo, fontsize=7, fontweight='bold')
    ax.legend(fontsize=4.5, ncol=2, loc='upper right')
    ax.grid(alpha=0.12)

panel(axes[0,0], x_nc1, y_nc1, res1['x'], res1['y'],
      xr1, yr1, 'Esc. 1: Crecimiento')
panel(axes[0,1], x_nc2a, y_nc2a, res2a['x'], res2a['y'],
      xr2a, yr2a, 'Esc. 2a: Mantener')
panel(axes[1,0], x_nc2b, y_nc2b, res2b['x'], res2b['y'],
      xr2b, yr2b, 'Esc. 2b: Crecer en x')
panel(axes[1,1], x_nc3, y_nc3, x_ctrl3, y_ctrl3,
      xr3, yr3, 'Esc. 3: Dinamico')

plt.suptitle('Comparativa sin control vs con MPC por escenario',
             fontsize=10, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_comparativa_4escenarios.png', dpi=200, bbox_inches='tight')
plt.show()
plt.close()
print("  fig_comparativa_4escenarios.png generado")
""")

# ============================================================
# SECTION 8: METRICAS
# ============================================================
md(r"""---
## 8. Metricas de error RMS

Error RMS en ventana estacionaria (ultimos 700 pasos):

$$
\sigma = \sqrt{\frac{1}{N}\sum_{k}(x_k - x_{\text{ref}})^2}
$$
""")

code(r"""print("=" * 60)
print("  METRICAS DE ERROR RMS (ultimos 700 pasos)")
print("=" * 60)

w = 700

def rms(err):
    return np.sqrt(np.mean(err**2))

esc = [
    ("Esc. 1 Crecimiento", x_nc1, y_nc1, res1['x'], res1['y'], xr1, yr1),
    ("Esc. 2a Mantener",   x_nc2a, y_nc2a, res2a['x'], res2a['y'], xr2a, yr2a),
    ("Esc. 2b Crecer",     x_nc2b, y_nc2b, res2b['x'], res2b['y'], xr2b, yr2b),
]

print(f"\n  {'Escenario':<22} {'Var':<4} {'Sin control':<14} {'Con MPC':<14} {'Mejora':<8}")
print(f"  {'-'*62}")

for nombre, x_nc, y_nc, x_c, y_c, xr, yr in esc:
    ex_nc = rms(x_nc[-w:] - xr)
    ey_nc = rms(y_nc[-w:] - yr)
    ex_c = rms(x_c[-w:] - xr)
    ey_c = rms(y_c[-w:] - yr)
    mx = ex_nc / ex_c if ex_c > 0 else float('inf')
    my = ey_nc / ey_c if ey_c > 0 else float('inf')
    print(f"  {nombre:<22} x    {ex_nc:<14.2f} {ex_c:<14.2f} {mx:<8.0f}x")
    print(f"  {'':<22} y    {ey_nc:<14.2f} {ey_c:<14.2f} {my:<8.0f}x")

# Dinamico
ex3_nc = rms(x_nc3[-w:] - xr3)
ey3_nc = rms(y_nc3[-w:] - yr3)
ex3_c = rms(x_ctrl3[-w:] - xr3)
ey3_c = rms(y_ctrl3[-w:] - yr3)
mx3 = ex3_nc / ex3_c if ex3_c > 0 else float('inf')
my3 = ey3_nc / ey3_c if ey3_c > 0 else float('inf')
print(f"  {'Esc. 3 Dinamico':<22} x    {ex3_nc:<14.2f} {ex3_c:<14.2f} {mx3:<8.0f}x")
print(f"  {'':<22} y    {ey3_nc:<14.2f} {ey3_c:<14.2f} {my3:<8.0f}x")
""")

# ============================================================
# SECTION 9: CONCLUSIONES
# ============================================================
md(r"""---
## 9. Conclusiones

1. **Solo $c$ como control:** $y^* = a/b$ es un equilibrio **fijo** que el
   MPC no puede alterar. Cuando $a$ es bajo, el numero de usuarios en
   equilibrio esta severamente restringido.

2. **Escenario 1 (Crecimiento):** Con $a=0.5$, $d=0.2$, el MPC escala la app
   de $(10,10)$ a niveles cercanos a $(50,70)$. Sin control $x$ colapsa.

3. **Escenario 2a (Mantener):** Con $a=0.1$, $d=0.6$, el MPC mantiene $y$
   vivo en $y\approx 18$. Sin control, $y\to 0$ (app muere).

4. **Escenario 2b (Crecer en $x$):** El MPC reduce $c$ para elevar
   $x^* = d/c$, manteniendo $x$ sobre el equilibrio natural.

5. **Escenario 3 (Dinamico):** El MPC se adapta en tiempo real a
   variaciones de $a(t)$ y $d(t)$, manteniendo la app con vida aun en
   condiciones extremas.

6. **Robustez:** SLSQP con warm-start converge al 100% en todos los
   escenarios. $c(t)$ siempre respeta las cotas $[0.002, 0.04]$.
""")

# ============================================================
# BUILD NOTEBOOK
# ============================================================
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('proyecto_mpc_completo.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook generado: proyecto_mpc_completo.ipynb")
print(f"Total de celdas: {len(cells)}")
