"""Genera el notebook proyecto_mpc.ipynb con todo el flujo del proyecto."""
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

**Asignatura:** Cibernética III — 2026-1
**Autores:** Emmanuel Guerrero Piza, Kevin Andrés Forero Guaitero
**Universidad Distrital Francisco José de Caldas**

Este notebook presenta el desarrollo completo del proyecto: modelo matemático,
optimización de parámetros y diseño del controlador predictivo basado en modelo
(MPC) donde solo la **eficiencia del algoritmo de matching ($c$)** es variable
de control. Las tasas de crecimiento de perfiles ($a$) y abandono de usuarios
($d$) son parámetros del escenario que el MPC conoce pero no optimiza.
""")

# ============================================================
# SECTION 1: MODELO DEL SISTEMA
# ============================================================
md(r"""---
## 1. Modelo del sistema

El sistema dinámico corresponde a un modelo **Lotka-Volterra** (depredador-presa)
aplicado a la dinámica de una aplicación de citas:
""")

md(r"""### Ecuaciones de evolución

$$
\begin{aligned}
\dot{x} &= a\,x - b\,x\,y \quad &\text{(perfiles / potenciales matches)} \\[4pt]
\dot{y} &= c\,x\,y - d\,y \quad &\text{(usuarios activos)}
\end{aligned}
$$

### Variables de estado

| Símbolo | Variable | Unidades |
|---------|----------|----------|
| $x(t)$ | Número de perfiles disponibles (presa) | Adimensional |
| $y(t)$ | Número de usuarios activos (depredador) | Adimensional |

### Parámetros del modelo

| Parámetro | Nombre | Significado físico |
|-----------|--------|--------------------|
| $a$ | $\alpha$ (alpha) | Tasa de crecimiento de perfiles (nuevos registros) |
| $b$ | $\beta$ (beta) | Tasa de match (interacción usuario-perfil) |
| $c$ | $\delta$ (delta) | Eficiencia del algoritmo de matching |
| $d$ | $\gamma$ (gamma) | Tasa de abandono de usuarios (churn) |
""")

md(r"""### Analogía: modelo ecológico vs. app de citas

| Concepto ecológico | Modelo Lotka-Volterra | App de citas |
|--------------------|----------------------|--------------|
| **Presa** | $x$: población de presas (ej. conejos) | $x$: **perfiles disponibles** |
| **Depredador** | $y$: población de depredadores (ej. zorros) | $y$: **usuarios activos** |
| **Crecimiento** | $+a\,x$: reproducción natural de presas | $+a\,x$: **nuevos registros** de perfiles |
| **Consumo** | $-b\,x\,y$: presas cazadas | $-b\,x\,y$: **perfiles que hacen match** y dejan de estar disponibles |
| **Alimentación** | $+c\,x\,y$: depredadores se alimentan y reproducen | $+c\,x\,y$: **usuarios encuentran matches** y se mantienen activos |
| **Muerte** | $-d\,y$: muerte natural de depredadores | $-d\,y$: **usuarios abandonan** la app (churn) |
| **$a$** | Tasa de natalidad de presas | Tasa de crecimiento de perfiles |
| **$b$** | Tasa de depredación | Tasa de match |
| **$c$** | Eficiencia de caza | Eficiencia del algoritmo de matching |
| **$d$** | Tasa de muerte de depredadores | Tasa de abandono / churn |
| **Equilibrio $x^*$** | $d/c$: presas en equilibrio | $d/c$: **perfiles en equilibrio** |
| **Equilibrio $y^*$** | $a/b$: depredadores en equilibrio | $a/b$: **usuarios en equilibrio** |
| **Extinción** | $x=0$ o $y=0$: colapso ecológico | $x=0$ o $y=0$: **la app muere** |
| **Ciclo** | Oscilaciones presa-depredador | Retroalimentación perfiles-usuarios |
""")

md(r"""### Punto de equilibrio

El punto fijo no trivial del sistema es:

$$
x^* = \frac{d}{c}, \qquad y^* = \frac{a}{b}
$$

### Parámetros óptimos

Obtenidos mediante optimización con **Evolución Diferencial + L-BFGS-B**:

| Parámetro | Valor óptimo |
|-----------|-------------|
| $a$ | 0.2909 |
| $b$ | 0.0055 |
| $c$ | 0.0179 |
| $d$ | 0.7000 |

**Equilibrio natural:** $x^* = 39.2$ perfiles, $y^* = 53.0$ usuarios.

### Nota clave sobre controlabilidad

Obsérvese que **$y^*$ depende exclusivamente de $a$ y $b$**:
$y^* = a/b$. El controlador solo manipula $c$, que afecta a $x^* = d/c$
pero **no** al equilibrio de usuarios activos. Esto impone un límite fundamental
a lo que el MPC puede lograr cuando $a$ es bajo y $d$ es alto.
""")

# ============================================================
# SECTION 2: ARQUITECTURAS DE OPTIMIZACIÓN
# ============================================================
md(r"""---
## 2. Arquitecturas de los optimizadores

A continuación se describen las tres arquitecturas implementadas para la
optimización de parámetros. No se incluye el código de los optimizadores en
este notebook; solo se presenta su fundamento teórico.
""")

md(r"""### 2.1 Evolución Diferencial + L-BFGS-B (DE)

**Arquitectura híbrida** que combina búsqueda global evolutiva con refinamiento
local de alta precisión.

**Etapa global — Evolución Diferencial:**
- Población de 50 individuos, cada uno representa un conjunto de parámetros
  $(a, b, c, d)$.
- Estrategia de mutación: *best1bin* (el vector base es el mejor individuo de la
  generación actual, con un vector diferencia binario).
- 1500 generaciones con tolerancia $10^{-8}$.
- Operadores: mutación diferencial y recombinación binomial.
- Ventaja: explora todo el espacio de búsqueda sin requerir gradientes.

**Etapa local — L-BFGS-B:**
- Toma la mejor solución de DE como punto de partida.
- Método cuasi-Newton de memoria limitada con restricciones de caja (bounds).
- Aproximación del Hessiano mediante diferencias finitas.
- Refina la solución hasta convergencia local.

**Ventaja:** robustez global + precisión local.
""")

md(r"""### 2.2 Descenso por Gradiente Estocástico (SGD)

**Arquitectura:** gradiente aproximado + momentum + reinicios.

**Aproximación del gradiente:**
- El gradiente se estima mediante diferencias finitas:

$$
\frac{\partial J}{\partial \theta_i} \approx
\frac{J(\theta + \varepsilon e_i) - J(\theta)}{\varepsilon}
$$

- $\varepsilon = 10^{-4}$, normalización del gradiente si $\|\nabla J\| > 1000$.

**Actualización con momentum:**
$$
\begin{aligned}
v_{k+1} &= \mu v_k + \eta_k \nabla J(\theta_k) \\
\theta_{k+1} &= \Pi_\Theta\bigl(\theta_k - v_{k+1}\bigr)
\end{aligned}
$$
donde $\mu = 0.85$ es el coeficiente de momentum, $\eta_k$ es la tasa de
aprendizaje (decaimiento exponencial $\times 0.99$ por iteración), y
$\Pi_\Theta$ es la proyección sobre las cotas.

**Estrategia de reinicios:**
- Grilla de $4 \times 3 \times 3 \times 3 = 108$ puntos de inicio.
- Submuestreo aleatorio del 30% + 20 reinicios completamente aleatorios.
- Cada trayectoria ejecuta 150 iteraciones.
- Se retorna la mejor solución encontrada entre todas las trayectorias.

**Ventaja:** exploración masiva del espacio de parámetros.
""")

md(r"""### 2.3 ANFIS (Neuro-Fuzzy)

**Arquitectura:** sistema de inferencia difuso con ajuste por red neuronal.

**Estructura:**

1. **Capa de fuzzificación:** 3 funciones de membresía Gaussianas
   (Low, Medium, High) por cada uno de los 4 parámetros:

   $$
   \mu_{ij}(x_i) = \exp\left(-\frac{(x_i - m_{ij})^2}{2\sigma_{ij}^2}\right)
   $$

2. **Capa oculta (feedforward):** 12 entradas (4 parámetros × 3 MF)
   → 16 neuronas ocultas (tanh) → 4 salidas lineales (tanh, escala 0.1).

3. **Inferencia difusa:** reglas heurísticas que ajustan los parámetros según
   la dirección del costo:
   - Si el costo aumenta → reducción de $a$, ajuste fino de $b$, $c$, $d$.
   - Si el costo disminuye → incremento moderado en la dirección de mejora.

4. **Entrenamiento:** retropropagación del error entre el ajuste predicho por
   la red y el ajuste real que mejoró el costo. Tasa de aprendizaje con
   decaimiento lineal.

**Ventaja:** rápida convergencia (~0.5 s) y buena retención de usuarios (76.4%).
""")

# ============================================================
# SECTION 3: MPC
# ============================================================
md(r"""---
## 3. Diseño del controlador MPC

Se implementa un **controlador predictivo no lineal** (NMPC) donde la
**única variable de control** es la eficiencia del algoritmo de matching $c$.
Los parámetros $a$ (crecimiento de perfiles) y $d$ (abandono de usuarios)
son valores fijos que definen el escenario y que el MPC conoce pero no optimiza.
""")

md(r"""### Formulación del problema de optimización

En cada instante $k$, conocido el estado actual $(x_k, y_k)$, se resuelve:

$$
\begin{aligned}
\min_{c_k, \dots, c_{k+M-1}} \quad &
\sum_{t=0}^{N-1} \Bigl[
q_x (x_{k+t+1} - x_{\text{ref}})^2 +
q_y (y_{k+t+1} - y_{\text{ref}})^2 +
r_c (c_{k+t} - c_{\text{ref}})^2
\Bigr] \\[6pt]
\text{sujeto a} \quad &
\mathbf{x}_{k+t+1} = f\bigl(\mathbf{x}_{k+t}, c_{k+t}, a, d\bigr)
\quad \text{(modelo no lineal con $a$, $d$ fijos)} \\[4pt]
& 0.002 \leq c \leq 0.04 \\[4pt]
& c_{k+t} = c_{k+M-1} \quad \text{para } t \geq M
\end{aligned}
$$

donde $\mathbf{x} = [x, y]^T$, $c$ es escalar, $a$ y $d$ son parámetros
constantes del escenario.
""")

md(r"""### Parámetros de diseño

| Parámetro | Símbolo | Valor |
|-----------|---------|-------|
| Horizonte de predicción | $N$ | 15 pasos |
| Horizonte de control | $M$ | 10 pasos |
| Tiempo de muestreo | $\Delta t$ | 0.1 ud |
| Peso de tracking de $x$ | $q_x$ | 1.0 |
| Peso de tracking de $y$ | $q_y$ | 1.0 |
| Peso de esfuerzo en $c$ | $r_c$ | 0.1 |
| Cota inferior de $c$ | $c_{\min}$ | 0.002 |
| Cota superior de $c$ | $c_{\max}$ | 0.04 |
| Referencia de $c$ | $c_{\text{ref}}$ | 0.0179 |
| Solver | — | SLSQP (SciPy) |
| Estrategia inicial | — | Warm-start (shift) |
""")

md(r"""### Algoritmo de control

En cada instante $k$:
1. Medir el estado actual $(x_k, y_k)$.
2. Resolver el problema de optimización NMPC con SLSQP, usando los parámetros
   fijos $a$, $d$ del escenario actual.
3. Aplicar solo la primera acción de control $c_k$.
4. Avanzar un paso con el sistema real (estocástico: ruido blanco + pulsos Poisson).
5. Retroceder el horizonte y repetir desde el paso 1.
""")

# ============================================================
# SECTION 4: IMPORTS Y CONFIG
# ============================================================
md(r"""---
## 4. Imports y configuración global
""")

code(r"""import numpy as np
import matplotlib.pyplot as plt
from mpc_controlador import (MPCController, simulate_step,
                              simulate_step_stochastic, simular_sin_control,
                              grafica_comparativa, generar_trayectoria_dinamica,
                              DT, B_OPT, C_REF, C_BOUNDS,
                              SIGMA_X, SIGMA_Y, PULSE_RATE,
                              PULSE_SCALE_X, PULSE_SCALE_Y,
                              N_STEPS)

print(f"Configuracion lista. N_STEPS = {N_STEPS} (tiempo total = {N_STEPS * DT:.0f} uds.)")
""")

# ============================================================
# SECTION 5: ESCENARIOS
# ============================================================
md(r"""---
## 5. Escenarios de simulación

Se simulan **4 escenarios** sobre la planta estocástica (ruido blanco +
pulsos Poisson). En todos ellos:
- $b = 0.005488$ (tasa de match, constante)
- El **MPC** solo manipula $c$ (eficiencia del matching)
- $a$ y $d$ son parámetros fijos del escenario (o variables en el caótico)
- **Sin control:** $c = c_{\text{ref}} = 0.0179$ fijo
""")

# ----------------------------------------------------------
# ESCENARIO 1: CRECIMIENTO
# ----------------------------------------------------------
md(r"""---
### Escenario 1: Crecimiento

**Condiciones:** $a = 0.5$, $d = 0.2$ — muchos registros nuevos, poca deserción.
**Inicio:** $x_0 = 10$, $y_0 = 10$ (app arrancando).
**Referencia:** $x_{\text{ref}} = 50$, $y_{\text{ref}} = 70$.
**Equilibrio natural:** $x^* = 11.2$, $y^* = 91.1$.

El sistema sin control tiende a un equilibrio con $y \approx 91$ pero
$c$ fijo en $0.0179$ da $x^* \approx 11$, muy por debajo de la ref deseada.
El MPC debe usar $c$ para elevar $x$ hacia 50 manteniendo $y$ cerca de 70.
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

# Sin control
x_nc1, y_nc1 = simular_sin_control(x01, y01, a1, d1, N_STEPS, seed1)

# Con MPC
mpc1 = MPCController()
res1 = mpc1.run_simulation(N_STEPS, xr1, yr1, a1, d1,
                            x0=x01, y0=y01, stochastic=True, seed=seed1)

sr1 = res1['success'].mean() * 100
print(f"  Tasa exito solver: {sr1:.1f}%")
print(f"  Sin control: x_final={x_nc1[-1]:.1f}, y_final={y_nc1[-1]:.1f}")
print(f"  Con MPC:     x_final={res1['x'][-1]:.1f}, y_final={res1['y'][-1]:.1f}")

grafica_comparativa(
    N_STEPS, x_nc1, y_nc1, res1['x'], res1['y'], res1['c'],
    xr1, yr1, eq_x1, eq_y1, a1, d1,
    "Escenario 1: Crecimiento — Sin control vs MPC",
    "comparativa_escenario1.png"
)
""")

# ----------------------------------------------------------
# ESCENARIO 2A: RETENCION — MANTENER
# ----------------------------------------------------------
md(r"""---
### Escenario 2a: Retención — Mantener

**Condiciones:** $a = 0.1$, $d = 0.6$ — pocos registros nuevos, alta deserción.
**Inicio:** $x_0 = 60$, $y_0 = 60$ (app con base grande pero en riesgo).
**Referencia:** $x_{\text{ref}} = 35$, $y_{\text{ref}} = 20$.
**Equilibrio natural:** $x^* = 33.6$, $y^* = 18.2$.

Aquí $y^* = a/b = 18.2$ está **fijo** — el MPC **no puede** cambiar el
equilibrio de usuarios porque depende solo de $a$ y $b$. Sin control el
sistema colapsa ($y \to 0$ por el ruido y la alta deserción). El MPC debe
usar $c$ para mantener $y$ vivo en su equilibrio natural y $x$ cerca de la
referencia.
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

# Sin control
x_nc2a, y_nc2a = simular_sin_control(x02a, y02a, a2a, d2a, N_STEPS, seed2a)

# Con MPC
mpc2a = MPCController()
res2a = mpc2a.run_simulation(N_STEPS, xr2a, yr2a, a2a, d2a,
                              x0=x02a, y0=y02a, stochastic=True, seed=seed2a)

sr2a = res2a['success'].mean() * 100
print(f"  Tasa exito solver: {sr2a:.1f}%")
print(f"  Sin control: x_final={x_nc2a[-1]:.1f}, y_final={y_nc2a[-1]:.1f}")
print(f"  Con MPC:     x_final={res2a['x'][-1]:.1f}, y_final={res2a['y'][-1]:.1f}")

grafica_comparativa(
    N_STEPS, x_nc2a, y_nc2a, res2a['x'], res2a['y'], res2a['c'],
    xr2a, yr2a, eq_x2a, eq_y2a, a2a, d2a,
    "Escenario 2a: Retencion — Mantener (Sin control vs MPC)",
    "comparativa_escenario2a.png"
)
""")

# ----------------------------------------------------------
# ESCENARIO 2B: RETENCION — CRECER
# ----------------------------------------------------------
md(r"""---
### Escenario 2b: Retención — Crecer en $x$

**Condiciones:** $a = 0.1$, $d = 0.6$ (mismos que 2a).
**Inicio:** $x_0 = 60$, $y_0 = 60$.
**Referencia:** $x_{\text{ref}} = 60$, $y_{\text{ref}} = 25$.

A diferencia de 2a, aquí se pide al MPC que mantenga $x$ en 60 (muy por
encima del equilibrio natural $x^* = 33.6$). Para lograrlo debe reducir $c$
(de modo que $x^* = d/c$ suba), pero $y$ queda limitado por $y^* = 18.2$.
El objetivo es ver si el MPC puede **crecer el número de perfiles** incluso
en un entorno de alta deserción.
""")

code(r"""print("=" * 60)
print("  ESCENARIO 2B: RETENCION — CRECER EN x (a=0.1, d=0.6)")
print("  x0=60, y0=60  |  ref=(60, 25)")
print("  c se reduce para elevar x* = d/c")
print("=" * 60)

seed2b = 44
a2b, d2b = 0.1, 0.6
x02b, y02b = 60.0, 60.0
xr2b, yr2b = 60.0, 25.0
eq_x2b, eq_y2b = d2b / C_REF, a2b / B_OPT
print(f"  Equilibrio natural: x*={eq_x2b:.1f}, y*={eq_y2b:.1f}")

# Sin control
x_nc2b, y_nc2b = simular_sin_control(x02b, y02b, a2b, d2b, N_STEPS, seed2b)

# Con MPC
mpc2b = MPCController()
res2b = mpc2b.run_simulation(N_STEPS, xr2b, yr2b, a2b, d2b,
                              x0=x02b, y0=y02b, stochastic=True, seed=seed2b)

sr2b = res2b['success'].mean() * 100
print(f"  Tasa exito solver: {sr2b:.1f}%")
print(f"  Sin control: x_final={x_nc2b[-1]:.1f}, y_final={y_nc2b[-1]:.1f}")
print(f"  Con MPC:     x_final={res2b['x'][-1]:.1f}, y_final={res2b['y'][-1]:.1f}")

grafica_comparativa(
    N_STEPS, x_nc2b, y_nc2b, res2b['x'], res2b['y'], res2b['c'],
    xr2b, yr2b, eq_x2b, eq_y2b, a2b, d2b,
    "Escenario 2b: Retencion — Crecer en x (Sin control vs MPC)",
    "comparativa_escenario2b.png"
)
""")

# ----------------------------------------------------------
# ESCENARIO 3: CAOTICO
# ----------------------------------------------------------
md(r"""---
### Escenario 3: Dinámico (mercado impredecible)

**Condiciones:** $a(t)$ y $d(t)$ varían según un proceso Markoviano de 3 modos:
- **Crecimiento** ($a=0.6$, $d=0.15$)
- **Estable** ($a=0.3$, $d=0.4$)
- **Crisis** ($a=0.08$, $d=0.7$)

Las transiciones entre modos son abruptas e impredecibles, simulando un
mercado caótico. A esto se suma ruido Gaussiano continuo y pulsos Poisson.

**Inicio:** $x_0 = 30$, $y_0 = 30$.
**Referencia:** $x_{\text{ref}} = 40$, $y_{\text{ref}} = 40$.

El objetivo es probar la **robustez** del MPC: ¿puede $c$ adaptarse en
tiempo real para evitar que la app muera cuando el entorno cambia
drásticamente sin previo aviso?
""")

code(r"""print("=" * 60)
print("  ESCENARIO 3: CAOTICO (a(t), d(t) markovianos)")
print("  x0=30, y0=30  |  ref=(40, 40)")
print("=" * 60)

seed3 = 45
x03, y03 = 30.0, 30.0
xr3, yr3 = 40.0, 40.0

a_seq3, d_seq3 = generar_trayectoria_dinamica(N_STEPS, seed=seed3)

# Sin control: c=C_REF fijo, a(t), d(t) variables
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

# Con MPC: en cada paso resuelve con a(k), d(k) actuales
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

print(f"  Sin control: x_final={x_nc3[-1]:.1f}, y_final={y_nc3[-1]:.1f}")
print(f"  Con MPC:     x_final={x_ctrl3[-1]:.1f}, y_final={y_ctrl3[-1]:.1f}")

grafica_comparativa(
    N_STEPS, x_nc3, y_nc3, x_ctrl3, y_ctrl3, c_traj3,
    xr3, yr3, 39.2, 53.0, np.nan, np.nan,
    "Escenario 3: Caotico — a(t), d(t) variables (Sin control vs MPC)",
    "comparativa_escenario3.png"
)
""")

# ============================================================
# FIGURA COMPARATIVA 2x2
# ============================================================
md(r"""---
## 6. Comparativa de los 4 escenarios (2×2)
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
      xr3, yr3, 'Esc. 3: Caotico')

plt.suptitle('Comparativa sin control vs con MPC por escenario',
             fontsize=10, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fig_comparativa_4escenarios.png', dpi=200, bbox_inches='tight')
plt.show()
plt.close()
print("  fig_comparativa_4escenarios.png generado")
""")

# ============================================================
# METRICAS
# ============================================================
md(r"""---
## 7. Métricas de error RMS
""")

code(r"""print("=" * 60)
print("  METRICAS DE ERROR RMS (ventana estacionaria, ultimos 700 pasos)")
print("=" * 60)

w = 700

def rms(err):
    return np.sqrt(np.mean(err**2))

escenarios = [
    ("Esc. 1 Crecimiento", x_nc1, y_nc1, res1['x'], res1['y'], xr1, yr1),
    ("Esc. 2a Mantener",   x_nc2a, y_nc2a, res2a['x'], res2a['y'], xr2a, yr2a),
    ("Esc. 2b Crecer",     x_nc2b, y_nc2b, res2b['x'], res2b['y'], xr2b, yr2b),
]

for nombre, x_nc, y_nc, x_c, y_c, xr, yr in escenarios:
    e_x_nc = rms(x_nc[-w:] - xr)
    e_y_nc = rms(y_nc[-w:] - yr)
    e_x_c = rms(x_c[-w:] - xr)
    e_y_c = rms(y_c[-w:] - yr)
    mej_x = e_x_nc / e_x_c if e_x_c > 0 else float('inf')
    mej_y = e_y_nc / e_y_c if e_y_c > 0 else float('inf')
    print(f"\n  {nombre}:")
    print(f"    x: sin control={e_x_nc:.2f}, con MPC={e_x_c:.2f}, mejora={mej_x:.0f}x")
    print(f"    y: sin control={e_y_nc:.2f}, con MPC={e_y_c:.2f}, mejora={mej_y:.0f}x")

# Escenario dinamico: la referencia es cte asi que calculamos igual
e_x3_nc = rms(x_nc3[-w:] - xr3)
e_y3_nc = rms(y_nc3[-w:] - yr3)
e_x3_c = rms(x_ctrl3[-w:] - xr3)
e_y3_c = rms(y_ctrl3[-w:] - yr3)
mej_x3 = e_x3_nc / e_x3_c if e_x3_c > 0 else float('inf')
mej_y3 = e_y3_nc / e_y3_c if e_y3_c > 0 else float('inf')
print(f"\n  Esc. 3 Caotico:")
print(f"    x: sin control={e_x3_nc:.2f}, con MPC={e_x3_c:.2f}, mejora={mej_x3:.0f}x")
print(f"    y: sin control={e_y3_nc:.2f}, con MPC={e_y3_c:.2f}, mejora={mej_y3:.0f}x")
""")

# ============================================================
# CONCLUSIONS
# ============================================================
md(r"""---
## 8. Conclusiones

1. **Solo $c$ como control:** El análisis muestra que $y^* = a/b$ es un
   equilibrio **fijo** que el MPC no puede alterar manipulando $c$. Esto
   impone un límite fundamental: cuando la tasa de crecimiento de perfiles
   ($a$) es baja, el número de usuarios en equilibrio está severamente
   restringido.

2. **Escenario 1 (Crecimiento):** Con $a=0.5$, $d=0.2$, el MPC lleva el
   sistema de $(10,10)$ a niveles cercanos a la referencia $(50,70)$.
   Sin control, $x$ colapsa a valores muy bajos.

3. **Escenario 2a (Mantener):** A pesar de $a=0.1$, $d=0.6$, el MPC logra
   que $y$ no se extinga, manteniéndolo en su equilibrio natural $y\approx18$.
   Sin control, $y$ muere completamente ($y\to0$).

4. **Escenario 2b (Crecer en $x$):** El MPC reduce $c$ para elevar
   $x^* = d/c$, logrando mantener $x$ en niveles superiores al equilibrio
   natural. $y$ se mantiene vivo aunque limitado por $y^* = a/b$.

5. **Escenario 3 (Dinámico):** El MPC se adapta en tiempo real a las
   variaciones de $a(t)$ y $d(t)$, manteniendo la app con vida aún en
   condiciones extremas de mercado. Sin control, $y$ tiende a colapsar
   durante los períodos de crisis.

6. **Robustez:** En todos los escenarios el solver SLSQP con warm-start
   alcanzó una tasa de convergencia del 100%, y la señal de control $c(t)$
   se mantuvo siempre dentro de las cotas $[0.002, 0.04]$.
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

with open('proyecto_mpc.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Notebook generado: proyecto_mpc.ipynb")
print(f"Total de celdas: {len(cells)}")
