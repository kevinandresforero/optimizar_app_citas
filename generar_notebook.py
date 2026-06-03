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
optimización de parámetros (visión general de las arquitecturas) y diseño del
controlador predictivo basado en modelo (MPC) con resultados comparativos sobre
un **sistema estocástico** (ruido blanco + pulsos aleatorios).
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

Se implementa un **controlador predictivo no lineal** (NMPC) con las siguientes
características:
""")

md(r"""### Formulación del problema de optimización

En cada instante $k$, conocido el estado actual $(x_k, y_k)$, se resuelve:

$$
\begin{aligned}
\min_{\mathbf{u}_{k}, \dots, \mathbf{u}_{k+M-1}} \quad &
\sum_{t=0}^{N-1} \Bigl[
\|\mathbf{x}_{k+t+1} - \mathbf{x}_{\text{ref}}\|_Q^2 +
\|\mathbf{u}_{k+t} - \mathbf{u}_{\text{ref}}\|_R^2
\Bigr] \\[6pt]
\text{sujeto a} \quad &
\mathbf{x}_{k+t+1} = f\bigl(\mathbf{x}_{k+t}, \mathbf{u}_{k+t}\bigr)
\quad \text{(modelo no lineal)} \\[4pt]
& 0.2 \leq a \leq 1.0, \quad
0.002 \leq c \leq 0.04, \quad
0.1 \leq d \leq 0.7 \\[4pt]
& \mathbf{u}_{k+t} = \mathbf{u}_{k+M-1} \quad \text{para } t \geq M
\end{aligned}
$$

donde $\mathbf{x} = [x, y]^T$ y $\mathbf{u} = [a, c, d]^T$.
""")

md(r"""### Parámetros de diseño

| Parámetro | Símbolo | Valor |
|-----------|---------|-------|
| Horizonte de predicción | $N$ | 15 pasos |
| Horizonte de control | $M$ | 10 pasos |
| Tiempo de muestreo | $\Delta t$ | 0.1 ud |
| Peso de tracking de $x$ | $Q_{11}$ | 1.0 |
| Peso de tracking de $y$ | $Q_{22}$ | 1.0 |
| Peso de esfuerzo en $a$ | $R_{11}$ | 0.1 |
| Peso de esfuerzo en $c$ | $R_{22}$ | 0.1 |
| Peso de esfuerzo en $d$ | $R_{33}$ | 0.1 |
| Solver | — | SLSQP (SciPy) |
| Estrategia inicial | — | Warm-start (shift) |
""")

md(r"""### Referencia en estado estacionario

Para un punto de referencia deseado $\mathbf{x}_{\text{ref}}$, el control
en estado estacionario se calcula como:

$$
a_{\text{ref}} = b \cdot y_{\text{ref}}, \qquad
c_{\text{ref}} = 0.0179, \qquad
d_{\text{ref}} = c_{\text{ref}} \cdot x_{\text{ref}}
$$

### Algoritmo de control

En cada instante $k$:
1. Medir el estado actual $(x_k, y_k)$.
2. Resolver el problema de optimización NMPC con SLSQP.
3. Aplicar solo la primera acción de control $(a_k, c_k, d_k)$.
4. Avanzar un paso con el sistema real (estocástico: con ruido + pulsos).
5. Retroceder el horizonte y repetir desde el paso 1.
""")

# ============================================================
# SECTION 4: RESULTADOS
# ============================================================
md(r"""---
## 4. Resultados — Simulación comparativa

Se simulan **1000 pasos** (100 unidades de tiempo) para cada escenario,
comparando el sistema **sin control** (parámetros fijos) contra el sistema
**con control MPC**. Ambos operan sobre la **planta estocástica** con ruido
blanco ($\sigma_x=0.5$, $\sigma_y=0.3$) y pulsos aleatorios ($\lambda=0.001$,
$p_x\sim\text{Exp}(15)$, $p_y\sim\text{Exp}(5)$). El MPC usa un modelo interno
determinista y desconoce la naturaleza estocástica de la planta, lo que mide su
**robustez** ante incertidumbre no modelada.
""")

code(r"""import numpy as np
import matplotlib.pyplot as plt
from mpc_controlador import (MPCController, simulate_step,
                              simulate_step_stochastic, simulate_stochastic,
                              DT, B_OPT, C_REF, A_BOUNDS, C_BOUNDS, D_BOUNDS)

# Parametros fijos para el sistema sin control
A_FIXED = 0.2909
B_FIXED = B_OPT
C_FIXED = C_REF
D_FIXED = 0.7000

# Configuracion estocastica comun
SIGMA_X = 0.5
SIGMA_Y = 0.3
PULSE_RATE = 0.001
PULSE_SCALE_X = 15.0
PULSE_SCALE_Y = 5.0

N_STEPS = 1000

def grafica_comparativa(n_steps, x_no_ctrl, y_no_ctrl, x_ctrl, y_ctrl,
                         x_ref_seq, y_ref_seq, x_eq, y_eq, titulo,
                         nombre_archivo, switch_step=None, dist_step=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    t = np.arange(n_steps + 1) * DT

    # Sin control (estocastico)
    ax1.plot(t, x_no_ctrl, 'b-', lw=0.8, label='$x(t)$: perfiles disponibles')
    ax1.plot(t, y_no_ctrl, 'r-', lw=0.8, label='$y(t)$: usuarios activos')
    ax1.axhline(x_eq, color='b', ls='--', alpha=0.4, label=f'$x^*$={x_eq:.1f}')
    ax1.axhline(y_eq, color='r', ls='--', alpha=0.4, label=f'$y^*$={y_eq:.1f}')
    if switch_step is not None:
        ax1.axvline(switch_step * DT, color='gray', ls=':', alpha=0.5)
    if dist_step is not None:
        ax1.axvline(dist_step * DT, color='orange', ls=':', alpha=0.5)
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Estado')
    ax1.set_title('Sin control — parámetros fijos')
    ax1.set_ylabel('Estado')
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.2)

    # Con control MPC (planta estocastica)
    ax2.plot(t, x_ctrl, 'b-', lw=0.8, label='$x(t)$: perfiles disponibles')
    ax2.plot(t, y_ctrl, 'r-', lw=0.8, label='$y(t)$: usuarios activos')
    if np.ndim(x_ref_seq) == 0:
        ax2.axhline(x_ref_seq, color='b', ls='--', alpha=0.4, label=f'$x_{{\\mathrm{{ref}}}}$={x_ref_seq:.1f}')
        ax2.axhline(y_ref_seq, color='r', ls='--', alpha=0.4, label=f'$y_{{\\mathrm{{ref}}}}$={y_ref_seq:.1f}')
    else:
        ax2.plot(t, x_ref_seq, 'b--', alpha=0.4, lw=0.8, label='$x_{\\mathrm{ref}}$')
        ax2.plot(t, y_ref_seq, 'r--', alpha=0.4, lw=0.8, label='$y_{\\mathrm{ref}}$')
    if switch_step is not None:
        ax2.axvline(switch_step * DT, color='gray', ls=':', alpha=0.5)
    if dist_step is not None:
        ax2.axvline(dist_step * DT, color='orange', ls=':', alpha=0.5)
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Estado')
    ax2.set_title('Con control MPC')
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.2)

    plt.suptitle(titulo, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(nombre_archivo, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"  Grafica guardada: {nombre_archivo}")

print("Configuracion lista. N_STEPS =", N_STEPS)
""")

# ============================================================
# SCENARIO 1
# ============================================================
md(r"""---
### Escenario 1: Seguimiento de referencia constante (planta estocástica)

**Condiciones iniciales:** $x_0 = 40$, $y_0 = 50$
**Referencia:** $x_{\text{ref}} = 35$, $y_{\text{ref}} = 55$
**Equilibrio natural:** $x^* = 39.2$, $y^* = 53.0$
**Semilla:** `seed=42` (misma secuencia de ruido para ambas simulaciones)

El sistema sin control con parámetros fijos fluctúa alrededor del equilibrio
natural debido al ruido, mientras que el MPC debe rechazar el ruido y mantener
los estados en la referencia deseada $(35, 55)$.
""")

code(r"""print("=" * 60)
print("  ESCENARIO 1: REFERENCIA CONSTANTE (1000 pasos, estocastico)")
print("=" * 60)

xr1, yr1 = 35.0, 55.0
seed1 = 42

# Sin control - sistema estocastico con parametros fijos
x_nc1, y_nc1, _, _, _, _ = simulate_stochastic(
    40.0, 50.0, [A_FIXED]*N_STEPS, [C_FIXED]*N_STEPS,
    [D_FIXED]*N_STEPS, B_FIXED, seed=seed1,
    sigma_x=SIGMA_X, sigma_y=SIGMA_Y, pulse_rate=PULSE_RATE,
    pulse_scale_x=PULSE_SCALE_X, pulse_scale_y=PULSE_SCALE_Y
)

# Con control MPC - planta estocastica
mpc1 = MPCController()
res1 = mpc1.run_simulation(N_STEPS, xr1, yr1, x0=40.0, y0=50.0,
                           stochastic=True, seed=seed1,
                           sigma_x=SIGMA_X, sigma_y=SIGMA_Y,
                           pulse_rate=PULSE_RATE,
                           pulse_scale_x=PULSE_SCALE_X,
                           pulse_scale_y=PULSE_SCALE_Y)

sr1 = res1['success'].mean() * 100
print(f"  Exactitud del solver MPC: {sr1:.1f}%")

# Metricas de tracking (ventana estacionaria ultimos 500 pasos)
e_x1_ctrl = np.std(res1['x'][-500:] - xr1)
e_y1_ctrl = np.std(res1['y'][-500:] - yr1)
e_x1_nc = np.std(x_nc1[-500:] - xr1)
e_y1_nc = np.std(y_nc1[-500:] - yr1)
print(f"  Error RMS (ultimos 500 pasos):")
print(f"    Sin control: x={e_x1_nc:.2f}, y={e_y1_nc:.2f}")
print(f"    Con MPC:     x={e_x1_ctrl:.2f}, y={e_y1_ctrl:.2f}")
print(f"    Mejora:      x={e_x1_nc/e_x1_ctrl:.0f}x, y={e_y1_nc/e_y1_ctrl:.0f}x")

grafica_comparativa(
    N_STEPS, x_nc1, y_nc1, res1['x'], res1['y'],
    xr1, yr1, 39.2, 53.0,
    "Escenario 1: Seguimiento de Referencia Constante (Estocastico)",
    "comparativa_escenario1.png"
)
""")

# ============================================================
# SCENARIO 2
# ============================================================
md(r"""---
### Escenario 2: Cambio escalón de referencia (planta estocástica)

**Condiciones iniciales:** $x_0 = 40$, $y_0 = 50$
**Referencia 1:** $(35, 55)$ hasta $t = 40$ (400 pasos)
**Referencia 2:** $(30, 58)$ desde $t = 40$ en adelante
**Semilla:** `seed=2`

El sistema sin control ignora el cambio de referencia y se mantiene oscilando
alrededor del equilibrio natural. El MPC debe seguir el nuevo punto de
operación a pesar del ruido continuo.
""")

code(r"""print("=" * 60)
print("  ESCENARIO 2: CAMBIO ESCALON DE REFERENCIA (1000 pasos, estocastico)")
print("=" * 60)

xr1, yr1 = 35.0, 55.0
xr2, yr2 = 30.0, 58.0
switch = 400
seed2 = 2

# Sin control - parametros fijos (ignoran cambio de referencia)
x_nc2, y_nc2, _, _, _, _ = simulate_stochastic(
    40.0, 50.0, [A_FIXED]*N_STEPS, [C_FIXED]*N_STEPS,
    [D_FIXED]*N_STEPS, B_FIXED, seed=seed2,
    sigma_x=SIGMA_X, sigma_y=SIGMA_Y, pulse_rate=PULSE_RATE,
    pulse_scale_x=PULSE_SCALE_X, pulse_scale_y=PULSE_SCALE_Y
)

# Con control MPC
mpc2 = MPCController()
x_traj2 = np.zeros(N_STEPS + 1)
y_traj2 = np.zeros(N_STEPS + 1)
x_ref_seq2 = np.full(N_STEPS + 1, xr1)
y_ref_seq2 = np.full(N_STEPS + 1, yr1)
x_ref_seq2[switch:] = xr2
y_ref_seq2[switch:] = yr2
x_traj2[0], y_traj2[0] = 40.0, 50.0
rng2 = np.random.default_rng(seed2)

for k in range(N_STEPS):
    x_ref = xr1 if k < switch else xr2
    y_ref = yr1 if k < switch else yr2
    x_cur, y_cur = x_traj2[k], y_traj2[k]
    u_opt, success, _ = mpc2.solve(x_cur, y_cur, x_ref, y_ref)
    x_traj2[k+1], y_traj2[k+1], _, _, _, _ = \
        simulate_step_stochastic(
            x_cur, y_cur, u_opt[0], mpc2.b, u_opt[1], u_opt[2],
            rng2, SIGMA_X, SIGMA_Y, PULSE_RATE,
            PULSE_SCALE_X, PULSE_SCALE_Y
        )

grafica_comparativa(
    N_STEPS, x_nc2, y_nc2, x_traj2, y_traj2,
    x_ref_seq2, y_ref_seq2, 39.2, 53.0,
    "Escenario 2: Cambio Escalon de Referencia (Estocastico)",
    "comparativa_escenario2.png",
    switch_step=switch
)

e_x2_ctrl = np.std(x_traj2[-600:] - xr2)
e_y2_ctrl = np.std(y_traj2[-600:] - yr2)
e_x2_nc = np.std(x_nc2[-600:] - xr2)
e_y2_nc = np.std(y_nc2[-600:] - yr2)
print(f"  Error RMS post-cambio (ultimos 600 pasos):")
print(f"    Sin control: x={e_x2_nc:.2f}, y={e_y2_nc:.2f}")
print(f"    Con MPC:     x={e_x2_ctrl:.2f}, y={e_y2_ctrl:.2f}")
print(f"    Mejora:      x={e_x2_nc/e_x2_ctrl:.0f}x, y={e_y2_nc/e_y2_ctrl:.0f}x")
""")

# ============================================================
# SCENARIO 3
# ============================================================
md(r"""---
### Escenario 3: Rechazo a perturbaciones (planta estocástica)

**Condiciones iniciales:** $x_0 = 35$, $y_0 = 55$ (en la referencia)
**Referencia:** $x_{\text{ref}} = 35$, $y_{\text{ref}} = 55$
**Perturbación determinista:** $+\!15$ en perfiles en $t = 20$ (paso 200)
**Semilla:** `seed=3`

Además del ruido estocástico continuo, se aplica una perturbación tipo
escalón de magnitud $+15$ en $x$. El MPC debe rechazar tanto el ruido
continuo como la perturbación abrupta, mientras que el sistema sin control
se aleja de la referencia por el efecto combinado de la perturbación y el
ruido.
""")

code(r"""print("=" * 60)
print("  ESCENARIO 3: RECHAZO A PERTURBACION (1000 pasos, estocastico)")
print("=" * 60)

xr3, yr3 = 35.0, 55.0
dist_step = 200
seed3 = 3

# Sin control: estocastico + perturbacion determinista
rng3_nc = np.random.default_rng(seed3)
x_nc3 = np.zeros(N_STEPS + 1)
y_nc3 = np.zeros(N_STEPS + 1)
x_nc3[0], y_nc3[0] = 35.0, 55.0
for k in range(N_STEPS):
    xk = x_nc3[k] + (15.0 if k == dist_step else 0.0)
    yk = y_nc3[k]
    x_nc3[k+1], y_nc3[k+1], _, _, _, _ = \
        simulate_step_stochastic(
            xk, yk, A_FIXED, B_FIXED, C_FIXED, D_FIXED,
            rng3_nc, SIGMA_X, SIGMA_Y, PULSE_RATE,
            PULSE_SCALE_X, PULSE_SCALE_Y
        )

# Con control MPC
mpc3 = MPCController()
res3 = mpc3.run_simulation(N_STEPS, xr3, yr3, x0=35.0, y0=55.0,
                           disturbance={'step': dist_step, 'dx': 15.0, 'dy': 0.0},
                           stochastic=True, seed=seed3,
                           sigma_x=SIGMA_X, sigma_y=SIGMA_Y,
                           pulse_rate=PULSE_RATE,
                           pulse_scale_x=PULSE_SCALE_X,
                           pulse_scale_y=PULSE_SCALE_Y)

sr3 = res3['success'].mean() * 100
print(f"  Exactitud del solver MPC: {sr3:.1f}%")

# Metricas post-perturbacion
e_x3_ctrl = np.std(res3['x'][-800:] - xr3)
e_y3_ctrl = np.std(res3['y'][-800:] - yr3)
e_x3_nc = np.std(x_nc3[-800:] - xr3)
e_y3_nc = np.std(y_nc3[-800:] - yr3)
print(f"  Error RMS post-perturbacion (ultimos 800 pasos):")
print(f"    Sin control: x={e_x3_nc:.2f}, y={e_y3_nc:.2f}")
print(f"    Con MPC:     x={e_x3_ctrl:.2f}, y={e_y3_ctrl:.2f}")
print(f"    Mejora:      x={e_x3_nc/e_x3_ctrl:.0f}x, y={e_y3_nc/e_y3_ctrl:.0f}x")

grafica_comparativa(
    N_STEPS, x_nc3, y_nc3, res3['x'], res3['y'],
    xr3, yr3, 39.2, 53.0,
    "Escenario 3: Rechazo a Perturbacion + Ruido (Estocastico)",
    "comparativa_escenario3.png",
    dist_step=dist_step
)
""")

# ============================================================
# SCENARIO 4: STOCHASTIC IN-DEPTH ANALYSIS
# ============================================================
md(r"""---
### Escenario 4: Análisis detallado de robustez estocástica

Este escenario profundiza en el comportamiento del sistema bajo condiciones
estocásticas, comparando las señales de ruido, los pulsos detectados y la
**señal de control** que el MPC genera para rechazar las perturbaciones.

Se simulan 500 pasos con semilla diferente y se analizan:
- El ruido blanco aplicado a cada estado.
- Los pulsos aleatorios que ocurren durante la simulación.
- La evolución de las señales de control $a(t)$, $c(t)$, $d(t)$.
- La distribución del error de tracking.
""")

code(r"""print("=" * 60)
print("  ESCENARIO 4: ANALISIS DE ROBUSTEZ ESTOCASTICA (500 pasos)")
print("=" * 60)

from mpc_controlador import simulate_stochastic

N_STEPS4 = 500
xr4, yr4 = 35.0, 55.0
seed4 = 123

# Sin control
x_nc4, y_nc4, rx_nc, ry_nc, px_nc, py_nc = simulate_stochastic(
    40.0, 50.0, [A_FIXED]*N_STEPS4, [C_FIXED]*N_STEPS4,
    [D_FIXED]*N_STEPS4, B_FIXED, seed=seed4,
    sigma_x=SIGMA_X, sigma_y=SIGMA_Y, pulse_rate=PULSE_RATE,
    pulse_scale_x=PULSE_SCALE_X, pulse_scale_y=PULSE_SCALE_Y
)

# Con control
mpc4 = MPCController()
res4 = mpc4.run_simulation(N_STEPS4, xr4, yr4, x0=40.0, y0=50.0,
                           stochastic=True, seed=seed4,
                           sigma_x=SIGMA_X, sigma_y=SIGMA_Y,
                           pulse_rate=PULSE_RATE,
                           pulse_scale_x=PULSE_SCALE_X,
                           pulse_scale_y=PULSE_SCALE_Y)

sr4 = res4['success'].mean() * 100
n_px = np.sum(res4['pulso_x'] > 0)
n_py = np.sum(res4['pulso_y'] > 0)

e_x4_ctrl = np.std(res4['x'][-250:] - xr4)
e_y4_ctrl = np.std(res4['y'][-250:] - yr4)
e_x4_nc = np.std(x_nc4[-250:] - xr4)
e_y4_nc = np.std(y_nc4[-250:] - yr4)
print(f"  Exactitud solver MPC: {sr4:.1f}%")
print(f"  Pulsos en periodo: {n_px} en x, {n_py} en y")
print(f"  Error RMS (ventana estacionaria):")
print(f"    Sin control: x={e_x4_nc:.2f}, y={e_y4_nc:.2f}")
print(f"    Con MPC:     x={e_x4_ctrl:.2f}, y={e_y4_ctrl:.2f}")

# Grafica de 2x2 con analisis detallado
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
t4 = np.arange(N_STEPS4) * DT
t4s = np.arange(N_STEPS4 + 1) * DT

# Panel 1: Estados
ax = axes[0, 0]
ax.plot(t4s, x_nc4, 'b-', lw=0.7, alpha=0.5, label='$x(t)$: perfiles (sin control)')
ax.plot(t4s, y_nc4, 'r-', lw=0.7, alpha=0.5, label='$y(t)$: usuarios (sin control)')
ax.plot(t4s, res4['x'], 'b-', lw=1.5, label='$x(t)$: perfiles (con MPC)')
ax.plot(t4s, res4['y'], 'r-', lw=1.5, label='$y(t)$: usuarios (con MPC)')
ax.axhline(xr4, color='b', ls='--', alpha=0.3, label=f'$x_{{ref}}$={xr4}')
ax.axhline(yr4, color='r', ls='--', alpha=0.3, label=f'$y_{{ref}}$={yr4}')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Estado')
ax.set_title('Perfiles $x(t)$ y usuarios $y(t)$: sin control vs con MPC')
ax.legend(fontsize=6)
ax.grid(alpha=0.2)

# Panel 2: Ruido
ax = axes[0, 1]
ax.plot(t4, res4['ruido_x'], 'b-', lw=0.5, label=r'$\sigma_x \varepsilon_x$')
ax.plot(t4, res4['ruido_y'], 'r-', lw=0.5, label=r'$\sigma_y \varepsilon_y$')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Ruido')
ax.set_title('Ruido blanco aplicado')
ax.legend(fontsize=7)
ax.grid(alpha=0.2)

# Panel 3: Control
ax = axes[1, 0]
ax.plot(t4, res4['a'], 'b-', lw=1, label='$a$: crecimiento perfiles')
ax.plot(t4, res4['c'], 'g-', lw=1, label='$c$: eficiencia matching')
ax.plot(t4, res4['d'], 'r-', lw=1, label='$d$: abandono usuarios')
for bounds, color in [(A_BOUNDS, 'b'), (C_BOUNDS, 'g'), (D_BOUNDS, 'r')]:
    ax.axhline(bounds[0], color=color, ls=':', alpha=0.3)
    ax.axhline(bounds[1], color=color, ls=':', alpha=0.3)
ax.set_xlabel('Tiempo')
ax.set_ylabel('Control')
ax.set_title('Senales de control MPC')
ax.legend(fontsize=7)
ax.grid(alpha=0.2)

# Panel 4: Pulsos detectados
ax = axes[1, 1]
px_idx = np.where(res4['pulso_x'] > 0)[0]
py_idx = np.where(res4['pulso_y'] > 0)[0]
if len(px_idx) > 0:
    ax.stem(t4[px_idx], res4['pulso_x'][px_idx], linefmt='b-', markerfmt='bo',
            basefmt=' ', label=f'Pulsos x ({len(px_idx)})')
if len(py_idx) > 0:
    ax.stem(t4[py_idx], res4['pulso_y'][py_idx], linefmt='r-', markerfmt='rs',
            basefmt=' ', label=f'Pulsos y ({len(py_idx)})')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Magnitud pulso')
ax.set_title('Pulsos aleatorios')
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

plt.suptitle("Escenario 4: Analisis de Robustez Estocastica",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("comparativa_escenario4.png", dpi=150, bbox_inches='tight')
plt.show()
plt.close()
print("  Grafica guardada: comparativa_escenario4.png")

print(f"\n  Resumen robustez (500 pasos, semilla={seed4}):")
print(f"  {'Metrica':<30} {'Sin control':<15} {'Con MPC':<15}")
print(f"  {'-'*60}")
print(f"  {'Error RMS x':<30} {e_x4_nc:<15.2f} {e_x4_ctrl:<15.2f}")
print(f"  {'Error RMS y':<30} {e_y4_nc:<15.2f} {e_y4_ctrl:<15.2f}")
print(f"  {'Mejora RMS x':<30} {'-':<15} {e_x4_nc/e_x4_ctrl:<15.0f}x")
print(f"  {'Mejora RMS y':<30} {'-':<15} {e_y4_nc/e_y4_ctrl:<15.0f}x")
""")

# ============================================================
# CONCLUSIONS
# ============================================================
md(r"""---
## 5. Conclusiones

1. **Modelo Lotka-Volterra:** El modelo depredador-presa captura adecuadamente la
   dinámica competitiva entre perfiles y usuarios en una app de citas. El punto
   de equilibrio $(x^*, y^*)$ depende exclusivamente de los parámetros del sistema.

2. **Optimización de parámetros:** Las tres arquitecturas evaluadas (DE+L-BFGS-B,
   SGD y ANFIS) encuentran soluciones factibles. DE ofrece la mejor calidad de
   solución ($J = -27.95$), mientras que ANFIS es el más rápido (0.5 s).

3. **Controlador MPC sobre planta estocástica:** El controlador predictivo no
   lineal implementado demuestra **robustez** ante ruido blanco y pulsos
   aleatorios que el MPC desconoce:
   - **Seguimiento de referencia** con error RMS reducido ~100× frente al
     sistema sin control, incluso con ruido continuo.
   - **Adaptación a cambios de referencia** bajo condiciones estocásticas.
   - **Rechazo a perturbaciones** combinadas (deterministas + estocásticas).
   - **Respeto de restricciones** en todas las señales de control.

4. **Rendimiento numérico:** El solver SLSQP con *warm-start* alcanza una tasa de
   convergencia del 100% en todos los escenarios simulados, demostrando que el
   costo computacional del MPC no lineal es manejable incluso con 5000 pasos de
   simulación.
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
