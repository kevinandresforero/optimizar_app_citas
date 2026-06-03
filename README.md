# MPC para control de matching en app de citas

Control Predictivo Basado en Modelo (MPC) para regular la tasa de
emparejamiento de una aplicación de citas, modelada con ecuaciones
de Lotka-Volterra ($x$: perfiles, $y$: usuarios, $c$: eficiencia
del matching).

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `mpc_controlador.py` | Controlador MPC + simulación + gráficas |
| `generar_notebook_completo.py` | Genera el notebook de resultados |
| `proyecto_mpc_completo.ipynb` | Notebook completo (sin ejecutar) |
| `proyecto_mpc_completo_ejecutado.ipynb` | Notebook con resultados |
| `reporte_tecnico.tex` | Fuente LaTeX del informe |
| `reporte_tecnico.pdf` | Informe compilado |
| `requirements.txt` | Dependencias |

## Arquitectura del controlador

Las ecuaciones de Lotka-Volterra y el controlador MPC están
implementados en `mpc_controlador.py`, que contiene:

- **Modelo de la planta:** Ecuaciones diferenciales discretizadas
  con Euler (`simulate_step`, `simulate_step_stochastic`).
- **Clase `MPCController`:** Controlador predictivo que optimiza
  solo $c$ (matching) con $a$ y $d$ como parámetros fijos.
  - Horizonte de predicción $N=15$, control $M=10$.
  - Función de costo cuadrática con pesos $Q=\text{diag}(1,1)$,
    $r_c=0.1$ y cotas $c\in[0.002, 0.04]$.
  - Solver SLSQP con warm-start (desplazamiento de la solución
    anterior).
- **Generación de datos aleatorios:** Ruido Gaussiano ($\sigma_x=0.5$,
  $\sigma_y=0.3$), pulsos Poisson ($\lambda=0.001$), y trayectorias
  Markovianas de 3 modos para $a(t)$ y $d(t)$.
- **Escenarios:** 4 configuraciones (crecimiento, mantener, crecer
  en $x$, dinámico) que usan la **misma instancia** del controlador.

## Uso

```bash
pip install -r requirements.txt

# Generar y ejecutar notebook
python generar_notebook_completo.py
jupyter nbconvert --to notebook --execute proyecto_mpc_completo.ipynb

# Compilar informe
pdflatex reporte_tecnico.tex
```
