# Dating App Optimization — Modelo Depredador-Presa (Lotka-Volterra)

Optimización no lineal de una app de citas usando el modelo de Lotka-Volterra con 3 optimizadores. Balancea eficiencia del algoritmo, retención de usuarios y crecimiento de perfiles para maximizar rentabilidad manteniendo un equilibrio estable.

## Archivos principales

| Archivo | Descripción |
|---------|-------------|
| `optimizar_app_citas.py` | Clase `DatingAppOptimizer` (DE + L-BFGS-B) |
| `optimizadores.py` | 3 optimizadores: DE, SGD, ANFIS con interfaz unificada |
| `comparacion_escenarios.ipynb` | Notebook: 4 escenarios de app (nueva, crecimiento, establecida, masiva) |
| `comparativa_3_optimizadores.ipynb` | Notebook: comparación de los 3 optimizadores |
| `presentacion.tex` | Presentación Beamer (29 diapositivas, 16:9) |

## Ecuaciones del sistema

```
ẋ = a·x − b·x·y    (perfiles / matches potenciales)
ẏ = c·x·y − d·y    (usuarios activos)
```

**Punto de equilibrio:** `x* = d/c`, `y* = a/b`

**Parámetros del modelo:**

| Parámetro | Nombre | Significado |
|-----------|--------|-------------|
| `a` | alpha (α) | Crecimiento de perfiles (nuevos registros) |
| `b` | beta (β) | Tasa de match (interacción entre usuarios) |
| `c` | delta (δ) | Eficiencia del algoritmo de emparejamiento |
| `d` | gamma (γ) | Abandono de usuarios (tasa de deserción) |

## Optimizadores

### 1. Differential Evolution + L-BFGS-B (`DifferentialEvolutionOptimizer`)
- Búsqueda evolutiva global con población de 50 individuos, 1500 generaciones
- Refinamiento local con L-BFGS-B
- **Costo:** –27.95 (mejor), **CV:** 0.103, **Ganancia:** \$70.60/mes

### 2. SGD (`SGDOptimizer`)
- Gradiente aproximado por diferencias finitas con momentum (0.85)
- Grid de >50 puntos de inicio + reinicios aleatorios
- **Costo:** –25.34, **CV:** 0.041 (más estable), **Ganancia:** \$61.72/mes

### 3. ANFIS (`ANFISOptimizer`)
- Fuzzificación con 3 funciones de membresía Gaussianas por parámetro
- Red neuronal feedforward (12→16→4)
- **Costo:** –12.11, **Retención:** 76.4% (mejor), **Tiempo:** 0.5s (más rápido)

## Métricas de evaluación

| Métrica | Mide | Dirección óptima |
|---------|------|-----------------|
| Costo (J) | Función objetivo multi-objetivo | Más negativo |
| CV | Coeficiente de variación (estabilidad) | Menor (<0.3) |
| Retención | `e^{-d}` (usuarios que no abandonan) | Mayor |
| Ganancia | Ingreso estimado en \$/mes | Mayor |
| Tiempo | Segundos de ejecución | Menor |

## Instalación y uso

```bash
pip install -r requirements.txt

# Clase DatingAppOptimizer
python optimizar_app_citas.py

# Notebooks
jupyter notebook comparacion_escenarios.ipynb
jupyter notebook comparativa_3_optimizadores.ipynb

# Compilar presentación
pdflatex presentacion.tex
```

## Stack

Python, NumPy, SciPy, Matplotlib, LaTeX (Beamer).
