# Identificación del Sistema Depredador-Presa — NN y ANFIS

Proyecto de Cibernética III (Entrega Final - Segundo Corte). Identificación basada en datos del modelo Lotka-Volterra usando Redes Neuronales (MLP) y ANFIS.

## Estructura del Proyecto

| Archivo | Descripción |
|---------|-------------|
| `identificacion_sistema.ipynb` | Notebook principal: genera datos, entrena modelos, genera gráficas y tablas de comparación |
| `simulacion.py` | Funciones `simulate()` y `generar_datos()` para simulación del modelo Lotka-Volterra |
| `identificacion.py` | Clases `IdentificadorNN` (MLP) y `IdentificadorANFIS` para identificación de sistemas |
| `paper.tex` | Paper académico en formato IEEE (dos columnas) |
| `poster.tex` / `poster.pdf` | Póster para presentación |
| `qr-code.png` | QR al repositorio |

## Descripción

Se implementa y compara el desempeño de dos enfoques de modelamiento predictivo:

- **MLP (Red Neuronal):** Arquitectura (64, 64, 32), activación ReLU, optimizador Adam con early stopping
- **ANFIS:** 9 funciones de membresía gaussianas por entrada, 81 reglas difusas, regresión Ridge

### Modelo Matemático

Sistema Lotka-Volterra:
```
ẋ = ax - bxy
ẏ = cxy - dy
```

Discretizado con Euler (T=0.1).

### Isomorfismo con App de Citas

| Lotka-Volterra | App de Citas | Valor |
|----------------|--------------|-------|
| Presas (x) | Usuarios activos | - |
| Depredadores (y) | Matches inactivos | - |
| a | α (nuevos perfiles) | 0.5 |
| b | β (match/salida) | 0.01 |
| c | γ (abandono) | 0.2 |
| d | δ (eficiencia) | 0.005 |

## Resultados

| Modelo | MAE (1-paso) | MAE (50 pasos) |
|--------|--------------|----------------|
| MLP | 0.604 | 1.41 |
| ANFIS | 0.898 | 3.95 |

## Requisitos

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Abrir notebook
jupyter notebook identificacion_sistema.ipynb

# Compilar paper
pdflatex paper.tex
pdflatex paper.tex

# Compilar póster
pdflatex poster.tex
pdflatex poster.tex
```

## Reproducibilidad

Todos los experimentos usan semilla fija (`seed=42`) para garantizar resultados reproducibles.

## Repositorio

\url{https://github.com/kevinandresforero/optimizar_app_citas/tree/main}