# Dating App Optimization — Predator-Prey Model (Lotka-Volterra)

Nonlinear optimization of a dating app using the Lotka-Volterra model with 3 optimizers. Balances algorithm efficiency, user retention, and profile growth to maximize profitability while maintaining a stable equilibrium.

## Main files

| File | Description |
|------|-------------|
| `optimizar_app_citas.py` | `DatingAppOptimizer` class (DE + L-BFGS-B) |
| `optimizadores.py` | 3 optimizers: DE, SGD, ANFIS with unified interface |
| `comparacion_escenarios.ipynb` | Notebook: 4 app stages (new, growing, established, massive) |
| `comparativa_3_optimizadores.ipynb` | Notebook: 3-optimizer comparison |
| `presentacion.tex` | Beamer presentation (29 slides, 16:9) |

## System equations

```
ẋ = a·x − b·x·y    (profiles / potential matches)
ẏ = c·x·y − d·y    (active users)
```

**Equilibrium point:** `x* = d/c`, `y* = a/b`

**Model parameters:**

| Parameter | Name | Meaning |
|-----------|------|---------|
| `a` | alpha (α) | Profile growth rate (new registrations) |
| `b` | beta (β) | Match rate (user interaction) |
| `c` | delta (δ) | Algorithm matching efficiency |
| `d` | gamma (γ) | User abandonment rate (churn) |

## Optimizers

### 1. Differential Evolution + L-BFGS-B (`DifferentialEvolutionOptimizer`)
- Global evolutionary search with population of 50, 1500 generations
- Local refinement with L-BFGS-B
- **Cost:** –27.95 (best), **CV:** 0.103, **Revenue:** \$70.60/month

### 2. SGD (`SGDOptimizer`)
- Gradient approximated by finite differences with momentum (0.85)
- Grid of >50 starting points + random restarts
- **Cost:** –25.34, **CV:** 0.041 (most stable), **Revenue:** \$61.72/month

### 3. ANFIS (`ANFISOptimizer`)
- Fuzzification with 3 Gaussian membership functions per parameter
- Feedforward neural network (12→16→4)
- **Cost:** –12.11, **Retention:** 76.4% (best), **Time:** 0.5s (fastest)

## Comparison metrics

| Metric | Measures | Optimal direction |
|--------|----------|-----------------|
| Cost (J) | Multi-objective objective function | More negative |
| CV | Coefficient of variation (stability) | Lower (<0.3) |
| Retention | `e^{-d}` (users who don't churn) | Higher |
| Revenue | Estimated income in \$/month | Higher |
| Time | Execution time in seconds | Lower |

## Setup and usage

```bash
pip install -r requirements.txt

# DatingAppOptimizer class
python optimizar_app_citas.py

# Notebooks
jupyter notebook comparacion_escenarios.ipynb
jupyter notebook comparativa_3_optimizadores.ipynb

# Compile presentation
pdflatex presentacion.tex
```

## Stack

Python, NumPy, SciPy, Matplotlib, LaTeX (Beamer).
