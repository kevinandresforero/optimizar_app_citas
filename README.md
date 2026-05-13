# Dating App Optimization — Predator-Prey Model (Lotka-Volterra)

Nonlinear optimization of a dating app using the Lotka-Volterra model. The algorithm balances matching efficiency (δ), user retention (γ), and profile growth (α) to maximize profitability while maintaining a stable equilibrium.

## Quick start

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd dating-app-optimization

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the optimizer
python optimizar_app_citas.py

# 5. Or launch the comparison notebook
jupyter notebook comparacion_escenarios.ipynb
```

**System equations:**
- `ẋ = αx − βxy` (profiles / potential matches)
- `ẏ = δxy − γy` (active users)

**Equilibrium point:** `x* = γ/δ`, `y* = α/β`

**System parameters:**
- `a` = α (profile growth rate)
- `b` = β (match rate)
- `c` = δ (algorithm efficiency)
- `d` = γ (user abandonment rate)

**Stack:** Python, NumPy, SciPy (optimization), Matplotlib.

## Optimization method

Uses **Differential Evolution** (`scipy.optimize.differential_evolution`) — a gradient-free evolutionary algorithm — followed by local refinement with L-BFGS-B. The cost function `J` minimizes:

| Objective | Target |
|---|---|
| Profile/user ratio | ~0.6 (enough to browse, not overwhelming) |
| Active users | ~55 (profitable base) |
| Stability | Low coefficient of variation (CV) at steady state |
| Retention | Maximize `e^{-d}` (users don't churn) |
| Profit | Ads + premium revenue minus operating costs |
| Fixed-point coherence | Simulated equilibrium matches `x* = d/c`, `y* = a/b` |

The optimal solution is the *only* stable scenario (CV ≈ 0.09) among all evaluated configurations.

## Class-based usage

```python
from optimizar_app_citas import DatingAppOptimizer

opt = DatingAppOptimizer(a=0.3, b=0.006, c=0.018, d=0.7)

# Custom initial conditions
opt.set_initial_conditions(x0=40, y0=50)

# Run global optimization
opt.optimize(n_steps=800)

# Analyse counterfactual scenarios
opt.analyse_scenarios()

# Plot results
opt.plot_results("resultados.png")

# Access optimal parameters
print(opt.a, opt.b, opt.c, opt.d)
```

Multiple independent instances can be created and compared:

```python
for x0, y0 in [(5, 8), (40, 50), (500, 300)]:
    o = DatingAppOptimizer()
    o.set_initial_conditions(x0, y0)
    o.optimize()
    o.plot_results(f"opt_{x0}_{y0}.png")
```

## Notebook

`comparacion_escenarios.ipynb` explores four app stages (new, growing, established, massive) side-by-side with plots and comparison tables.

Run: `python optimizar_app_citas.py`
