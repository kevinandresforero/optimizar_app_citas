# Dating App Optimization — Predator-Prey Model (Lotka-Volterra)

Nonlinear optimization of a dating app using the Lotka-Volterra model. The algorithm balances matching efficiency (δ), user retention (γ), and profile growth (α) to maximize profitability while maintaining a stable equilibrium.

**System equations:**
- `ẋ = αx − βxy` (profiles / potential matches)
- `ẏ = δxy − γy` (active users)

**Equilibrium point:** `x* = γ/δ`, `y* = α/β`

**Stack:** Python, NumPy, SciPy (optimization), Matplotlib.

Run: `python optimizar_app_citas.py`
