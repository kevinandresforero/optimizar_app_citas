# MPC para control de matching en app de citas

Control Predictivo Basado en Modelo (MPC) para regular la tasa de
emparejamiento de una aplicación de citas, modelada con ecuaciones
de Lotka-Volterra.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `mpc_controlador.py` | Controlador MPC + simulación + gráficas |
| `generar_notebook_completo.py` | Genera el notebook `proyecto_mpc_completo.ipynb` |
| `proyecto_mpc_completo.ipynb` | Notebook completo (sin ejecutar) |
| `proyecto_mpc_completo_ejecutado.ipynb` | Notebook con resultados |
| `reporte_tecnico.tex` | Fuente LaTeX del informe |
| `reporte_tecnico.pdf` | Informe compilado |
| `guion_video.md` | Guion para video de sustentación |
| `requirements.txt` | Dependencias |

## Uso

```bash
pip install -r requirements.txt

# Generar y ejecutar notebook
python generar_notebook_completo.py
jupyter nbconvert --to notebook --execute proyecto_mpc_completo.ipynb

# Compilar informe
pdflatex reporte_tecnico.tex
```
