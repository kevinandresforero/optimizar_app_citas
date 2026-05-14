# Identificación del Sistema Depredador-Presa — NN y ANFIS

Segunda parte del proyecto de Cibernética III. Identificación basada en datos del modelo Lotka-Volterra usando Redes Neuronales (NN) y ANFIS.

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `simulacion.py` | Generación de datos por simulación del modelo Lotka-Volterra |
| `identificacion.py` | Clases `IdentificadorNN` y `IdentificadorANFIS` para identificación de sistema |
| `identificacion_sistema.ipynb` | Notebook principal: genera datos, entrena modelos, compara y calcula métricas |
| `presentacion.tex` | Paper en formato artículo |
| `Proyecto_2__ciber_3.pdf` | Enunciado de la parte 2 |

## Requisitos

```
pip install -r requirements.txt
```

## Reproducibilidad

Todos los experimentos usan semilla fija (`seed=42`) para garantizar resultados reproducibles.

## Uso

```bash
jupyter notebook identificacion_sistema.ipynb

# Compilar paper
pdflatex presentacion.tex
pdflatex presentacion.tex
```

## Métricas de error

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
