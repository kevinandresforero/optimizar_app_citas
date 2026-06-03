# Guion para video expositivo — MPC aplicado a app de citas

**Duración estimada:** 11–13 minutos

---

## 1. Introducción (≈1 min)

**Qué mostrar:** Pantalla en blanco o portada del PDF/notebook.

**Qué decir:**

> Este trabajo aplica un controlador predictivo basado en modelo (MPC)
> para regular la tasa de emparejamiento de una aplicación de citas.
>
> Usamos el modelo Lotka-Volterra, clásicamente usado para describir
> la interacción depredador-presa en ecología. En nuestra analogía,
> los perfiles disponibles —las personas registradas en la app— son
> las presas, y los usuarios activos —los que interactúan— son los
> depredadores.
>
> La variable de control es la eficiencia del sistema de matching,
> que llamamos $c$. El objetivo es ajustar $c$ en tiempo real para
> mantener la aplicación saludable.

---

## 2. Modelo matemático (≈1.5 min)

**Qué mostrar:** Las ecuaciones del modelo en el PDF o escritas en
una pizarra/diapositiva.

**Qué decir:**

> Las ecuaciones de Lotka-Volterra que gobiernan el sistema son:
>
>   $$\dot{x} = a x - b x y$$
>   $$\dot{y} = c x y - d y$$
>
> Donde:
> - $x$ son los perfiles disponibles (presa)
> - $y$ son los usuarios activos (depredador)
> - $a$ es la tasa de nuevos registros de perfiles
> - $b$ es la tasa de matches por unidad de tiempo
> - $c$ es la eficiencia del sistema de matching (nuestra variable de control)
> - $d$ es la tasa de abandono de usuarios
>
> El punto de equilibrio natural es:
>   $$x^* = \frac{d}{c}, \qquad y^* = \frac{a}{b}$$
>
> Noten algo importante: $y^* = a/b$ depende únicamente de $a$ y $b$,
> no de $c$. Esto significa que hay un límite fundamental en el
> número de usuarios que podemos sostener, independientemente de
> cuánto ajustemos el matching. El MPC solo puede influir en $x^*$
> a través de $c$.

---

## 3. Ruido y datos aleatorios (≈1 min)

**Qué mostrar:** El código `simulate_step_stochastic` y
`generar_trayectoria_dinamica` en el notebook.

**Qué decir:**

> Para hacer la simulación más realista, añadimos dos tipos de
> perturbaciones aleatorias a la planta:
>
> Primero, **ruido blanco Gaussiano** con desviaciones de 0.5 para
> $x$ y 0.3 para $y$. Esto simula las variaciones diarias normales
> en registros y actividad de usuarios.
>
> Segundo, **pulsos Poisson** con tasa $\lambda = 0.001$ y
> magnitudes exponenciales. Esto modela eventos atípicos como
> campañas virales, caídas del servidor o tendencias estacionales.
>
> En el escenario 3, además, los parámetros $a$ y $d$ evolucionan
> mediante una **cadena de Markov de 3 modos**: crecimiento,
> estable y crisis. Cada modo tiene transiciones abruptas cada
> 30 a 60 pasos aproximadamente, simulando el comportamiento
> dinámico de un sistema real de mercado.

---

## 4. Arquitectura del MPC (≈3 min)

**Qué mostrar:** Las secciones de la clase `MPCController` en el
código (`mpc_controlador.py`) y la tabla de parámetros en el PDF,
pasando por cada método uno a uno.

---

### 4.1. ¿Qué es un MPC? — Filosofía general

**Qué decir:**

> El control predictivo basado en modelo (MPC) no es un controlador
> tradicional con una fórmula de realimentación fija. En cada paso
> de tiempo, el MPC **resuelve un problema de optimización en
> línea**: predice el comportamiento futuro del sistema durante un
> horizonte $N$ usando un modelo interno, y selecciona la secuencia
> de acciones de control que minimiza una función de costo.
>
> Solo aplica la primera acción de esa secuencia, luego desplaza
> el horizonte hacia adelante y repite todo el proceso. Esto se
> conoce como **horizonte deslizante** o *receding horizon*.
>
> La gran ventaja del MPC frente a un controlador clásico como un
> PID es que maneja de forma natural:
> - **Restricciones** en la variable de control (por ejemplo, $c$
>   no puede ser negativa ni exceder la capacidad del sistema)
> - **No linealidades** del modelo (las ecuaciones de Lotka-Volterra
>   son bilineales por el término $x y$)
> - **Múltiples variables** con pesos diferenciados ($x$ e $y$)

---

### 4.2. Nuestra implementación — Vista general de la clase

**Qué mostrar:** El `__init__` de `MPCController`.

**Qué decir:**

> Nuestra implementación está en la clase `MPCController` dentro de
> `mpc_controlador.py`. El constructor recibe solo tres parámetros:
> el peso del control $r_c$, las cotas de $c$ y el peso $b$.
>
> Pero noten algo importante: **no recibe $a$ ni $d$ en el
> constructor**. ¿Por qué? Porque $a$ y $d$ son parámetros del
> escenario de mercado, no de la arquitectura del controlador.
> Se pasan en tiempo de ejecución, en cada llamada a `solve` o
> `run_simulation`. Esto permite que la **misma instancia** del
> controlador se use en los 4 escenarios simplemente cambiando
> los argumentos.
>
> El constructor también inicializa:
> - La matriz de pesos $Q$ como identidad
> - Una referencia $c_\text{ref}$ calculada del equilibrio nominal
> - La variable `c_prev` que almacena la última solución óptima
>   para usarla como **warm-start** en el siguiente paso

---

### 4.3. Modelo interno — El corazón de las predicciones

**Qué mostrar:** El método `_predice` (o la función `simulate_step`
que usa internamente).

**Qué decir:**

> El MPC necesita un modelo para predecir el futuro. Usamos el
> mismo modelo Lotka-Volterra pero en **versión determinista**:
> el controlador **desconoce** que la planta real tiene ruido
> Gaussiano y pulsos Poisson.
>
> Esto es intencional. En la práctica, nunca conocemos el modelo
> exacto de la planta. El MPC debe ser **robusto** frente a esas
> incertidumbres. Si el controlador supiera exactamente el ruido,
> sería un problema de control óptimo con información perfecta,
> que no es realista.
>
> La predicción se hace con Euler discretizado:
>
> $$x_{k+1} = x_k + \Delta t \cdot (a x_k - b x_k y_k)$$
> $$y_{k+1} = y_k + \Delta t \cdot (c x_k y_k - d y_k)$$
>
> con la protección $\max(0, \cdot)$ para evitar estados negativos.
> El paso de integración es $\Delta t = 0.1$.

---

### 4.4. Función de costo — ¿Qué estamos optimizando?

**Qué mostrar:** El método `_cost` en el código.

**Qué decir:**

> La función de costo que minimizamos en cada paso es:
>
> $$J = \sum_{t=0}^{N-1} \bigl[
>   (x_{k+t+1} - x_\text{ref})^2 +
>   (y_{k+t+1} - y_\text{ref})^2 +
>   r_c \cdot (c_{k+t} - c_\text{ref})^2 \bigr]$$
>
> Tiene tres componentes:
>
> 1. **Error de perfiles** $(x - x_\text{ref})^2$: penaliza que los
>    perfiles disponibles se desvíen de su valor deseado. Peso $q_x = 1$.
>
> 2. **Error de usuarios** $(y - y_\text{ref})^2$: penaliza que los
>    usuarios activos se desvíen de su referencia. Peso $q_y = 1$.
>    Ambos estados tienen el mismo peso porque queremos mantener
>    ambas poblaciones saludables.
>
> 3. **Esfuerzo de control** $r_c (c - c_\text{ref})^2$: penaliza
>    que $c$ se aleje de su valor de referencia. El peso $r_c = 0.1$
>    es pequeño, lo que le da al controlador libertad para ajustar
>    $c$ agresivamente cuando sea necesario. Si $r_c$ fuera grande,
>    el controlador sería muy tímido y no corregiría las
>    desviaciones.
>
> El vector de optimización es un arreglo de $M$ elementos: los
> valores de $c$ para los próximos $M$ pasos. Después del horizonte
> de control $M$, el control se mantiene constante en el último
> valor calculado hasta el final del horizonte de predicción $N$.
> Esto reduce la dimensionalidad del problema (10 variables en
> lugar de 15).

---

### 4.5. El solver — Cómo se resuelve la optimización

**Qué mostrar:** El método `solve` con la llamada a `minimize`.

**Qué decir:**

> Usamos `scipy.optimize.minimize` con el método **SLSQP**
> (*Sequential Least Squares Quadratic Programming*). Elegimos
> SLSQP porque:
>
> 1. Maneja **restricciones de caja** (los límites $c \in [0.002, 0.04]$)
>    de forma nativa
> 2. No requiere calcular derivadas analíticamente — usa
>    diferencias finitas para estimar el gradiente
> 3. Es eficiente para problemas no lineales de dimensión moderada
>    (en nuestro caso $M = 10$ variables)
>
> En cada paso de tiempo, `minimize` recibe:
> - Una función que calcula el costo dado un vector de control $c$
> - Las cotas $[0.002, 0.04]$ para cada elemento del vector
> - Una **estimación inicial** (*warm-start*) que acelera la
>    convergencia
>
> El warm-start funciona así: en el paso $k$, la solución óptima
> fue $c^*_{[k:k+M]}$. Para el paso $k+1$, tomamos esa solución,
> **descartamos el primer elemento** (que ya se aplicó) y
> **duplicamos el último** para llenar el hueco:
>
> $$c_\text{guess} = [c^*_{k+1}, c^*_{k+2}, \dots, c^*_{k+M-1}, c^*_{k+M-1}]$$
>
> Esto hace que el solver arranque cerca de la solución óptima del
> paso anterior, reduciendo drásticamente el tiempo de cómputo.
> En las simulaciones, SLSQP convergió al **100% de las veces**
> sin fallar ni violar restricciones.

---

### 4.6. El bucle de simulación — Cómo se conecta todo

**Qué mostrar:** El método `run_simulation` en el código.

**Qué decir:**

> El método `run_simulation` orquesta todo el proceso:
>
> 1. Inicializa los arreglos para almacenar las trayectorias de
>    $x$, $y$ y $c$
> 2. En cada paso $k$:
>    a. Mide el estado actual $(x_k, y_k)$
>    b. Llama a `solve` para obtener el $c$ óptimo
>    c. Aplica la primera acción de control $c_k$
>    d. Simula la **planta real** —que incluye ruido y pulsos—
>       para obtener $(x_{k+1}, y_{k+1})$
> 3. Al final, retorna las trayectorias completas
>
> Es importante notar la separación: el MPC usa un modelo interno
> determinista para decidir qué hacer, pero la evolución real del
> sistema incluye perturbaciones que el controlador no modela.
> Esta es exactamente la situación que enfrentaría un sistema de
> control desplegado en producción.

---

### 4.7. Tabla resumen de parámetros

**Qué mostrar:** La tabla de parámetros del MPC en el PDF.

**Qué decir recorriendo la tabla:**

> | Parámetro | Valor | ¿Por qué? |
> |---|---|---|
> | Modelo interno | Lotka-Volterra determinista | El MPC debe funcionar sin conocer el ruido |
> | $N$ (predicción) | 15 pasos (1.5 uds. tiempo) | Suficiente para ver la dinámica, no tanto que el ruido domine |
> | $M$ (control) | 10 pasos | Menor que $N$ para reducir la dimensionalidad |
> | $Q$ (estados) | $\text{diag}(1, 1)$ | Misma prioridad para $x$ e $y$ |
> | $r_c$ (control) | 0.1 | Bajo para permitir acción agresiva del MPC |
> | $c$ bounds | $[0.002, 0.04]$ | Límites físicos del sistema de matching |
> | Solver | SLSQP | Robusto para NLP con cajas |
> | Arranque | Warm-start (shift) | Reduce iteraciones del solver |
>
> Y lo más relevante: **esta misma configuración funciona para los
> 4 escenarios**. No tocamos ni $N$, ni $M$, ni $Q$, ni $r_c$ entre
> un escenario y otro. Esto demuestra que un MPC bien diseñado puede
> ser lo suficientemente flexible para operar bajo condiciones de
> mercado radicalmente distintas sin necesidad de re-sintonización.

---

## 5. Recorrido por los 4 escenarios (≈3–4 min)

**Qué mostrar:** Cada figura `comparativa_escenario*.png` una por
una (4 figuras en total).

---

### Escenario 1 — Crecimiento

**Parámetros:** $a = 0.5$, $d = 0.2$, estado inicial (10, 10),
referencia (50, 70).

**Qué decir:**

> En este escenario la app arranca desde un estado bajo, con pocos
> perfiles y pocos usuarios. La tasa de registro es alta y la
> deserción baja. Sin control —con $c$ fijo— $x$ colapsa y $y$ se
> dispara sin rumbo hacia su equilibrio natural de 91 usuarios.
>
> Con MPC, ambos estados convergen a la referencia de 50 perfiles
> y 70 usuarios. La señal de control $c$ se ajusta continuamente
> para mantener el crecimiento.

---

### Escenario 2a — Mantener

**Parámetros:** $a = 0.1$, $d = 0.6$, estado inicial (60, 60),
referencia (35, 20).

**Qué decir:**

> Aquí enfrentamos un entorno adverso: la tasa de registro es baja
> y la deserción es alta. La app tiene una base grande de 60 en
> cada población, pero está en riesgo. Sin control, los usuarios
> $y$ tienden a cero —la aplicación muere— porque la deserción
> supera la capacidad de generar matches.
>
> Con MPC, el control mantiene $y$ vivo cerca de su equilibrio
> natural de 18 usuarios, y $x$ se estabiliza alrededor del valor
> de referencia. La app sobrevive.

---

### Escenario 2b — Crecer en perfiles

**Parámetros:** $a = 0.1$, $d = 0.6$, estado inicial (60, 60),
referencia (60, 25).

**Qué decir:**

> Mismos parámetros que el anterior, pero la referencia pide llevar
> los perfiles a 60. Para lograrlo, el MPC debe reducir $c$, lo que
> eleva el equilibrio de $x^* = d/c$. Como resultado, los perfiles
> crecen significativamente, mientras $y$ se sostiene en su cota
> natural de 18 usuarios.

---

### Escenario 3 — Dinámico (Markov)

**Parámetros:** $a(t)$, $d(t)$ markovianos, estado inicial (30, 30),
referencia (40, 40).

**Qué decir:**

> En el escenario más exigente, $a$ y $d$ varían abruptamente entre
> tres modos: crecimiento, estable y crisis. El MPC recibe el valor
> actual de cada parámetro en cada paso y debe adaptar $c$ en
> tiempo real.
>
> Sin control, la aplicación colapsa en las fases de crisis.
> Con MPC, la app sobrevive la mayor parte del tiempo, aunque con
> 10000 pasos de simulación se observan episodios donde $x$ o $y$
> tocan cero incluso con control. Esto demuestra la dificultad
> extrema de operar bajo condiciones de mercado altamente volátiles.

---

## 6. Tabla de extinción y RMS (≈1 min)

**Qué mostrar:** Las tablas de resultados del PDF (Tabla de
extinción y Tabla RMS).

**Qué decir:**

> La tabla de extinción muestra que el MPC evita que $x$ o $y$
> lleguen a cero en los escenarios 1, 2a y 2b. Sin control, en
> cambio, $y$ colapsa siempre que la deserción es alta.
>
> En el escenario dinámico, tanto con como sin control se
> registran episodios de extinción, lo que refleja la volatilidad
> extrema del mercado simulado.
>
> En cuanto al error RMS —medido sobre los últimos 700 pasos— el
> MPC reduce significativamente el error en $y$ (usuarios) en todos
> los escenarios. La mejora es más notable en los escenarios 2a y
> 2b, donde el error en $y$ se reduce entre 3 y 7 veces.
>
> En $x$ (perfiles) la mejora es menor o incluso negativa en
> algunos casos, porque el controlador prioriza mantener los
> usuarios vivos antes que reducir el error de perfiles.

---

## 7. Conclusiones (≈30 s)

**Qué mostrar:** La sección de conclusiones en el PDF.

**Qué decir:**

> En resumen:
>
> Un solo MPC con una única variable de control —la eficiencia
> del matching— logra operar la aplicación de citas en cuatro
> escenarios con condiciones de mercado muy distintas.
>
> El límite fundamental $y^* = a/b$ impone una restricción
> física: no podemos aumentar usuarios más allá de lo que
> permiten la tasa de registro y la tasa de matches.
>
> Con ruido, pulsos Poisson y condiciones dinámicas de mercado,
> el MPC demuestra ser robusto, aunque en escenarios extremadamente
> volátiles la extinción sigue siendo posible.
>
> Esto concluye nuestra presentación. Gracias.
