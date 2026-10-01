# Aprender significa cambiar números: la derivada desde cero {#ch:derivada}

::: idea Empezamos con una perilla
Una máquina predice el precio de una entrada multiplicando el número de personas por una tarifa. Si la predicción se equivoca, queremos saber en qué dirección mover la tarifa. La derivada será una forma precisa de expresar esa sensibilidad local.
:::

## Un error que podemos calcular a mano

Supongamos que tres personas deberían pagar doce unidades. Nuestro modelo tiene una tarifa ajustable w y predice $p=3w$. Si w vale dos, predice seis. Para convertir la discrepancia en un número no negativo, elevamos al cuadrado el error: $L=(p-12)^2$. L se llama pérdida.

Con w=2, L=36. Con w=2.1, p=6.3 y L=32.49. Aumentar un poco w ha reducido la pérdida. Con w=1.9, p=5.7 y L=39.69: disminuir w la empeora. Todavía no hemos necesitado una fórmula de derivación para identificar la dirección correcta.

La pregunta cuantitativa es cuánto cambia L por un cambio pequeño de w. El cociente entre ambos cambios aproxima una pendiente. Cerca de w=2 esa pendiente es negativa: al avanzar hacia la derecha, la pérdida baja.

::: traduccion Qué dice el signo
Pendiente positiva: aumentar la variable aumenta localmente la función. Pendiente negativa: aumentarla disminuye localmente la función. Pendiente cero: la primera variación local no indica una dirección; no garantiza por sí sola un mínimo.
:::

## Derivar no es dividir dos números escritos

La notación $\dd L/\dd w$ representa una sensibilidad. En este ejemplo puede obtenerse con álgebra: $L=9w^2-72w+144$, así que la pendiente es $18w-72$. En w=2 vale -36.

El número -36 no es la pérdida ni la tarifa nueva. Dice que un incremento muy pequeño de w produce aproximadamente -36 veces ese incremento en L. Para un incremento de 0.1, la aproximación predice una reducción de 3.6; la reducción real es 3.51 porque la función es curva.

Si escogemos un paso de aprendizaje de 0.01 y nos movemos contra la pendiente, obtenemos w nueva igual a $2-0.01(-36)=2.36$. No saltamos directamente a la solución w=4. Damos un paso cuya magnitud depende de una decisión del optimizador.

## La regla de la cadena como transmisión de cambios

Nuestro cálculo tiene varias estaciones: w produce p; p produce un error e; e produce L. Cada estación transforma una pequeña variación que recibe. La variación final resulta de multiplicar esas sensibilidades locales a lo largo del camino.

```diagram
Tarifa w | Predicción p=3w | Pérdida L=(p-12)²
```

En w=2, p=6 y e=-6. La sensibilidad de p respecto a w es 3. La de e respecto a p es 1. La de L respecto a e es $2e=-12$. Multiplicamos: $(-12)\cdot1\cdot3=-36$.

Ahora podemos escribir la versión compacta sin que sea una colección de símbolos nuevos:

```math
\frac{\dd L}{\dd w}=\frac{\dd L}{\dd e}\frac{\dd e}{\dd p}\frac{\dd p}{\dd w}.
```

Cada factor responde una pregunta local. El compilador conoce qué operación creó cada nodo y puede construir automáticamente esas preguntas en orden inverso.

## Dos caminos obligan a sumar

Considera $y=x\cdot x$. x aparece en los dos operandos de la multiplicación. Si x cambia, cambia tanto el primer factor como el segundo. Cada camino contribuye x a la pendiente, de modo que el total es $2x$.

Un motor de autodiferenciación que guardara solo la última contribución devolvería x en lugar de 2x. Este fallo es especialmente fácil de cometer en un grafo con subexpresiones compartidas: el grafo no es siempre un árbol.

Con $y=x\cdot z+x$, hay dos caminos desde x hasta y. El producto aporta z y la suma directa aporta 1. La sensibilidad total es z+1. Para z=3 vale 4, aunque ninguna operación local individual tuviera por derivada 4.

::: cuidado Asignar y acumular no son lo mismo
En la propagación inversa, cuando un nodo recibe otra contribución usamos suma. Reemplazar la contribución anterior destruye información de los demás caminos.
:::

## Qué necesitamos guardar del avance

Para derivar una multiplicación, necesitamos los valores de sus operandos. Para derivar una exponencial, podemos reutilizar su resultado. Para una suma, no necesitamos los valores de entrada: su sensibilidad local es constante.

Esa diferencia conecta autodiff con memoria. Guardar todos los valores facilita el retroceso, pero consume espacio. Volver a calcular algunos reduce memoria a cambio de cómputo. La decisión no debería hacerse antes de saber qué datos necesita cada regla de gradiente.

Nuestro compilador construye un grafo de gradientes y después lo planifica como cualquier otro. No ejecuta una fórmula en Python por cada elemento del tensor durante el entrenamiento. Python construye la receta; el código C generado hace los cálculos numéricos.

## Una derivada numérica para comprobar, no para entrenar todo

La diferencia central aproxima la pendiente usando dos evaluaciones:

```math
\frac{\partial L}{\partial w_i}\approx
\frac{L(w+h e_i)-L(w-h e_i)}{2h}.
```

Aquí $e_i$ significa cambiar únicamente la componente i. Para entenderlo, imagina un panel de cien perillas: movemos solo una hacia arriba y hacia abajo, dejando las otras noventa y nueve en el mismo sitio.

La aproximación tiene dos problemas. Si h es muy grande, deja de medir la pendiente local. Si es demasiado pequeño, el redondeo puede borrar la diferencia entre las dos evaluaciones. No existe una elección universal que arregle todos los casos.

Para un tensor pequeño y funciones suaves, la diferencia central es una prueba excelente. Para millones de parámetros sería muy costosa: requiere evaluaciones adicionales por cada componente. El retroceso reutiliza la estructura del cálculo para obtener muchas sensibilidades juntas.

## Primera práctica con Lumbre

```python
import numpy as np
from lumbre import param, Program, gradients

w = param("tarifa", ())
pred = 3.0 * w
loss = (pred - 12.0) * (pred - 12.0)
grad = gradients(loss, [w])[0]
with Program([pred, loss, grad]) as program:
    values = program.run({"tarifa": np.array(2, dtype="float32")})
    print([float(x) for x in values])
```

La salida esperada es 6, 36 y -36. Primero comprueba esos tres números, no solo el último. Si la predicción ya es incorrecta, investigar la regla del gradiente sería empezar por el lugar equivocado.

Después cambia w a cuatro. La predicción es doce, la pérdida cero y el gradiente cero. Cambia w a cinco: predicción quince, pérdida nueve y gradiente dieciocho. El signo ahora señala que conviene reducir la tarifa.

## Dónde acaba esta intuición

En varias dimensiones, cada componente del gradiente mide el cambio respecto a una variable manteniendo las demás fijas. El conjunto de esas sensibilidades orienta un paso local. No garantiza encontrar el mejor mínimo global de una red neuronal.

Además, algunas operaciones no son diferenciables en ciertos puntos. ReLU cambia de pendiente en cero; máximo puede tener empates. Un framework elige una convención o una subderivada para esos casos. El compilador debe documentarla y probarla, no fingir que existe una única derivada clásica donde no la hay.

# Autodiferenciación inversa sobre un grafo de UOps {#ch:autodiff}

## El retroceso empieza con una pregunta

Queremos saber cómo cambia una pérdida escalar al cambiar cada parámetro. La sensibilidad de la pérdida respecto a sí misma es uno: si L aumenta una unidad, L aumenta una unidad. Ese uno es la semilla del retroceso.

Recorremos los nodos en orden topológico inverso. Cuando visitamos una operación, ya hemos reunido las contribuciones procedentes de sus consumidores. Aplicamos su regla local y añadimos las contribuciones a sus operandos.

```text
adjunto[perdida] = 1
para nodo en orden_topologico_inverso:
    g = adjunto[nodo]
    para operando, contribucion en regla_local(nodo, g):
        adjunto[operando] += contribucion
```

Este es pseudocódigo explicativo. La implementación completa de `gradients` está en el apéndice y devuelve nuevos UOps, no arrays NumPy ya calculados.

## Qué significa un adjunto

Llamaremos adjunto de x a la sensibilidad de la pérdida respecto a x. El nombre evita repetir una frase larga. Si x es un tensor, su adjunto tiene la forma necesaria para representar la sensibilidad respecto a cada elemento de x.

Una operación vectorial podría describirse con una gran matriz de derivadas llamada jacobiano. No necesitamos construir esa matriz completa. La regla inversa calcula directamente cómo una sensibilidad de salida se transforma en sensibilidades de entrada. Esa operación es un producto vector-jacobiano, o VJP.

::: ejemplo Una suma de mil elementos
Para $s=x_0+\cdots+x_{999}$, cada entrada tiene sensibilidad uno respecto a s. Si el adjunto de s es 0.2, cada entrada recibe 0.2. No construimos una matriz de mil columnas para descubrirlo: expandimos un escalar.
:::

## Reglas locales que debemos poder justificar

Para z=x+y, ambas entradas reciben g. Para z=x-y, x recibe g e y recibe -g. Para z=xy, x recibe gy e y recibe gx. Para z=x/y, x recibe g/y e y recibe $-gx/y^2$.

Para z=exp(x), el adjunto de x es gz. Para z=log(x), es g/x, dentro del dominio donde la función está definida. Para z=sqrt(x), es $g/(2z)$ en los puntos donde la derivada ordinaria existe. Para z=tanh(x), es $g(1-z^2)$.

No memorices la tabla sin verificar una operación. Con z=xy y y fijo, aumentar x en una cantidad h aumenta z en yh. Esa es la razón de la contribución gy. La misma explicación intercambiando los factores da gx.

| Operación | Contribución hacia x | Dato que hace falta |
|---|---|---|
| x+y | g | No necesita valores de x ni y. |
| x·y | g·y | El otro operando. |
| exp(x) | g·exp(x) | Entrada o resultado. |
| log(x) | g/x | La entrada. |
| tanh(x) | g·(1-z²) | El resultado z. |

El dominio numérico sigue siendo parte del contrato. Una regla algebraica no convierte `log(-1)` en una operación válida ni evita que aparezcan infinitos al dividir por cero.

## Broadcasting hacia delante, suma hacia atrás

Supón que sumamos un sesgo de tres elementos a dos filas:

```math
Y_{ij}=X_{ij}+b_j,\qquad i=0,1;\ j=0,1,2.
```

El sesgo $b_0$ se utilizó en dos celdas: $Y_{00}$ y $Y_{10}$. Por tanto, su gradiente recibe las dos contribuciones. Si los adjuntos de salida son las filas [1,2,3] y [4,5,6], el gradiente del sesgo es [5,7,9].

Devolver el tensor de dos filas como gradiente de un sesgo de una fila sería un error de forma y de significado. La operación inversa del broadcasting suma sobre los ejes que se replicaron y conserva o restaura las dimensiones de tamaño uno.

Lumbre implementa esa reconciliación mediante una reducción hacia la forma original. Los ejes añadidos por delante también cuentan. La regla debe contemplar la diferencia entre `(3,)`, `(1,3)` y `(2,3)`, aunque algunas operaciones produzcan los mismos valores visibles.

## Reducir hacia delante, expandir hacia atrás

Para una suma por filas, cada elemento de una fila contribuyó a su resultado. El adjunto del resultado se expande a todas esas posiciones. Para una media, además se divide por el número de elementos reducidos.

Si x=[2,4,6] y y=media(x), el resultado es cuatro. Aumentar solo x0 en 0.3 aumenta la media en 0.1. La sensibilidad respecto a cada entrada es un tercio. Una implementación que olvidara esa división entrenaría con gradientes multiplicados por el tamaño del eje.

Para máximo, únicamente las posiciones que alcanzan el valor máximo participan según la convención elegida. Nuestra implementación distribuye el adjunto entre las posiciones empatadas. Es una elección explícita; otros sistemas pueden seleccionar una posición. Las pruebas deben compararse con una referencia que use la misma convención.

## El producto de matrices explicado con una celda

Si $Y=AB$, la celda $Y_{ij}$ es la suma de $A_{ik}B_{kj}$. Una entrada $A_{ik}$ influye en todas las columnas j de la fila i. Cada contribución se multiplica por $B_{kj}$. Al sumar las sensibilidades de salida aparece:

```math
\overline A=\overline Y B^T,\qquad
\overline B=A^T\overline Y.
```

La barra significa adjunto. Comprueba las formas antes de calcular: si A es M×K, B es K×N y el adjunto de Y es M×N, el primer producto devuelve M×K y el segundo K×N.

En un matmul por lotes con broadcasting, esas fórmulas producen contribuciones por cada lote expandido. Después hay que sumarlas hacia las formas originales. Omitir ese paso puede dar gradientes correctos cuando los lotes coinciden y fallar cuando un operando se comparte.

## Movimientos: deshacer direcciones, no derivar etiquetas

Una permutación se deshace con la permutación inversa. Un reshape recupera la forma de entrada. Un slice coloca su gradiente en las posiciones seleccionadas y deja cero en las demás. Un padding elimina del gradiente las posiciones añadidas.

Un gather de embeddings exige más cuidado. Si dos tokens seleccionan la misma fila de una tabla, sus contribuciones se suman en esa fila. No se sobrescriben. La operación inversa es una acumulación dispersa, habitualmente descrita como scatter-add.

::: practica Un embedding con un token repetido
Una tabla tiene cuatro filas y dos columnas. Los índices son [2,0,2]. El adjunto de las tres selecciones es [[1,2],[3,4],[5,6]]. ¿Qué recibe cada fila de la tabla?
:::

La fila cero recibe [3,4]. La fila uno recibe [0,0]. La fila dos recibe [6,8], suma de la primera y tercera selección. La fila tres recibe [0,0]. Este caso pequeño detecta una implementación que simplemente escribe el último gradiente recibido.

## Comparaciones, selección y stop-gradient

Una comparación produce una decisión discreta. En nuestro motor no propagamos una derivada hacia el cambio de esa decisión. En `where(cond,a,b)`, el adjunto se dirige a a donde cond es verdadera y a b donde es falsa.

Eso no implica que todas las funciones con decisiones sean suaves. Describe la regla que usamos para una ejecución y una convención concreta. En una frontera donde la condición cambia, la diferencia numérica puede no coincidir con esa regla.

`detach` conserva el valor hacia delante y corta la propagación hacia atrás. Es útil para constantes auxiliares o para escribir algoritmos con una separación explícita de dependencias. No debe añadirse solo para hacer desaparecer un gradiente que revela un error.

## Autodiff también se compila

La salida de `gradients` es otro grafo. Podemos simplificarlo, identificar subexpresiones compartidas, fusionar elementwise, planificar reducciones y generar C. Esa decisión hace que el desarrollo de optimizaciones beneficie tanto al avance como al retroceso.

Sin embargo, un patrón rápido del avance no define automáticamente un retroceso correcto. Si añadimos una operación nueva de atención online, necesitamos su VJP o una descomposición diferenciable equivalente. El compilador debe rechazar una operación sin regla cuando se piden gradientes, en lugar de devolver ceros por defecto.

## Pruebas por capas

Primero se prueba cada regla local con tensores diminutos y valores lejos de singularidades. Después se prueban composiciones, broadcasting y nodos compartidos. Finalmente se comprueba una actualización de parámetros y una pérdida que descienda en un problema controlado.

Una pérdida decreciente no demuestra por sí sola que todos los gradientes sean correctos: un modelo podría aprender lentamente con un gradiente parcialmente incorrecto. A la inversa, una pérdida que no desciende no demuestra necesariamente un fallo de autodiff; puede haber una tasa de aprendizaje inadecuada o datos sin señal.

El diagnóstico combina pruebas numéricas locales y comportamiento global. Esta separación evita culpar al optimizador antes de comprobar el signo de una derivada.

# Optimizar parámetros sin corromper el estado {#ch:optimizadores}

## Descenso de gradiente con un ejemplo completo

Para $L=(w-3)^2$, el gradiente es $2(w-3)$. Con w=0 y tasa 0.1, el primer paso da 0.6. La pérdida pasa de 9 a 5.76. El segundo gradiente es -4.8 y el nuevo valor 1.08. La pérdida vuelve a disminuir.

El siguiente código compila tanto la pérdida como la propuesta de actualización. La llamada `assign` copia los nuevos valores a los parámetros después de que el programa termine.

```python
import numpy as np
from lumbre import param, gradients, Program

w = param("w", ())
loss = (w-3.0)*(w-3.0)
g = gradients(loss, [w])[0]
new_w = w-0.1*g
with Program([loss, new_w]) as p:
    p.set("w", np.array(0.0, dtype="float32"))
    for step in range(5):
        before, proposal = p.run()
        print(step, float(before), float(proposal))
        p.assign({"w": new_w})
```

La pérdida impresa corresponde a los parámetros antes de aplicar la propuesta de esa iteración. Confundir ese momento puede hacer que un gráfico parezca desplazado un paso o que se compare con el checkpoint equivocado.

## Por qué la actualización no debe pisar sus entradas

Supón que dos parámetros se actualizan usando valores antiguos de ambos. Si escribimos el primero antes de calcular el segundo, el resultado puede dejar de representar un paso simultáneo. El orden de copia habrá cambiado el algoritmo.

Nuestro diseño calcula las salidas de actualización y después las asigna. Además, rechaza intercambios directos de parámetros como `a<-b, b<-a` a través de la API de asignación cuando las salidas son los propios parámetros: esa operación requiere un manejo explícito del aliasing. Un rechazo claro es preferible a una corrupción silenciosa.

En una implementación de producción se pueden demostrar condiciones de actualización in-place o usar buffers alternos. El optimizador y el planificador de memoria deben compartir información sobre qué valores antiguos siguen vivos.

## Momentum: recordar una dirección reciente

Una pendiente puede cambiar mucho entre minibatches. Momentum mantiene una media de gradientes anteriores para suavizar la dirección. Imagina empujar un carrito: un pequeño cambio de pendiente no invierte instantáneamente su movimiento.

La analogía tiene un límite: el algoritmo no es una simulación física exacta, sino una regla numérica. Hay que especificar cómo se pondera la memoria anterior y cómo se usa para actualizar el parámetro. Dos convenciones de momentum pueden parecer distintas y estar relacionadas por una escala de la tasa.

## AdamW en cuatro piezas

Adam mantiene una media del gradiente y una media de su cuadrado. La primera orienta la actualización. La segunda adapta su escala por componente. Como ambas medias empiezan en cero, se corrige el sesgo de las primeras iteraciones. AdamW aplica además un decaimiento desacoplado de los pesos.

La separación entre Adam y decaimiento desacoplado se documenta en sus trabajos originales. [@adam;@adamw] La implementación didáctica usa estas ecuaciones:

```math
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2,
```

```math
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},\qquad
w_{t+1}=w_t(1-\eta\lambda)-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
```

Lee las ecuaciones en ese orden. t es el número de actualización, no el número de tokens. $\eta$ es la tasa de aprendizaje, $\lambda$ el decaimiento y $\epsilon$ una protección numérica. Los cuadrados y divisiones se aplican por elemento.

En el primer paso, con m y v iniciales cero, las correcciones de sesgo recuperan aproximadamente g y g². Eso permite una comprobación manual sencilla para un parámetro escalar.

La política de qué parámetros reciben decaimiento es una decisión adicional. En redes reales se suelen separar grupos según su función; el ejemplo del paquete mantiene una política simple y explícita. No presentamos esa simplificación como la receta óptima para entrenar todos los LLM.

## Recortar el gradiente sin cambiar cada componente a ojo

El clipping por norma global calcula una longitud conjunta de todos los gradientes. Si supera un umbral, multiplica todos por el mismo factor. Así limita la magnitud conservando la dirección global del vector de gradiente.

Para gradientes [3,4], la norma es 5. Con umbral 1, se aplica factor 0.2 y quedan[0.6,0.8]. Recortar cada componente por separado a 1 produciría[1,1], una dirección diferente. Son algoritmos distintos.

El cálculo de la norma también forma parte del grafo compilado. Tiene un coste y requiere una reducción; no conviene omitirlo del tiempo de entrenamiento si se usa en cada actualización.

## Gradientes acumulados y lotes efectivos

Cuando no cabe un lote grande, podemos sumar contribuciones de varios microbatches antes de actualizar. Para equivaler a una media sobre todos los ejemplos, hay que ponderar correctamente sus tamaños y pérdidas.

Si dos microbatches tienen cuatro y dos tokens válidos, promediar sus dos pérdidas con peso un medio no equivale a promediar los seis tokens. El primero debería pesar cuatro sextos y el segundo dos sextos. Las máscaras y longitudes variables hacen que este detalle sea frecuente en modelos de lenguaje.

Además, el contador t de Adam debe avanzar por actualización del optimizador, no por cada microbatch si todavía no se actualiza. El clipping global se aplica normalmente al gradiente agregado cuando esa es la receta que se pretende reproducir.

## Checkpoint: mucho más que pesos

Para continuar el mismo proceso necesitamos pesos, estados del optimizador, contador de actualización, configuración del modelo y estado del generador aleatorio. Si cambió el vocabulario, también debe detectarse. Un archivo que contiene solo pesos sirve para iniciar otro proceso, no para prometer una reanudación equivalente.

El ejemplo guarda esos componentes en un archivo NPZ y valida configuración y vocabulario al reanudar. La equivalencia bit a bit entre distintos dispositivos o toolchains no se promete: el orden numérico puede cambiar. Dentro del mismo entorno, una prueba de interrupción y reanudación permite comprobar la lógica del checkpoint.

::: comprueba Qué debe poder explicar el estudiante
Una actualización usa gradientes calculados con los parámetros adecuados, conserva los estados que necesita el paso siguiente y tiene un momento de aplicación definido. El entrenamiento es una secuencia de estados, no solo una función que devuelve una pérdida pequeña.
:::
