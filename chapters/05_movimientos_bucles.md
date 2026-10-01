# Movement ops: cambiar el mapa sin mover las cajas {#ch:movement}

## Una estantería y dos maneras de leerla

Coloca seis cajas en una estantería lineal con los valores 1,2,3,4,5,6. Puedes dibujar alrededor una tabla de dos filas y tres columnas, o una de tres filas y dos columnas. Las cajas no se han movido; ha cambiado la forma de interpretar sus posiciones.

Las **movement ops** reorganizan ejes, seleccionan regiones o reutilizan valores. A menudo pueden representarse como cambios de índices sin copiar inmediatamente el contenido. Tinygrad destaca esta separación entre operaciones aritméticas, reducciones y movimientos. La ausencia de copia es una posibilidad de representación, no una garantía universal de coste cero para cualquier cadena de consumidores. [@tinygrad-site]

## Strides: cuánto avanzar al cambiar una coordenada

En una matriz contigua de dos filas y tres columnas, aumentar una columna avanza un elemento; aumentar una fila avanza tres. Los strides son `(3,1)`. La dirección lógica del elemento `(i,j)` es `3*i+j`, más un desplazamiento inicial si la vista comienza dentro de otro buffer.

Para una forma `(2,3,4)` contigua por filas, los strides son `(12,4,1)`. El primer eje salta un bloque de doce elementos; el segundo, una fila de cuatro; el tercero, una posición. La última dimensión varía más deprisa.

::: traduccion Dirección de una vista afín
Con offset o, coordenadas $i_0,\ldots,i_{r-1}$ y strides $s_0,\ldots,s_{r-1}$, el índice de almacenamiento es:
```math
\operatorname{addr}(i_0,\ldots,i_{r-1})=o+\sum_{a=0}^{r-1}i_a s_a.
```
La suma recorre ejes, no los valores del tensor. Multiplicar después por bytes por elemento convierte ese índice en un desplazamiento de bytes.
:::

## Transposición como intercambio de strides

La matriz original `(2,3)` tiene strides `(3,1)`. Su transpuesta tiene forma `(3,2)` y strides `(1,3)`. El elemento `(2,1)` de la transpuesta apunta a `2*1+1*3=5`, donde está el valor 6. No hemos cambiado el buffer.

El acceso secuencial por filas de la vista transpuesta salta ahora distancias de tres elementos. Para algunos consumidores esto es adecuado; para otros conviene materializar una copia contigua. La decisión depende de cuántas veces se utilizará y de la eficiencia de los accesos que evita.

```tikz
\begin{center}
\begin{tikzpicture}
\foreach \x/\v in {0/1,1/2,2/3,3/4,4/5,5/6}{\node[draw,minimum width=1cm,minimum height=.65cm] at (\x,0) {\v};}
\node[align=center,font=\small] at (2.5,1) {Un único almacenamiento lineal};
\node[box,text width=4.6cm] (a) at (0,-1.7) {Vista $(2,3)$\\strides $(3,1)$};
\node[box,text width=4.6cm] (b) at (5,-1.7) {Vista $(3,2)$\\strides $(1,3)$};
\draw[flow] (a.north)--(1,-.4);\draw[flow] (b.north)--(4,-.4);
\end{tikzpicture}
\end{center}
```

## Reshape no siempre es una vista simple

Cambiar una forma contigua `(2,3)` a `(3,2)` conserva el orden lineal. Pero intentar reinterpretar una transpuesta no contigua puede requerir más que asignar nuevos strides. La pregunta no es únicamente si el número de elementos coincide: también importa el orden lógico de esos elementos.

El laboratorio `layout.py` permite un reshape contiguo y rechaza otro que no pueda justificar. El IR tensorial de Lumbre, en cambio, representa reshape como una transformación de su orden lógico y puede insertar materialización en las fronteras elegidas por el scheduler. No debemos confundir estas dos superficies de la entrega: una es un modelo de vistas afines; la otra, una representación de operaciones tensoriales.

::: ejemplo Un reshape que revela el orden
Partimos de `[[1,2,3],[4,5,6]]`. Su transpuesta se lee lógicamente como `[[1,4],[2,5],[3,6]]`. Aplanarla en ese orden produce `[1,4,2,5,3,6]`, no `[1,2,3,4,5,6]`. Cambiar solo la etiqueta de forma sobre el buffer original daría la segunda secuencia y sería incorrecto para ese contrato.
:::

## Expand y stride cero

Un valor que se reutiliza a lo largo de un eje puede representarse con stride cero en ese eje. Una fila de tres números expandida a cuatro filas tiene forma `(4,3)` y strides `(0,1)`: cambiar de fila no cambia la dirección.

Leer esa vista es sencillo. Escribir en ella es peligroso si varias posiciones lógicas apuntan al mismo lugar. Dos hilos que escriben valores distintos en filas diferentes podrían estar escribiendo realmente la misma dirección. El compilador necesita restricciones de escritura o una semántica explícita de combinación.

Por eso una vista de broadcasting no debe tratarse como un array independiente solo porque tiene una forma grande. El número de posiciones lógicas no es necesariamente el número de ubicaciones físicas distintas.

## Slicing: cambiar el inicio y la extensión

Seleccionar columnas 1 y 2 de una matriz de cuatro columnas conserva el stride de fila cuatro, pero cambia el offset inicial en uno y la forma de la selección. Una región puede ser rectangular en coordenadas lógicas y no contigua en memoria.

Para un slice con paso uno, el mapeo de una coordenada de salida j a la entrada es `j+inicio`. Para pasos diferentes, sería `inicio+j*paso`, con más condiciones de límites. El IR ejecutable de esta edición admite slices de paso uno con extremos no negativos; el capítulo enseña cómo ampliar el contrato sin afirmar que esa ampliación ya esté implementada.

## Invertir un eje y strides negativos

Una fila `[1,2,3,4]` puede leerse al revés empezando en la última posición y usando stride -1. El offset debe cambiar a tres. Si solo cambias el signo del stride y mantienes el inicio en cero, intentarás leer antes del buffer.

La regla general para invertir un eje de longitud n añade `(n-1)*stride` al offset y cambia el signo del stride. Para ejes vacíos hay que definir el caso sin crear un acceso. El laboratorio de vistas cubre inversión como ejercicio de direcciones; el frontend tensorial no ofrece toda esa API de strides al runtime C.

## Padding: una lectura condicional

Añadir un borde de ceros crea posiciones de salida que no corresponden a ninguna entrada. Podemos evitar materializar el borde si el consumidor utiliza una lectura condicional: dentro de la región válida lee la entrada; fuera, devuelve cero.

El orden de la condición importa. No es seguro leer primero una dirección inválida y después multiplicar el valor por una máscara cero. La lectura ya ha ocurrido. En C se puede utilizar una expresión condicional que solo evalúe la rama seleccionada bajo la semántica del lenguaje.

```c
float valor = (j >= inicio && j < final)
            ? entrada[j - inicio]
            : 0.0f;
```

Un backend GPU debe confirmar que su instrucción de carga enmascarada respeta el acceso, no solo el valor final. Las máscaras de carga, las de cálculo y las de escritura son conceptos relacionados pero diferentes.

## Componer movimientos

Una permutación, un slice y un expand pueden componerse en una única función de índice. Esto evita crear buffers intermedios para cada movimiento. Pero una expresión de índice demasiado complicada puede añadir divisiones, módulos y ramas a cada acceso.

La política óptima depende del uso. Una transposición compartida por veinte multiplicaciones puede merecer una copia contigua reutilizable. Una selección usada una sola vez puede integrarse en la carga. El compilador debe comparar coste de materializar, coste de recomputar índices y beneficio de acceso.

## El backward de un mapa

Una vista no cambia necesariamente los valores, pero sí cómo se acumulan sus gradientes. Una permutación se deshace con la permutación inversa. Un reshape recupera la forma anterior. Un expand suma las contribuciones de todas las posiciones que reutilizaron un mismo dato. Un slice coloca sus gradientes en las posiciones seleccionadas y deja cero en las demás.

Esta relación anticipa una idea fundamental: **reutilizar en forward implica acumular en backward**. Si tres salidas leen el mismo escalar, las tres pueden influir en su gradiente. No basta con escoger la contribución de la primera.

::: practica Calcular una dirección completa
Una vista tiene forma `(3,2)`, strides `(1,4)` y offset 2. Calcula la dirección lógica de `(2,1)`. Después invierte el primer eje: ¿qué offset y strides necesitas para conservar las mismas posiciones en orden inverso?
:::

**Solución.** La dirección es `2+2*1+1*4=8`. Al invertir el eje de longitud tres, el offset pasa a `2+(3-1)*1=4`; los strides pasan a `(-1,4)`. El elemento nuevo `(0,1)` apunta a ocho y corresponde al antiguo `(2,1)`.

::: comprueba Antes de continuar
Debes distinguir forma, strides, offset y almacenamiento. Explica por qué expand puede ser barato al leer y problemático al escribir. Después demuestra con seis números por qué no todo reshape no contiguo puede resolverse cambiando etiquetas.
:::

# Loops y rangeify: convertir un tensor en recorridos {#ch:rangeify}

## La receta tensorial necesita índices

Una operación de suma dice qué relación tienen entradas y salida, pero no enumera posiciones. Para ejecutarla necesitamos rangos: i recorre filas, j columnas, k términos de una reducción. Introducir explícitamente esos rangos es el puente entre álgebra tensorial y código de bucles.

En el vocabulario de tinygrad, **rangeify** forma parte de ese descenso hacia rangos y accesos. El pipeline real cambia con el proyecto; el recorrido de esta edición se vincula al commit fijado. Lumbre implementa un descenso más sencillo a expresiones de índices y bucles C. La similitud es conceptual, no identidad de implementación. [@tinygrad-pin;@tinygrad-codegen]

## Un bucle lineal puede recorrer varios ejes

Para una salida `(2,3)`, q recorre de cero a cinco. La fila es `q//3` y la columna `q%3`. La misma técnica se amplía a más ejes: dividimos por el producto de dimensiones posteriores y tomamos resto por el tamaño del eje.

Para `(2,3,4)` y q=17, el primer eje vale `17//12=1`; el segundo `(17//4)%3=1`; el tercero `17%4=1`. La coordenada es `(1,1,1)`. Reconstruir el índice da `1*12+1*4+1=17`.

::: ejemplo Un recorrido verificable
Enumera las coordenadas de q=0,1,3,4,11,12,23 para `(2,3,4)`. Son `(0,0,0)`, `(0,0,1)`, `(0,0,3)`, `(0,1,0)`, `(0,2,3)`, `(1,0,0)` y `(1,2,3)`. Las fronteras muestran cuándo se reinicia un eje y avanza el anterior.
:::

## Bucles anidados y equivalencia con el índice lineal

La versión anidada recorre explícitamente cada eje. La versión lineal calcula coordenadas a partir de q. Ambas pueden visitar las mismas posiciones en el mismo orden para un layout contiguo. La elección afecta al código que ve el compilador C y a las oportunidades de optimización.

```c
for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
        y[i*3+j] = x[i*3+j] + 2.0f;
    }
}
```

Si las dimensiones son constantes, un compilador C puede simplificar divisiones y módulos. Si son variables, algunos índices resultan más caros. Esto no obliga a evitar siempre la linealización: permite mapear fácilmente un hilo GPU a una posición. El diseño compara ventajas y costes.

## Ejes de salida y ejes de reducción

En GEMM, i y j identifican una salida; k recorre valores que contribuyen a ella. Los dos tipos de eje no son intercambiables. Repartir salidas entre trabajadores no requiere combinarlas. Repartir k obliga a sumar resultados parciales de una misma salida.

```c
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[i*K+k] * B[k*N+j];
        }
        C[i*N+j] = acc;
    }
}
```

El acumulador se reinicia para cada pareja `(i,j)`. Situarlo fuera del bucle de j sumaría resultados de columnas diferentes. Esta es una de las primeras pruebas de scope que debe cubrir un compilador de reducciones.

## Invariantes de bucle

Un invariante es una afirmación que se mantiene en un punto del bucle. Después de procesar los primeros k términos de una salida GEMM, el acumulador representa la suma de esos términos bajo el orden y redondeo del programa. Al empezar no se ha procesado ninguno y el acumulador vale cero. Cada vuelta añade exactamente el siguiente producto.

Al terminar k=K, se han incluido todos los términos requeridos. Ese argumento explica la corrección para cualquier K permitido. Probar K=3 es útil para detectar errores, pero no sustituye el invariante.

::: traduccion El invariante en notación compacta
Antes de procesar el término k, el acumulador contiene la reducción ordenada de $A_{it}B_{tj}$ para $0\leq t<k$. En aritmética real se escribe $\sum_{t=0}^{k-1}A_{it}B_{tj}$. En FP32 debemos añadir que se ha ejecutado la secuencia concreta de redondeos elegida.
:::

## División de rangos en bloques

Para recorrer N=10 posiciones en bloques de cuatro, un índice exterior selecciona el bloque y uno interior la posición dentro de él. La coordenada es `q=4*b+t`. Los bloques b=0,1,2 producen q de 0 a 11, por lo que necesitamos la guardia `q<10` en el último.

Otra posibilidad separa una región completa de ocho elementos y una cola de dos. Ambas implementaciones pueden ser correctas. La elección afecta a ramas, vectorización y tamaño de código. El compilador puede generar una variante rápida para tamaños múltiplos del bloque y otra general para el resto.

## Intercambiar bucles

Si dos ejes de salida son independientes, intercambiar sus bucles puede conservar el cálculo de cada casilla y cambiar el orden de acceso a memoria. En una matriz por filas, recorrer columnas en el bucle interior suele dar accesos contiguos. Recorrer filas en el interior salta por el stride de fila.

Intercambiar un eje de reducción con otro bucle requiere analizar qué acumuladores siguen vivos y dónde se inicializan. En GEMM, el orden `i,k,j` permite reutilizar A[i,k] a lo largo de varias columnas, pero exige que las salidas C se hayan inicializado adecuadamente antes de acumular.

```c
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) C[i*N+j] = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = A[i*K+k];
        for (int j = 0; j < N; j++) C[i*N+j] += a * B[k*N+j];
    }
}
```

El orden de k dentro de cada salida puede conservarse, pero el tráfico de C y la vectorización cambian. No afirmamos que esta versión sea siempre mejor: es una alternativa que debe medirse.

## Dependencias de memoria

Considera `x[i]=x[i-1]+1` para i creciente. La iteración i utiliza el resultado de la anterior. Reordenarlas libremente cambia la tarea. El hecho de que un bucle tenga muchas iteraciones no implica que sean paralelas.

Un análisis de dependencias identifica lecturas y escrituras que pueden referirse a la misma dirección. Para un compilador tensorial funcional, muchos efectos quedan fuera del grafo de valores. Cuando añadimos actualizaciones in-place, buffers compartidos o comunicación, esas dependencias deben representarse explícitamente.

## Lowering por etapas

No intentamos transformar una red completa directamente en ensamblador con una sola función. Una secuencia útil conserva niveles: operaciones tensoriales, índices y rangos, asignación de buffers, kernels, instrucciones del backend y ejecución. Cada nivel resuelve una clase de decisiones.

Esta separación permite localizar errores. Si la forma del tensor es incorrecta, el problema está antes del mapeo de GPU. Si el C es correcto pero el runtime copia el tamaño equivocado, el problema está después del renderer. Un buen compilador permite inspeccionar las fronteras entre etapas.

## Verificar que no faltan ni sobran posiciones

Una transformación de rangos debe demostrar cobertura y ausencia de duplicación de escrituras, salvo que exista una operación de combinación definida. Para `q=4*b+t`, con b en el rango apropiado y t entre cero y tres, la guardia asegura que solo se escriben posiciones válidas. La división y el resto recuperan b y t para cualquier q válido, lo que demuestra que no se pierde ninguna posición.

Esta forma de argumento reaparecerá al mapear bloques e hilos GPU. La GPU no elimina la necesidad de razonar sobre rangos: la hace más importante, porque un error puede convertirse en una carrera difícil de reproducir.

::: practica Diseñar un tile con cola
Transforma un rango de 23 elementos en bloques de ocho. Escribe los índices generados por el último bloque y una condición que impida accesos inválidos. ¿Cuántas posiciones válidas tiene ese bloque?
:::

**Solución.** Hay tres bloques. El último genera 16 a 23. Son válidos 16 a 22, siete elementos. La condición es `q<23`. Una comprobación `q<=23` incluye un elemento inexistente. Una condición aplicada solo a la escritura no protege una lectura inválida realizada antes.

::: comprueba Antes de continuar
Distingue ejes de salida y reducción, formula un invariante de acumulador y justifica un bloque con cola. Después identifica qué decisión de loop order puede mejorar accesos sin alterar la fórmula de cada salida.
:::

# Reducciones: identidad, orden y cooperación {#ch:reducciones}

## La suma de una caja vacía

Una reducción combina muchos valores en uno. Para comenzar necesita un valor inicial. En una suma, cero permite que añadir el primer elemento produzca ese elemento. En un producto, la identidad es uno. En un máximo, la identidad conceptual para valores extendidos puede ser menos infinito, pero un API puede preferir rechazar una reducción vacía.

Lumbre admite suma vacía y rechaza máximo sobre un eje vacío. Esa elección es explícita. No se deduce de que ambos algoritmos utilicen un acumulador. Las operaciones tienen dominios y contratos diferentes.

## Una reducción por fila

Para una matriz `(M,N)`, cada salida de suma por fila procesa N elementos. Podemos asignar una fila a cada trabajador. Dentro de una fila, el trabajador acumula secuencialmente. Es una implementación sencilla y correcta, pero puede desperdiciar paralelismo cuando hay pocas filas muy largas.

```c
for (int row = 0; row < M; row++) {
    float acc = 0.0f;
    for (int col = 0; col < N; col++) acc += x[row*N+col];
    y[row] = acc;
}
```

La fórmula de dirección mezcla dos decisiones: qué fila produce cada salida y qué eje se reduce. Reducir columnas en vez de filas cambia la secuencia de accesos y el número de resultados.

## Árbol de reducción

Para ocho números, podemos sumar parejas, luego parejas de resultados y finalmente los dos valores restantes. La profundidad de dependencias baja frente a una cadena secuencial. Hay más operaciones que pueden ejecutarse simultáneamente.

```tikz
\begin{center}
\begin{tikzpicture}[x=1cm,y=1cm]
\foreach \x in {0,...,7}{\node[dot,minimum size=6mm] (a\x) at (\x,0) {$x_{\x}$};}
\foreach \x/\p/\q in {0/0/1,1/2/3,2/4/5,3/6/7}{\node[dot] (b\x) at ({2*\x+.5},-1) {$+$};\draw[flow](a\p)--(b\x);\draw[flow](a\q)--(b\x);}
\node[dot](c0) at (1.5,-2){$+$};\node[dot](c1) at (5.5,-2){$+$};
\draw[flow](b0)--(c0);\draw[flow](b1)--(c0);\draw[flow](b2)--(c1);\draw[flow](b3)--(c1);
\node[dot](d)at(3.5,-3){$+$};\draw[flow](c0)--(d);\draw[flow](c1)--(d);
\end{tikzpicture}
\end{center}
```

En enteros sin desbordamiento, la suma conserva el resultado. En FP32 cambia el orden de redondeos. Un árbol puede incluso reducir ciertos errores, pero no garantiza identidad bit a bit con la cadena. El contrato de comparación debe reconocer esa diferencia.

## Particionar y combinar

Si una fila es demasiado grande para un bloque, varios bloques pueden producir sumas parciales. Un segundo kernel las combina. La frontera entre kernels proporciona un punto de orden que no existe automáticamente entre bloques de un mismo kernel ordinario.

Intentar implementar una barrera global haciendo que todos los bloques esperen dentro de un kernel puede bloquear el dispositivo si no todos están residentes simultáneamente. El plan seguro inicial utiliza varios lanzamientos o mecanismos de cooperación explícitamente soportados y correctamente configurados.

## Atomics no significan determinismo numérico

Una operación atómica permite combinar actualizaciones de una ubicación sin perder escrituras bajo su semántica. No garantiza un orden fijo entre todos los productores. En una suma FP32, distintos órdenes pueden dar últimos bits diferentes.

Además, muchas actualizaciones a una sola dirección pueden serializarse. Un diseño jerárquico reduce primero localmente y limita el número de operaciones atómicas globales. Es otra muestra de que corrección, determinismo y rendimiento son tres preguntas separadas.

## Máximo y empates

El máximo de una lista puede aparecer en varias posiciones. Para forward eso no presenta ambigüedad de valor. Para el gradiente hay que elegir una convención de subgradiente en los empates. Lumbre reparte la contribución entre las posiciones iguales al máximo dentro de la reducción.

Ese comportamiento debe probarse explícitamente. Si una optimización cambia una comparación o el tratamiento de NaN, puede alterar tanto la salida como las posiciones a las que se asigna gradiente. Una capa de softmax estable utiliza el máximo como desplazamiento y lo desconecta del gradiente en la implementación para simplificar el camino numérico sin alterar la derivada matemática del softmax finito.

## Promedio y ejes de tamaño uno

El promedio puede implementarse como suma dividida por el número de términos. Si ese número es cero, no debemos dividir por cero y fingir un promedio válido. Si es uno, la reducción puede desaparecer en el valor, aunque quizá deba conservarse o reconstruirse la forma según `keepdim`.

Las dimensiones unitarias son excelentes casos de test porque revelan confusiones entre eliminar un eje y reducirlo manteniéndolo de longitud uno. Las dimensiones cero revelan accesos indebidos e identidades mal elegidas.

## Log-sum-exp

La expresión `log(sum(exp(x)))` aparece en pérdidas de clasificación. Se estabiliza restando un máximo y añadiéndolo al final: `m + log(sum(exp(x-m)))`. Esta transformación evita muchas exponenciales enormes y es la base de una entropía cruzada estable.

No conviene calcular primero softmax, convertir probabilidades diminutas a cero por redondeo y después aplicar log. El resultado puede ser menos estable que calcular directamente log-softmax. El orden algebraico de una fórmula puede tener un impacto práctico grande aunque describa el mismo objeto matemático.

## RMSNorm como combinación de reducciones y operaciones locales

RMSNorm calcula la media de los cuadrados a lo largo de un eje, añade una pequeña constante, toma la raíz y divide cada elemento por ella, con una escala aprendida. La reducción produce una estadística compartida por la fila; la normalización posterior es elemento a elemento.

Un compilador puede fusionar parte del trabajo, pero tiene que conservar la dependencia: no puede normalizar una fila con una estadística que todavía no está completa. En GPU, la combinación de reducción local y escrituras de la fila necesita sincronización entre los participantes.

## Pruebas que separan mecanismos

Para una suma por filas, prueba una sola fila larga, muchas filas de un elemento, dimensiones no potencias de dos y una fila que combine magnitudes muy diferentes. Para un máximo, añade negativos y empates. Para log-sum-exp, añade un desplazamiento grande común a todas las entradas.

Una buena propiedad para softmax es que sumar una constante finita común a toda la fila no cambia matemáticamente las probabilidades. La prueba numérica debe utilizar valores que no provoquen por sí mismos pérdida extrema de información al representar ese desplazamiento.

::: practica Dos fases o una atómica
Debes sumar un millón de valores con muchos bloques GPU. Propón una implementación con sumas parciales y otra con atomics. ¿Qué compararías además del tiempo?
:::

**Solución.** En la primera, cada bloque reduce su región y un segundo kernel combina las parciales. En la segunda, cada bloque puede reducir localmente y añadir una parcial a una salida inicializada. Hay que comparar precisión, repetibilidad entre ejecuciones, coste de inicialización, número de lanzamientos, memoria auxiliar y restricciones de la operación atómica. No basta medir una sola llamada caliente.

::: comprueba Antes de continuar
Explica identidad, árbol, suma parcial y efecto del orden. Distingue una barrera de bloque de una sincronización global y explica por qué una operación atómica no fija necesariamente el resultado bit a bit de una reducción FP32.
:::
