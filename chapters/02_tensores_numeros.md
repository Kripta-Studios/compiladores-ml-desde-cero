# Tensores: números con una organización {#ch:tensores}

::: idea La idea antes del nombre
Un tensor del curso es una colección de números organizada por ejes. Una temperatura es un número; las temperaturas de una semana forman una fila; las de varias ciudades forman una tabla. Añadir ejes no añade magia: añade preguntas para localizar cada valor.
:::

## Escalar, vector y matriz

Un **escalar** es un único valor. Un **vector** es una secuencia con un eje. Una **matriz** tiene dos ejes, que imaginaremos como filas y columnas. Un tensor puede tener más ejes: lote, altura, anchura y canal, por ejemplo. En este libro «tensor» designa la estructura computacional de un array multidimensional; no necesitamos empezar por la definición geométrica de tensores de la física.

Una tabla de dos ciudades y tres días tiene forma `(2, 3)`. El primer número indica cuántas ciudades; el segundo, cuántos días. Contiene seis elementos, no cinco. La forma cuenta combinaciones: para cada una de las dos ciudades existen tres posiciones de día.

| Ciudad / día | Día 0 | Día 1 | Día 2 |
|---|---|---|---|
| Ciudad 0 | 10 | 12 | 15 |
| Ciudad 1 | 8 | 9 | 11 |

El elemento de índices `(1, 2)` vale 11. El rango, en el sentido de número de ejes usado por las APIs de arrays, es dos. No confundas este uso de «rango» con el rango algebraico de una matriz, que es otra propiedad.

## Forma y número de elementos

Una forma se escribe como una tupla. Para `(2, 3, 4)` hay dos grupos, cada uno con tres filas de cuatro números: veinticuatro elementos. Una forma escalar se representa por la tupla vacía `()`, y contiene un elemento. Una forma `(0, 3)` contiene cero elementos. Es importante distinguir el escalar de un array vacío.

::: traduccion La fórmula después de contar
Si la forma es $(d_0,d_1,\ldots,d_{r-1})$, el número de elementos es el producto de sus dimensiones: $d_0d_1\cdots d_{r-1}$. Para la forma sin ejes, el producto vacío se toma como uno. Si alguna dimensión vale cero, el producto vale cero.
:::

Los casos vacíos revelan errores en compiladores: una suma sobre ningún elemento puede tener una identidad bien definida, mientras un máximo vacío necesita una convención adicional o un error. No añadiremos un acceso a la primera posición para «simplificar» el código cuando esa posición no existe.

## Operaciones elemento a elemento

Sumar dos matrices de igual forma significa sumar posiciones correspondientes. Para `A=[[1,2],[3,4]]` y `B=[[10,20],[30,40]]`, la suma es `[[11,22],[33,44]]`. Ningún resultado utiliza toda una fila ni toda una columna; utiliza una pareja de valores.

Multiplicar elemento a elemento también empareja posiciones: el resultado sería `[[10,40],[90,160]]`. En muchas APIs se escribe `A * B`. La multiplicación matricial, que veremos después, hace otra cosa y se suele escribir `A @ B`. La diferencia no es un detalle sintáctico: cambia las dependencias y el coste.

```python
import numpy as np
A = np.array([[1, 2], [3, 4]], dtype=np.float32)
B = np.array([[10, 20], [30, 40]], dtype=np.float32)
assert np.array_equal(A + B, [[11, 22], [33, 44]])
assert np.array_equal(A * B, [[10, 40], [90, 160]])
```

Aquí NumPy es un instrumento para comprobar una definición. Más adelante generaremos nuestros propios bucles para producir esas mismas salidas.

## Broadcasting: reutilizar sin inventar dimensiones

Queremos sumar a cada ciudad una corrección distinta por día: `[1, 0, -1]`. La fila tiene forma `(3,)` y la tabla `(2,3)`. Podemos reutilizar esa fila en ambas ciudades. No necesitamos almacenar físicamente dos copias para expresar la operación.

La regla general alinea las formas por la derecha. Dos dimensiones son compatibles si son iguales o si una de ellas vale uno. Una dimensión ausente se considera uno a efectos de esta comparación. Por tanto, `(2,3)` y `(3,)` se alinean como `(2,3)` y `(1,3)` y producen `(2,3)`.

::: ejemplo Tres comparaciones completas
`(4,1,7)` con `(3,7)` se alinea con `(1,3,7)` y produce `(4,3,7)`. En cambio, `(4,2)` con `(3,)` falla porque 2 y 3 no coinciden y ninguna vale uno. Finalmente, `(0,3)` con `(1,3)` produce `(0,3)`: no aparecen datos de la nada. Usar simplemente el máximo de cada pareja daría uno para la primera dimensión y sería incorrecto.
:::

Broadcasting no significa «todas las formas se pueden combinar». Tampoco significa concatenar arrays. Es una regla concreta para reutilizar valores a lo largo de ejes de tamaño uno. En el backward, esa reutilización se transforma en una suma de contribuciones.

## Reducciones: muchas posiciones producen una

Sumar los tres días de cada ciudad transforma una tabla `(2,3)` en una fila `(2,)`: 37 y 28. Hemos reducido el eje de los días. Sumar todas las ciudades y todos los días produce el escalar 65. Mantener un eje de longitud uno, con `keepdim`, permite recordar qué posición dimensional se ha reducido.

La operación no es elemento a elemento porque un resultado depende de varios datos. En un programa secuencial utilizaremos un acumulador. En uno paralelo necesitaremos combinar resultados parciales. Cambiar el orden de esa combinación puede afectar a la coma flotante, aunque la suma matemática sea la misma.

```python
A = np.array([[10, 12, 15], [8, 9, 11]], dtype=np.float32)
assert np.array_equal(A.sum(axis=1), [37, 28])
assert A.sum() == 65
assert A.sum(axis=1, keepdims=True).shape == (2, 1)
```

`axis=1` identifica el segundo eje, no la segunda fila. Esta diferencia merece una pausa: un eje es una dirección de organización; una fila concreta es una selección dentro de ella.

## Producto escalar: comparar dos listas de características

Supón que un objeto tiene características `[2,3]` y damos importancia `[4,5]` a esas características. Multiplicamos parejas y sumamos: `2*4 + 3*5 = 23`. El resultado es un único número. A esta operación se la llama **producto escalar**.

La longitud de ambas listas debe coincidir. Cada posición tiene una pareja y todas las parejas contribuyen. La expresión compacta es $\sum_k a_k b_k$. El signo de suma significa repetir la multiplicación para cada índice permitido y acumularla; no introduce una operación distinta de la que acabamos de realizar.

## Multiplicación matricial desde una sola casilla

Sea `A=[[1,2,3],[4,5,6]]` y `B=[[7,8],[9,10],[11,12]]`. A tiene dos filas y tres columnas; B tiene tres filas y dos columnas. El resultado tendrá dos filas y dos columnas. Para su casilla `(0,0)`, multiplicamos la fila 0 de A por la columna 0 de B: `1*7 + 2*9 + 3*11 = 58`.

La casilla `(0,1)` vale `1*8 + 2*10 + 3*12 = 64`. La `(1,0)` vale `4*7 + 5*9 + 6*11 = 139`. La `(1,1)` vale `4*8 + 5*10 + 6*12 = 154`. Hemos construido todo el resultado sin esconder ninguna suma.

::: traduccion Las tres letras de GEMM
Usaremos $M$ para las filas de A, $K$ para sus columnas y $N$ para las columnas de B. A tiene forma $(M,K)$, B tiene forma $(K,N)$ y C tiene forma $(M,N)$. La dimensión compartida K es la que se reduce:
```math
C_{ij}=\sum_{k=0}^{K-1} A_{ik} B_{kj}.
```
:::

Cada salida necesita K multiplicaciones y, según cómo contemos la inicialización, K o K-1 sumas. En informes de rendimiento es habitual contar aproximadamente $2MNK$ operaciones de coma flotante. Debes declarar esa convención al convertir tiempo en FLOP/s.

## Transponer no es invertir

Transponer una matriz intercambia filas y columnas. La matriz `[[1,2,3],[4,5,6]]` se convierte en `[[1,4],[2,5],[3,6]]`. La forma cambia de `(2,3)` a `(3,2)`. No estamos calculando la inversa algebraica ni cambiando los valores.

Un compilador puede representar la transposición como un cambio de cómo calcula direcciones, sin copiar datos inmediatamente. Sin embargo, que una vista no copie no significa que todos sus consumidores accedan con la misma eficiencia. Una lectura que antes recorría posiciones contiguas puede pasar a saltar distancias mayores.

## Lotes de matrices

Una red suele procesar varios ejemplos a la vez. Si A tiene forma `(B,M,K)` y Bmat tiene forma `(B,K,N)`, hay B multiplicaciones independientes. El resultado es `(B,M,N)`. La letra B de lote no debe confundirse con el nombre de la segunda matriz; por eso en explicaciones largas escribiremos «lote» o `batch`.

Lumbre admite multiplicaciones con dos o más ejes y broadcasting de los ejes de lote. No admite por esa misma función todas las convenciones especiales que una API externa pueda tener para vectores de rango uno. La firma de una operación forma parte del contrato, incluso cuando el nombre coincide entre bibliotecas.

## Ejercicios resueltos

::: practica Forma, operación y resultado
Tienes X de forma `(5,4)` y un sesgo de forma `(4,)`. ¿Qué forma tiene X más el sesgo? Después multiplicas por W de forma `(4,3)`. ¿Qué forma sale? ¿Cuántas multiplicaciones escalares requiere esta multiplicación matricial?
:::

**Solución.** El sesgo se reutiliza en las cinco filas, por lo que la suma conserva `(5,4)`. La multiplicación por W produce `(5,3)`. Cada una de sus quince salidas acumula cuatro productos: sesenta multiplicaciones. El sesgo no altera la dimensión que debe coincidir con las filas de W.

::: practica Una igualdad aparente
Con una matriz identidad, ciertas multiplicaciones dejan la otra matriz igual. ¿Es una buena única prueba para un compilador de GEMM? ¿Y con matrices llenas de cero?
:::

**Solución.** Son pruebas útiles de casos particulares, pero insuficientes. Un error que omita productos o invierta índices puede quedar oculto por ceros, simetrías o patrones especiales. Añade datos no simétricos, dimensiones distintas y un ejemplo calculado a mano.

::: comprueba Antes de continuar
Calcula una casilla de GEMM, explica una reducción y decide compatibilidad de broadcasting. Debes poder hacer estas tres tareas sin recurrir a «lo hace NumPy». Las siguientes transformaciones del compilador se apoyan en esa comprensión.
:::

# Los números de la máquina no son los números de la pizarra {#ch:numeros}

## Una regla con pocas marcas

Imagina una regla que solo tiene marcas de centímetro. Una longitud de 2,4 centímetros debe aproximarse al utilizarla. La longitud real no cambia, pero la representación tiene precisión limitada. Los formatos de coma flotante funcionan con otra distribución de marcas: cerca de números pequeños hay más detalle absoluto; a medida que crece la magnitud, la separación entre valores representables también crece.

La metáfora no cubre todos los detalles, pero aclara la idea central: guardar y operar con un número puede introducir redondeo. Por eso una transformación correcta sobre números reales puede no conservar exactamente el programa que ejecuta una CPU o una GPU.

## Bits, bytes y tipos

Un bit distingue dos estados. Ocho bits forman un byte. Un `float32` ocupa cuatro bytes y un `float64` ocho. La memoria disponible se mide en bytes, mientras la forma de un tensor cuenta elementos. Confundir ambas unidades produce presupuestos de memoria erróneos.

Una matriz de 1024 por 1024 elementos `float32` contiene 1.048.576 valores y ocupa 4.194.304 bytes, es decir, 4 MiB. El prefijo MiB indica potencias de 1024; MB suele indicar un millón de bytes. Si comparas cifras de herramientas distintas, comprueba qué unidad utiliza cada una.

Un tipo numérico especifica más que el tamaño. También define cómo interpretar los bits: entero, coma flotante, máscara o puntero. Los mismos cuatro bytes pueden representar cosas radicalmente diferentes. Cambiar la etiqueta sin transformar el contenido es un `bitcast`, no una conversión numérica ordinaria.

## Signo, exponente y fracción

Un formato de coma flotante representa aproximadamente un signo, una parte significativa y una escala de potencia de dos. El exponente controla la escala; la fracción almacena detalle dentro de esa escala. FP32 dispone de ocho bits de exponente y veintitrés bits de fracción almacenada, con una unidad implícita para números normales. BF16 y FP16 reparten de forma diferente sus dieciséis bits. [@ieee-nvidia]

No necesitas memorizar todas las codificaciones para escribir tu primer kernel. Sí debes entender que **rango** y **precisión** son propiedades diferentes. Un formato puede representar magnitudes grandes y ofrecer pocos dígitos de detalle; otro puede distinguir mejor números cercanos pero desbordarse antes.

| Formato | Bits totales | Exponente | Fracción almacenada |
|---|---|---|---|
| FP32 | 32 | 8 | 23 |
| FP16 | 16 | 5 | 10 |
| BF16 | 16 | 8 | 7 |

Esta tabla describe los formatos habituales usados en el curso. No convierte FP8 o FP4 en nombres de una única codificación universal. Sus variantes requieren especificar exponente, fracción, valores especiales, escala y reglas de conversión.

## El orden de una suma puede cambiar el resultado

En números reales, `(a+b)+c` y `a+(b+c)` son iguales. En coma flotante las sumas intermedias se redondean. Para una magnitud muy grande y otra pequeña, la pequeña puede desaparecer al sumar porque no hay una marca representable cercana que la conserve.

```python
import numpy as np
a = np.float32(1e20)
b = np.float32(-1e20)
c = np.float32(3.0)
izquierda = np.float32(np.float32(a + b) + c)
derecha = np.float32(a + np.float32(b + c))
print(izquierda, derecha)
```

En el comportamiento FP32 ordinario de este ejemplo, la primera agrupación produce 3 y la segunda 0. El punto importante no es memorizar esas magnitudes, sino localizar el redondeo que pierde información. Un compilador que reordene sumas debe tener permiso para cambiar esa semántica o aceptar un criterio aproximado declarado.

::: cuidado Una reescritura no es una identidad porque resulte familiar
Factorizar `a*b + a*c` como `a*(b+c)` cambia operaciones y redondeos. Reemplazar una división por una multiplicación aproximada por el recíproco también puede cambiar el error. Estas transformaciones requieren un contrato numérico, no una confianza general en el álgebra escolar.
:::

## Infinito, NaN y ceros con signo

La coma flotante incluye valores especiales. Un desbordamiento puede producir infinito. Operaciones indefinidas pueden producir NaN, «no es un número». Existen además cero positivo y cero negativo, que pueden distinguirse en ciertas operaciones y observaciones.

La reescritura `x * 0 -> 0` parece inocente. Sin embargo, con `x` infinito el producto puede ser NaN; con NaN también puede mantenerse NaN; el signo del cero puede importar. Por eso Lumbre separa las simplificaciones exactas de índices enteros de las posibles optimizaciones numéricas de tensores.

Comparar NaN con una tolerancia habitual tampoco funciona como comparar dos números finitos. Un test que decide «todos los resultados son aproximadamente iguales» debe especificar qué hacer con NaN e infinitos. Aceptar NaN en ambas salidas puede esconder un error que vuelve inválido todo el entrenamiento.

## FMA: una multiplicación y una suma con un único redondeo

Una instrucción fused multiply-add calcula una expresión de la forma `a*b+c` redondeando una sola vez el resultado combinado. Una multiplicación seguida de una suma ordinarias puede redondear dos veces. FMA suele ser valiosa para rendimiento y precisión, pero no siempre reproduce bit a bit la secuencia no fusionada. [@ieee-nvidia]

El backend CPU de referencia de Lumbre utiliza `-ffp-contract=off`. Esto ayuda a mantener una política explícita al comparar recorridos. No afirma que FMA sea mala. En un proyecto de rendimiento podrás habilitarla, medir su efecto y cambiar la definición de equivalencia de manera documentada.

El compilador también debe distinguir la contracción de una multiplicación-suma de la fusión de kernels. Podemos evitar escribir un tensor intermedio en memoria sin necesariamente introducir FMA; son transformaciones relacionadas, pero no idénticas.

## Error absoluto y relativo

El error absoluto compara la distancia: `|resultado - referencia|`. El relativo compara esa distancia con una escala, normalmente ligada al valor de referencia. Si la referencia está cerca de cero, dividir por ella puede producir una medida enorme o indefinida aunque la diferencia absoluta sea pequeña.

Una regla práctica frecuente acepta una casilla si su error absoluto es menor que una tolerancia absoluta más una tolerancia relativa multiplicada por la magnitud de la referencia. Las tolerancias no deben escogerse después de observar el peor error para forzar un aprobado. Primero fija el contrato y luego ejecuta el test.

```math
|y-\widehat y|\leq \mathrm{atol}+\mathrm{rtol}\,|\widehat y|.
```

::: ejemplo Leer una tolerancia con números
Si la referencia es 100, `atol=0.001` y `rtol=0.0001`, el margen es 0,011. Si la referencia es cero, el margen es 0,001. La primera componente protege los valores pequeños; la segunda adapta la comparación a la escala. No se deduce que estos valores sean adecuados para cualquier modelo.
:::

## Reducciones largas y acumulación

Sumar mil productos acumula más oportunidades de redondeo que sumar dos. Una reducción por árbol cambia el orden respecto al recorrido secuencial. Un tensor core puede multiplicar entradas de menor precisión y acumular en FP32. Hay que describir todos esos pasos al comparar con una referencia de FP64.

Una referencia de mayor precisión es útil para aproximar la respuesta real, pero no equivale necesariamente a la semántica exacta de la API que intentamos reproducir. Conviene tener dos preguntas separadas: ¿reproduce el contrato del framework? ¿Qué error tiene respecto a una referencia más precisa?

Técnicas como acumulación compensada, sumas por bloques o acumuladores más anchos pueden reducir error con costes distintos. El compilador necesita conocer cuándo la aplicación valora reproducibilidad bit a bit, estabilidad numérica o velocidad dentro de una tolerancia.

## Softmax y el truco que evita un desbordamiento

Softmax convierte puntuaciones en pesos positivos que suman uno. Primero exponencia cada puntuación y después divide entre la suma de exponenciales. Si las puntuaciones son 1000 y 1001, calcular directamente esas exponenciales puede desbordar. Restar el máximo convierte el problema en exponenciar -1 y 0.

No cambia el resultado matemático porque todos los términos se multiplican por el mismo factor, que se cancela entre numerador y denominador. Sí cambia favorablemente el comportamiento de la máquina al evitar magnitudes enormes. En el capítulo de atención construiremos una versión que actualiza ese máximo mientras procesa bloques.

```math
\frac{e^{x_i}}{\sum_j e^{x_j}}
=\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},\qquad m=\max_j x_j.
```

Aquí cada símbolo ya tiene un trabajo: `i` selecciona una salida; `j` recorre todas las puntuaciones; `m` es el mayor valor. La fórmula no es un adorno que se añade a la implementación: explica por qué la transformación conserva la intención.

## Qué comprobar antes de optimizar

Una batería numérica útil incluye valores pequeños, magnitudes desiguales, signos opuestos, ceros, dimensiones no múltiplos del tile y casos que estresen las reducciones. Los valores especiales se prueban si pertenecen al dominio de la operación; en otro caso se valida y documenta su exclusión.

Para un entrenamiento, controla además pérdida finita, norma de gradiente, magnitud de parámetros y estados del optimizador. Una pérdida decreciente no demuestra que todos los gradientes sean correctos: algunas direcciones equivocadas pueden mejorar temporalmente una tarea diminuta.

::: practica Clasificar una transformación
¿Es siempre válida la regla `n*0 -> 0` cuando n es un índice entero matemático sin desbordamiento? ¿Y cuando n es un tensor FP32 que puede contener infinito? ¿Es el mismo problema que cambiar el orden de dos bucles independientes?
:::

**Solución.** En el dominio entero indicado sí es una identidad exacta. En FP32 con valores especiales no lo es en general. Cambiar el orden de bucles independientes puede conservar cada operación individual, mientras reordenar una reducción cambia la secuencia de redondeos. Hay que identificar el dominio y las dependencias, no aplicar una regla por parecido visual.

::: comprueba Antes de continuar
Distingue almacenamiento y valor, error absoluto y relativo, precisión y rango, fusión de kernels y FMA. Explica por qué el compilador tiene que conocer el contrato de coma flotante antes de simplificar una expresión.
:::
