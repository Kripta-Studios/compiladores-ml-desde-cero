# Un modelo se convierte en varios kernels {#ch:schedule}

## La diferencia entre fórmula y plan de ejecución

Un grafo describe dependencias de valores. Un plan de ejecución decide qué operaciones se agrupan, dónde se almacenan resultados y en qué orden se lanzan tareas. A una unidad de trabajo compilada para ejecutarse en un dispositivo la llamaremos **kernel de cálculo**. No es el kernel del sistema operativo Linux.

Para `y=(x+1)*2`, un único kernel suele ser suficiente. Para una normalización que necesita una reducción completa y después combina su resultado con muchas posiciones, el compilador puede generar varios kernels o uno cooperativo especializado. La respuesta depende del backend y de la estrategia de memoria.

::: idea La pregunta del scheduler
¿Qué valores deben existir en memoria, cuáles pueden calcularse dentro de sus consumidores y qué dependencias deben respetar los lanzamientos? El scheduler no inventa una nueva función matemática: escoge una realización de la que ya existe.
:::

## Materializar: dar una ubicación a un resultado

Un resultado **materializado** se escribe en un buffer para que otros lo lean. Uno no materializado puede quedar como expresión dentro de otro kernel. La materialización cuesta memoria y tráfico, pero puede evitar repetir trabajo o facilitar una implementación eficiente.

Supón `a=exp(x)`, `b=a+1`, `c=a*2`. Si a se inserta en ambos consumidores, se calcula la exponencial dos veces. Si se materializa una vez, ambos leen el mismo resultado. La mejor opción depende del coste de exp, del tamaño de x, de la fusión posible y del tráfico.

No todas las operaciones compartidas merecen materializarse siempre. Una suma escalar barata puede repetirse sin coste relevante. Lumbre utiliza una heurística sencilla basada en número de usos, clase de operación y profundidad; el libro enseña a sustituirla por un modelo más rico cuando existan pruebas que lo justifiquen.

## Fronteras naturales y fronteras elegidas

Las salidas declaradas necesitan conservarse para el usuario. Los parámetros son entradas persistentes. Reducciones, GEMM, gather y llamadas especializadas suelen formar fronteras útiles porque requieren recorridos distintos o implementaciones propias.

Pero «suele» no es «siempre». Una biblioteca GEMM puede admitir un epílogo que añada sesgo y activación sin escribir una salida intermedia. Un kernel de atención puede integrar dos productos matriciales y un softmax por bloques. El scheduler general puede reconocer esos patrones y delegar en una implementación especializada con contrato explícito.

## Dependencias de un kernel fusionado

Si un kernel contiene una expresión de varios nodos, sus entradas reales son los buffers materializados que aparecen al recorrer esa expresión. No necesitamos pasar como argumento cada suma interna, porque no tiene ubicación independiente.

Un algoritmo de dependencias recorre desde la raíz del kernel. Cuando encuentra otro nodo materializado, lo registra y se detiene en esa rama. Cuando encuentra una operación interna fusionable, continúa hacia sus entradas. Esta distinción evita tanto argumentos redundantes como lecturas de buffers que nunca se escribieron.

```diagram
Grafo de valores | Selección de materializaciones | Dependencias entre kernels
```

## Orden topológico del plan

Todo kernel que produce una entrada de otro debe ejecutarse antes, salvo que el runtime utilice una sincronización explícita equivalente. Un orden topológico secuencial es una primera solución correcta. Más adelante podremos superponer kernels independientes o transferencias, pero solo después de representar sus dependencias.

El runtime GPU didáctico de Lumbre lanza en una secuencia y sincroniza al finalizar `run`. Esta política es conservadora y fácil de observar. No pretende reproducir la planificación asíncrona avanzada de un sistema distribuido.

## Por qué una fusión grande puede ser peor

Fusionar muchas operaciones mantiene más valores intermedios vivos en registros. Si el hardware no dispone de suficientes registros por hilo o grupo, parte del estado puede derramarse a memoria, un fenómeno llamado **spill**. También puede disminuir el número de grupos residentes y limitar la capacidad de ocultar latencia.

El código generado puede crecer hasta hacer lenta la compilación o impedir simplificaciones del compilador de bajo nivel. Por eso hay que medir coste de primera compilación, tamaño de módulo y rendimiento caliente. Una optimización que solo considera número de lanzamientos puede empeorar la experiencia total.

## Epílogos de GEMM

Después de `C=A@B`, muchas redes calculan `relu(C+b)` o una función similar. Un epílogo aplica esas operaciones antes de escribir la salida final. Así evita volver a leer y escribir todo C en un kernel separado.

La implementación debe fijar dónde se realiza cada conversión de tipo. Aplicar activación en FP32 antes de redondear a FP16 no es necesariamente igual a redondear primero y aplicar después. La fusión tiene que preservar el contrato numérico elegido, no solo la forma de las expresiones.

CUTLASS y hipBLASLt ofrecen mecanismos y configuraciones de GEMM y epílogos en sus respectivos ecosistemas. Estudiarlos ayuda a separar la operación matemática de la variedad de realizaciones de hardware. No significa que una llamada a esas bibliotecas sea automáticamente compatible con cualquier layout o tipo. [@cutlass;@hipblaslt]

## Scheduling con efectos

Una actualización de parámetros no puede adelantarse a un kernel que todavía necesita los valores viejos. Una copia entre dispositivos no puede consumirse antes de que termine. Un buffer no puede liberarse mientras una operación asíncrona lo utiliza.

Estas dependencias no siempre aparecen como aristas de valores puras. Los compiladores pueden introducir tokens de efecto, eventos, regiones o relaciones explícitas de orden. El nombre concreto cambia entre sistemas; la obligación de representar el efecto no cambia.

Lumbre utiliza una barrera conceptual simple: evaluar todo el grafo de nuevos valores y después llamar a `assign`. Rechaza usar directamente otro parámetro como salida de actualización para evitar ciclos de copia que pisarían un valor antes de leerlo. Una implementación más general podría utilizar copias temporales o análisis de aliasing.

## Programas con varias salidas

Un programa de entrenamiento produce más que la pérdida. También puede producir gradientes, normas, nuevos momentos del optimizador y parámetros actualizados. Si esas salidas se declaran, el plan de memoria debe conservarlas hasta que el runtime termine de leerlas o comprometerlas.

Eliminar una salida del conjunto puede permitir reutilizar su buffer antes. Por tanto, cambiar qué observamos puede cambiar el plan y el rendimiento. Un benchmark que solicita todos los intermedios para depuración no mide exactamente el mismo programa que una ejecución de producción que solo conserva la salida final.

## Un experimento de fusión

Construye una cadena de cinco operaciones elemento a elemento y compárala con `fuse=False`. Guarda cantidad de kernels, memoria de arena, tiempo de compilación y tiempo de ejecución residente. Después introduce un subgrafo compartido por dos salidas y repite.

La hipótesis razonable es que la fusión reduzca tráfico en la cadena simple, pero el experimento debe permitir que el resultado la contradiga. Para tensores diminutos puede dominar el coste de llamada; para expresiones complejas puede aumentar el tiempo de compilación. La conclusión debe explicar el régimen observado.

## Preguntas de revisión de un scheduler

Al revisar un plan, sigue un valor desde su producción hasta su último uso. Comprueba que el productor se ejecuta antes, que existe un buffer o expresión válido, que la forma es consistente y que ninguna escritura lo destruye prematuramente. Después examina las salidas y los efectos que no se ven en una sola fórmula.

Una visualización de kernels es útil si muestra dependencias, tamaños y buffers, no solo cajas con nombres. La originalidad de un proyecto puede estar en hacer observable este recorrido y detectar violaciones automáticamente.

::: practica Escoger materializaciones
Tienes `a=x*2`, `b=exp(a)`, `c=b+1`, `d=b*b`, y necesitas c y d. Propón un plan sencillo de dos o tres kernels. Explica dónde evitarías repetir exp y qué cambiaría si solo necesitases c.
:::

**Solución.** Un plan materializa b calculando `exp(x*2)` en el primer kernel y produce c y d en uno o dos kernels posteriores según la capacidad de múltiples salidas. Así no repite exp. Si solo se necesita c, toda la cadena puede fusionarse en una única salida. La propuesta no demuestra que sea el plan más rápido; define una alternativa correcta y fácil de medir.

::: comprueba Antes de continuar
Distingue grafo de valores y grafo de kernels. Explica materialización, dependencia, epílogo y efecto. Debes poder identificar por qué un nodo aparece como argumento de un kernel o queda insertado en su cuerpo.
:::

# Memoria del programa: reservar, reutilizar y no pisar el pasado {#ch:arena}

## Un almacén compartido por tareas sucesivas

Si cada resultado intermedio conserva un buffer para siempre, un modelo puede consumir mucha más memoria de la necesaria. Una vez que un valor ya no se utilizará, su espacio puede servir para otro. El problema consiste en saber exactamente cuándo termina su vida útil.

Imagina tres bandejas y una secuencia de tareas. La primera prepara una mezcla; la segunda la consume por completo; la tercera prepara otra mezcla del mismo tamaño. Podemos reutilizar la bandeja después de la segunda tarea, pero no mientras esa tarea todavía lee de ella.

## Vida útil y último uso

Para cada buffer registramos el kernel que lo produce y el último kernel que lo consume. Las salidas declaradas se mantienen vivas hasta el final observable del programa. Los parámetros persistentes se conservan entre ejecuciones y no se mezclan sin más con temporales.

| Valor | Se produce | Último uso | Puede reutilizarse después de |
|---|---|---|---|
| a | Kernel 0 | Kernel 2 | Kernel 2 completo |
| b | Kernel 1 | Kernel 3 | Kernel 3 completo |
| salida | Kernel 3 | Lectura del usuario | Lectura o siguiente contrato |

Si un kernel lee a y escribe un resultado nuevo, reutilizar exactamente el espacio de a dentro de ese kernel exige analizar si los accesos pueden solaparse de forma segura. El planificador simple evita esa situación: libera solo valores cuyo último uso es estrictamente anterior al kernel actual.

## Arena y offsets

Una **arena** es una reserva grande de memoria dividida en regiones. Cada tensor materializado recibe un offset y un tamaño. El runtime suma el offset al puntero base para obtener la dirección del tensor. Esto reduce muchas reservas pequeñas y hace visible el presupuesto total.

Lumbre alinea regiones a múltiplos de 64 bytes. La alineación facilita ciertos accesos y evita direcciones arbitrarias, pero desperdicia algo de espacio en tensores pequeños. La cifra 64 es una decisión de esta implementación, no una ley universal de todos los dispositivos.

## Best fit y fragmentación

Cuando una región queda libre, el planificador puede guardarla en una lista. Para una nueva petición, una estrategia best fit escoge la región libre más pequeña que todavía sea suficiente. Si sobra espacio, conserva el resto como otro fragmento.

La estrategia puede fragmentar la memoria: existen suficientes bytes libres en total, pero repartidos en trozos demasiado pequeños. Un planificador más avanzado combina regiones vecinas, reordena decisiones o calcula una asignación global. La versión didáctica no pretende resolver óptimamente todos los casos.

::: ejemplo Reutilización con tamaños
Queda libre una región de 1024 bytes y necesitamos 256. Podemos asignar los primeros 256 y devolver los 768 restantes a la lista libre. Si más tarde se liberan dos regiones contiguas de 256, una política que no las combine no podrá atender con ellas una petición única de 512, aunque físicamente estén juntas. Ese es un ejemplo de fragmentación administrativa.
:::

## Alias: dos nombres, una dirección

Dos vistas pueden apuntar al mismo almacenamiento. Dos parámetros también podrían recibir arrays externos que se solapan. El runtime de Lumbre copia los datos de entrada a regiones propias, lo que simplifica el contrato de aliasing externo, a costa de una copia explícita.

En un sistema zero-copy, el compilador debe conocer mejor los solapamientos. Marcar punteros como `restrict` en C promete ciertas ausencias de aliasing. Si el programa viola esa promesa, el compilador C puede generar resultados inesperados bajo las reglas del lenguaje. No se utiliza `restrict` como un talismán de rendimiento sin comprobar su precondición.

## Actualizaciones de entrenamiento

Supón que calculas nuevos pesos a partir de pesos viejos y gradientes. Si sobrescribes un peso mientras otro gradiente todavía lo necesita, mezclas estados de dos iteraciones. El error puede producir una pérdida que aparentemente disminuye y, sin embargo, implementar otro algoritmo.

Lumbre construye un grafo funcional para nuevos parámetros y momentos. Después de ejecutarlo, `assign` copia las salidas hacia las regiones persistentes. Esa frontera es una simplificación pedagógica poderosa: permite razonar sobre una iteración con un estado inicial y otro final bien separados.

## Memoria asíncrona

En CPU secuencial, al volver de una función normalmente han terminado sus accesos ordinarios. En GPU, lanzar un kernel puede devolver el control antes de que termine. La vida útil física de un buffer depende entonces de eventos y streams, no solo de la posición de una llamada en el programa anfitrión.

Un allocator asíncrono debe respetar qué streams utilizan cada región. Liberar desde el host no significa que los consumidores del dispositivo hayan acabado. La implementación del curso sincroniza al finalizar cada `run`; una versión avanzada puede eliminar esa sincronización global, pero debe reemplazarla por dependencias correctas.

## Pico de memoria y memoria acumulada

La suma de tamaños de todos los intermedios creados durante una ejecución no es el pico simultáneo. Un plan puede producir un terabyte acumulado de temporales a lo largo del tiempo y necesitar solo unos pocos gigabytes a la vez. Para decidir si un modelo cabe importa el pico, junto con reservas persistentes y workspace.

Tampoco basta sumar pesos y activaciones visibles. El optimizador, gradientes, buffers de comunicación, cachés de compilación, áreas de bibliotecas y memoria reservada por allocators pueden contribuir. En el capítulo de escalado construiremos un presupuesto completo por categoría.

## Recomputation frente a almacenamiento

Guardar una activación permite utilizarla en backward sin repetir forward. Recalcularla evita mantenerla viva durante mucho tiempo. Este intercambio se llama checkpointing o rematerialización de activaciones, distinto de guardar un checkpoint de entrenamiento en disco.

La decisión depende de coste de recomputar, tamaño del valor y momento de uso. Una operación barata que produce un tensor grande es una buena candidata conceptual. Una operación muy costosa con salida pequeña puede merecer almacenamiento. Un compilador puede automatizar parte de esa elección mediante análisis de vida útil y costes.

## Verificar un plan de memoria

Una prueba útil ejecuta el mismo grafo con reutilización activada y desactivada. Deben coincidir sus salidas bajo el contrato numérico. Añade grafos con ramas, valores compartidos y varias salidas. Los grafos lineales simples no estresan todos los errores de vida útil.

Para depuración, rellena regiones liberadas con patrones especiales o utiliza herramientas de detección de accesos inválidos cuando el backend lo permita. Ese modo no debe mezclarse con el benchmark final sin declararlo: el relleno añade trabajo y puede cambiar cachés.

## Un certificado sencillo de no solapamiento

Dos buffers pueden compartir un intervalo de bytes si sus vidas útiles no se solapan. Si sus vidas sí se solapan, sus regiones de bytes deben ser disjuntas, salvo un alias explícito permitido por el contrato. Esa condición puede comprobarse automáticamente para cada pareja de buffers materializados.

Es una forma de convertir una propiedad difícil de observar mediante salidas en una verificación estructural. No demuestra por sí sola que los índices internos de cada kernel sean válidos, pero cubre una clase diferente de errores.

::: practica Encontrar un uso tardío
Un tensor a se produce en el kernel 0 y se utiliza en los kernels 1 y 4. Un planificador lo libera después del kernel 1 porque «ya se ha consumido». ¿Qué dato olvidó? Diseña un grafo pequeño que haga visible el error.
:::

**Solución.** Olvidó que el consumo no es único: debe registrar el último uso, aquí el kernel 4. Un ejemplo es una rama corta que utiliza a inmediatamente y otra rama larga que vuelve a combinarlo al final. Reutilizar su región durante la rama larga cambia el valor observado por la combinación final.

::: comprueba Antes de continuar
Debes poder dibujar intervalos de vida, distinguir memoria persistente y temporal, y explicar por qué último uso igual al kernel actual no permite una reutilización arbitraria. También debes diferenciar checkpoint de activaciones y checkpoint en disco.
:::

# Descomponer modelos: operaciones generales y límites explícitos {#ch:ops_modelos}

## Un catálogo pequeño puede expresar muchas redes

Las redes no necesitan una instrucción de hardware para cada nombre de capa. Una capa lineal se expresa mediante GEMM y sesgo. Una normalización combina reducciones y operaciones locales. Una convolución puede expresarse mediante accesos a ventanas, productos y sumas. La atención combina productos matriciales, máscara, softmax y otro producto.

Esta composicionalidad explica por qué un núcleo relativamente pequeño puede ejecutar modelos variados. No demuestra que cualquier operación imaginable ya esté soportada, ni que la descomposición general sea rápida. Corrección por descomposición y rendimiento por especialización son etapas diferentes.

## Capa lineal

Para una entrada de forma `(lote,características)` y una matriz de pesos `(características,salidas)`, GEMM produce `(lote,salidas)`. Un sesgo de forma `(salidas,)` se añade por broadcasting. Si cada fila representa un ejemplo independiente, cambiar el orden de filas solo cambia el orden de resultados.

Una red de varias capas añade funciones no lineales entre esas transformaciones. Sin ellas, ciertas cadenas de matrices podrían combinarse en otra transformación lineal, aunque el entrenamiento y la parametrización tengan propiedades distintas.

## Activaciones

ReLU devuelve el máximo entre x y cero. Sigmoid comprime valores a un intervalo entre cero y uno. Tanh devuelve valores entre menos uno y uno. SiLU multiplica x por sigmoid(x), y SwiGLU utiliza una rama de compuerta junto con otra proyección.

Cada activación necesita un contrato para valores extremos y una regla de gradiente. Una aproximación rápida puede ser aceptable si su error y su efecto en entrenamiento se evalúan. No debe sustituirse una función por otra únicamente porque ambas tienen una curva parecida.

El núcleo ejecutable dispone de exp, log, sqrt y tanh, a partir de las que se construyen varias de esas funciones. Para ReLU puede utilizarse una condición y `where`. No hay una biblioteca oculta que resuelva automáticamente cualquier activación no declarada.

## Convolución unidimensional desde una ventana

Supón una señal `[1,2,3,4]` y un filtro `[2,1]`, sin padding y con paso uno. La primera ventana `[1,2]` produce `1*2+2*1=4`; la segunda `[2,3]` produce 7; la tercera `[3,4]` produce 10. Hemos aplicado el mismo conjunto de pesos a posiciones distintas.

En aprendizaje automático muchas APIs llaman convolución a una correlación cruzada, sin invertir el filtro. El contrato debe aclarar esa convención. Cambiar la orientación del filtro puede pasar desapercibido con filtros simétricos, por lo que conviene utilizar uno asimétrico en el test.

## Dimensiones de salida de una convolución

Con longitud de entrada L, filtro de longitud R, padding p por lado, dilatación d y stride s, el tamaño efectivo del filtro es `d*(R-1)+1`. La última ventana debe caber dentro de la región extendida. De ahí se obtiene el número de posiciones válidas.

```math
L_{\text{salida}}=
\left\lfloor\frac{L+2p-d(R-1)-1}{s}\right\rfloor+1.
```

La fórmula presupone parámetros válidos y una convención simétrica de padding. Para casos donde no cabe ninguna ventana hay que definir si la API produce una dimensión cero o rechaza la configuración. No se debe permitir que un tamaño negativo llegue a una reserva de memoria.

## Convolución bidimensional

En una imagen, cada salida recorre filas y columnas del filtro, además de canales de entrada. Una disposición posible es NCHW: lote, canal, altura, anchura. Otra es NHWC. El nombre de la capa no fija por sí solo el layout físico.

Para cada lote n, canal de salida o y posición `(y,x)`, se suman productos de pesos y entradas de la ventana correspondiente. El acceso incorpora stride, dilatación y padding. Es una reducción multidimensional con un patrón de movimiento; los conceptos anteriores ya bastan para describirla correctamente.

## Im2col: hacer explícitas las ventanas

Podemos convertir cada ventana en una fila de una matriz y aplicar GEMM. Esa transformación se conoce como im2col en una de sus formas habituales. Simplifica la reutilización de un kernel GEMM eficiente, pero puede crear un intermedio grande que repite muchos píxeles.

Una implementación implícita calcula los índices de ventana durante la carga y evita materializar toda esa matriz. El intercambio es claro: menos memoria intermedia, más complejidad de índices y de scheduling. No existe una elección universalmente mejor para todas las formas.

::: ejemplo Estimar el crecimiento
Una imagen de un canal con H por W posiciones y filtro de 3 por 3 puede contribuir aproximadamente nueve valores por posición de salida a una matriz de ventanas, ignorando bordes para una estimación. El intermedio puede acercarse a nueve veces el número de posiciones originales. La cifra exacta depende de padding, stride, canales y tamaño de salida.
:::

## Gather y embedding

Una tabla de embeddings tiene una fila de características por identificador de token. Gather selecciona las filas indicadas por una lista de enteros. A diferencia de una suma elemento a elemento, las direcciones dependen de datos de entrada, no solo de los índices de salida.

Lumbre implementa gather para una tabla FP32 bidimensional y un tensor de índices int32. El runtime de entrenamiento genera identificadores válidos a partir de un vocabulario cerrado. El kernel evita una lectura fuera de rango, pero la validez semántica de etiquetas e identificadores también debe comprobarse en la canalización de datos.

En backward, dos tokens iguales contribuyen a la misma fila de gradiente. Se necesita una suma de contribuciones, un scatter-add conceptual. La versión de referencia utiliza un recorrido determinista sencillo; una versión GPU rápida necesitará una estrategia de atomics, agrupación o reducción segmentada.

## Formas dinámicas y guardias

Si el número de tokens cambia, un compilador estático puede generar una variante por longitud. Un compilador con formas simbólicas puede aceptar varias longitudes bajo guardias de rango y relaciones entre dimensiones. Otra estrategia agrupa longitudes en buckets y rellena hasta un tamaño conocido.

Cada estrategia tiene costes: compilación, memoria desperdiciada, complejidad de índices o diversidad de variantes. Una API que acepta una longitud distinta no es necesariamente dinámica en todos sus niveles; puede estar recompilando silenciosamente.

## Control de flujo

Una condición que depende de una configuración conocida puede resolverse al construir el grafo. Una condición que depende de datos exige representar selección, ramas o control de flujo en ejecución. `where` calcula una selección elemento a elemento, pero no sustituye todos los bucles y ramas de un lenguaje general.

Un bucle cuyo número de iteraciones depende de una búsqueda, una estructura de datos irregular o un criterio de convergencia necesita soporte adicional. La universalidad de Python como lenguaje anfitrión no transfiere automáticamente esa capacidad al grafo compilado.

## Un manifiesto de operaciones soportadas

Un proyecto serio mantiene un catálogo con operación, tipos, formas, backward, backends y tests. «Soporta atención» es demasiado impreciso si solo funciona para una longitud, un layout, una máscara y forward sin gradiente.

En Lumbre se distinguen las operaciones generales del IR, el laboratorio de vistas, el kernel de atención online y los backends de dispositivo. C y CUDA cuentan con pruebas ejecutadas; HIP sigue pendiente de hardware compatible. El catálogo de la entrega permite saber qué constituye una implementación ejecutada y qué constituye una práctica avanzada.

::: practica Descomponer una capa sin inventar una instrucción
Describe una capa que normaliza cada fila por la raíz de la media de sus cuadrados y luego aplica una proyección lineal. Enumera qué operaciones primitivas necesita y dónde aparecen reducciones.
:::

**Solución.** Se multiplica la fila por sí misma, se calcula la media sobre características, se añade epsilon, se toma la raíz, se divide la fila por esa estadística y se aplica GEMM. Puede añadirse una escala aprendida por broadcasting. La media es una reducción; la proyección contiene otra reducción sobre su dimensión K. Las demás operaciones son locales o movimientos de forma.

::: comprueba Antes de continuar
Debes poder descomponer una capa conocida y declarar una limitación concreta del compilador. Un mensaje «no soportado» bien definido es preferible a una promesa ambigua de compilar cualquier modelo.
:::
