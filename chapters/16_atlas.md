# Leer tinygrad sin perderse: del tensor al programa {#ch:atlas-tinygrad}

::: idea No abras todos los archivos a la vez
Un repositorio grande se entiende siguiendo una pregunta pequeña. La nuestra será: «¿qué decisiones faltan entre escribir una multiplicación de matrices y enviar instrucciones a un dispositivo?». Cada archivo que abramos debe contestar una parte. Leer miles de líneas sin una pregunta no equivale a comprender la arquitectura.
:::

## George Hotz y la idea de estudiar una pila completa

George Hotz, conocido como geohot, describe en su retrospectiva de diciembre de 2025 el desarrollo de tinygrad desde su primer commit de octubre de 2020. Ese texto es una fuente histórica sobre las motivaciones del proyecto; sus cifras de equipo y tamaño corresponden a aquella fecha, no constituyen un censo de octubre de 2026. El sitio de Tiny Corp presenta tinygrad como una pila de cálculo que conecta operaciones tensoriales, compilación y dispositivos. [@geohot;@tinygrad-site]

La lección que adoptamos no es «cuantas menos líneas, mejor». Es **hacer visible la distancia entre la intención matemática y la ejecución**. Un sistema corto puede tener conceptos difíciles; uno largo puede estar bien dividido. Para aprender, interesa que el estudiante pueda señalar dónde se decide una forma, dónde se produce un bucle y dónde se reserva una región de memoria. Lumbre organiza esas preguntas en módulos pequeños, pero no pretende reproducir todos los mecanismos de tinygrad.

Hay tres formas útiles de leer un proyecto. La lectura de usuario pregunta cómo obtener un resultado. La lectura de compilador pregunta qué representaciones atraviesa. La lectura de sistemas pregunta cómo ese resultado llega al dispositivo y cómo se detecta que ha terminado. En este curso se usan las tres; confundirlas provoca frases como «el compilador es el controlador» o «el kernel de la GPU es Linux».

## Qué versión se está leyendo

El snapshot de tinygrad fijado para este libro es el commit `c3aec477b99d9bb87c54d91897cf60acd3f17441`, de 30 de septiembre de 2026. Se ha inspeccionado su pipeline de generación, además de documentación y páginas del proyecto. No se afirma haber ejecutado su batería completa ni auditado cada archivo del repositorio. Una ruta observada en una importación demuestra una dependencia del pipeline; no sustituye la lectura de la implementación importada. [@tinygrad-pin;@tinygrad-codegen]

Un procedimiento de lectura reproducible conserva el identificador del commit, el archivo y la pregunta respondida. Por ejemplo: «en este snapshot, ¿dónde se representa el tipo de eje?». La respuesta se conecta con `AxisType` y con los usos observados en `codegen/__init__.py`. Si una edición posterior reorganiza las carpetas, la pregunta sigue siendo válida aunque la ruta cambie.

::: ejemplo Una ficha de lectura mejor que una captura
**Pregunta:** ¿cómo se convierte una reducción en estado de acumulación?

**Evidencia inspeccionada:** en el pipeline fijado aparece `reduce_ranges_to_acc`, que crea almacenamiento de registro, inicializa con el elemento identidad y construye actualizaciones y finalización de rangos.

**Interpretación didáctica:** una suma declarativa necesita un valor inicial, un orden de contribuciones y un punto donde el resultado esté disponible.

**No demostrado por esa lectura:** qué ensamblador exacto se obtiene para cualquier GPU, ni cuánto tarda.
:::

## Cuatro traducciones de una misma intención

Volvamos a $C=AB$. En la entrada, es una relación entre tensores. En una representación por índices, es una suma sobre $k$. En una organización por tiles, es un reparto de submatrices. En una instrucción matricial, ciertos fragmentos deben estar distribuidos entre hilos y registros como exige la máquina.

```diagram
Relación tensorial: C = AB | Índices, rangos y memoria | Instrucciones y lanzamiento
```

No todas las etapas tienen que usar clases de objetos diferentes. Una IR puede conservar un tipo de nodo general y cambiar los operadores que admite en cada fase. Lo importante es el **contrato de fase**: después de bajar una operación no debe reaparecer una forma que el siguiente paso no entiende. Por ejemplo, un renderer de C elemental no puede recibir una reducción abstracta sin una regla que la traduzca a bucles.

En el archivo fijado de tinygrad se observan familias de patrones para movimientos, operaciones, tipos, dimensiones GPU, coalescencia, linealización y asignación de registros. También se ven transformaciones de ejes `UNROLL` y `UPCAST`, así como tratamiento de `WMMA`. Esto muestra que «generar código» no es una sola interpolación de cadenas. [@tinygrad-codegen]

Lumbre conserva parte de esa separación conceptual, pero su renderer es mucho más directo. La elección evita presentar al principiante un gran sistema de patrones antes de que comprenda un índice. Cuando Lumbre usa un bucle C explícito, el compilador de C todavía decide numerosas instrucciones. Por tanto, «nuestro compilador» no significa «hemos escrito también GCC, el ensamblador y el enlazador».

## Cómo seguir una reducción agrupada

Supón que una fila tiene 1.024 valores y participan 256 hilos. Una organización posible entrega cuatro valores a cada hilo. Cada hilo obtiene una suma parcial; después se combinan esas 256 contribuciones. La primera reducción vive en el trabajo privado de cada hilo. La segunda necesita comunicación y una disciplina de sincronización.

En una representación abstracta, ambos pasos podrían seguir siendo reducciones. Al bajar a memoria, algunas contribuciones se materializan en almacenamiento local al grupo. Al bajar a control, se introducen condiciones y barreras. Es peligroso intentar deducir una barrera mirando solo el nombre `SUM`: hay que conocer **quién produce, quién consume y qué memoria comparten**.

Para leer el pipeline, anota junto a cada transformación qué información añade. «Añade un índice» es diferente de «añade un buffer». «Añade una dependencia de orden» es diferente de «reordena un cálculo puro». Esa anotación permite detectar una optimización sospechosa: si desaparece una dependencia sin una justificación de independencia, el resultado puede ser una carrera aunque la fórmula aritmética siga pareciendo correcta.

## Qué copiar como idea y qué no copiar a ciegas

Es razonable aprender del uso de nodos inmutables, verificadores por fase, reglas locales y trazas de compilación. No es razonable copiar una transformación específica de una arquitectura sin conservar sus precondiciones. Una instrucción matricial puede necesitar alineamiento, tipos concretos y participación colectiva. Una regla que los omita no es una versión más sencilla: es una regla incorrecta fuera de un dominio accidentalmente favorable.

Tampoco se deben medir dos sistemas con contratos distintos. Si uno permite aproximaciones y el otro conserva un orden de suma, una diferencia de velocidad no se explica únicamente por la habilidad del compilador. La comparación debe declarar ambos contratos y, cuando proceda, ofrecer dos tablas: equivalencia estricta y modo de rendimiento con tolerancias explícitas.

::: comprueba Antes de cerrar el repositorio
Debes poder dibujar el recorrido de una operación, señalar dos lugares donde cambia su representación y explicar qué se verifica después. No necesitas memorizar los nombres de todas las funciones. Sí necesitas distinguir el dato del programa que lo calcula.
:::

# Tiny Corp más allá del compilador: sistema, firmware y colas {#ch:tinycorp-sistemas}

## Un mapa sin mezclar responsabilidades

La organización pública de Tiny Corp contiene proyectos que no son bibliotecas de redes neuronales. `tinyos` se describe como un **constructor de imágenes de sistema para tinybox**, basado en `ubuntu-image`; no es, por ese nombre, un kernel de sistema operativo escrito desde cero. Su README también documenta etapas de actualización del sistema. [@tinyos]

`custom_mec_public` se presenta como una reimplementación del firmware MEC de GPUs AMD gfx1100, con firmware, emulador y pruebas. El README explica su posición entre paquetes de comandos y lanzamiento de trabajo, y atribuye a sus autores pruebas en hardware de agosto de 2026. Esas pruebas externas no se han repetido durante la elaboración del libro. [@tiny-mec]

Esta diversidad enseña una idea importante: acelerar una multiplicación no basta si el trabajo se envía mal, la imagen de sistema es irreproducible o la cola queda bloqueada. Pero no se arregla un fallo de GEMM reinstalando firmware al azar. Cada capa debe producir una evidencia que permita localizar el problema antes de modificar otra.

| Capa | Objeto principal | Pregunta de diagnóstico |
|---|---|---|
| Modelo | Tensores, pérdida, parámetros | ¿La operación pedida es la correcta? |
| Compilador | IR, bucles, layouts | ¿El programa preserva esa operación? |
| Runtime | Buffers, módulos, lanzamientos | ¿Reciben las funciones los argumentos correctos? |
| Controlador | Memoria y envío al dispositivo | ¿Se autoriza y completa el trabajo? |
| Firmware | Protocolos del procesador de comandos | ¿Se interpretan correctamente los paquetes? |
| Imagen de sistema | Servicios y versiones desplegadas | ¿Se reproduce el entorno previsto? |

## Una cola explicada con pedidos de cocina

Imagina una cinta con pedidos numerados. El cocinero no recibe una llamada nueva por cada ingrediente: lee un pedido completo, usa su configuración y avisa al terminar. Una cola de comandos cumple una función parecida. El host escribe descriptores; un mecanismo de notificación anuncia que hay trabajo; el dispositivo lo consume y publica una señal de finalización.

El modelo siguiente es **una simulación didáctica**, no una especificación de PM4, CUDA ni HSA. Sirve para razonar sobre orden y propiedad de los datos.

```python
from dataclasses import dataclass
from collections import deque

@dataclass(frozen=True)
class Job:
    sequence: int
    operation: str
    inputs: tuple[str, ...]
    output: str

queue = deque()
completed = 0
queue.append(Job(1, "add", ("A", "B"), "C"))
queue.append(Job(2, "square", ("C",), "D"))

while queue:
    job = queue.popleft()
    # Ejecutar aqui la operacion y publicar su salida.
    completed = job.sequence
```

La variable `completed` no debe cambiar antes de que la salida sea visible. En el modelo secuencial resulta obvio. En una máquina con ejecución asíncrona, cachés y varios motores, «he emitido el comando» y «sus escrituras son observables» dejan de coincidir. Ahí aparecen protocolos de espera, visibilidad y finalización.

## Tres errores que parecen uno

**Error de cálculo:** `C` contiene números equivocados aun después de una finalización válida. Investiga índices, tipos y fórmula.

**Error de vida de memoria:** el buffer de `A` se reutiliza antes de que el dispositivo lo lea. El kernel puede ser matemáticamente correcto y recibir datos distintos a los esperados.

**Error de observación:** el host lee `C` antes de que el trabajo termine. Puede ver ceros, valores anteriores o una mezcla. Añadir un `sleep` cambia la probabilidad, pero no establece una dependencia correcta.

Una depuración cuidadosa conserva el mismo programa y modifica una sola hipótesis. Por ejemplo, esperar explícitamente la finalización puede separar un fallo de observación de uno aritmético. Esa prueba no convierte la espera global en la mejor solución de producción: más adelante se sustituye por dependencias precisas para conservar concurrencia.

## Estado, comandos y eventos no son intercambiables

Un comando de configuración puede modificar cómo se interpretan los lanzamientos siguientes. Un comando de lanzamiento inicia trabajo. Un evento o señal permite observar un punto del flujo. Si un optimizador reordena los tres como si fueran sumas puras, puede cambiar el significado completo de la secuencia.

En una IR de efectos conviene representar el estado que una operación consume y el estado que produce. No tiene por qué materializarse como un número físico; puede ser una dependencia de compilación. Así, dos operaciones aritméticas independientes se reordenan, pero una publicación de finalización queda detrás de las escrituras que certifica.

```tikz
\begin{center}
\begin{tikzpicture}[node distance=1.0cm]
\node[box,text width=10cm] (a) {Reservar y escribir los argumentos};
\node[box,text width=10cm,below=of a] (b) {Publicar el descriptor y notificar trabajo};
\node[box,text width=10cm,below=of b] (c) {Ejecutar y hacer visibles las escrituras};
\node[box,text width=10cm,below=of c] (d) {Publicar finalización; después reutilizar memoria};
\draw[flow] (a)--(b);\draw[flow] (b)--(c);\draw[flow] (c)--(d);
\end{tikzpicture}
\end{center}
```

## Un proyecto de firmware puede estudiarse sin instalarlo

El primer trabajo del estudiante es leer la documentación, identificar entradas y salidas del emulador y reproducir un modelo de estados con datos ficticios. No necesita escribir registros físicos, reemplazar módulos del sistema ni cargar una imagen en su GPU personal. Una máquina de desarrollo con trabajo académico no es un banco de pruebas desechable.

La prueba interesante en un emulador compara efectos observables: posiciones de lectura y escritura de la cola, señales, registros modelados y vida de los trabajos. Aun si dos implementaciones producen el mismo estado final en cien ejemplos, pueden discrepar en intercalaciones no examinadas. El informe debe indicar el alcance del modelo, qué eventos se simplificaron y qué propiedades se comprobaron exhaustivamente.

::: practica Diseña una prueba de cola, con solución
Se publican los trabajos 1 y 2. El segundo lee la salida del primero. Una implementación actualiza `completed=2` al sacar el pedido de la cola, antes de ejecutarlo. ¿Qué observación puede hacer el host?

**Solución:** puede interpretar que ambos trabajos terminaron y reutilizar sus buffers, aunque el segundo siga pendiente. El arreglo no consiste en cambiar el número a 1 por comodidad. La señal debe estar ligada al final real de las escrituras que representa. Si hay ejecución fuera de orden, una única frontera consecutiva exige además distinguir trabajos terminados de trabajos anteriores todavía pendientes.
:::

# LLVM y MLIR: comprender SSA antes de aprender sus siglas {#ch:ssa-mlir}

## Dar un nombre nuevo a cada resultado

En un programa corriente, `x` puede cambiar varias veces. Para analizarlo, resulta útil dar un nombre distinto a cada definición. Esta idea se denomina **asignación única estática**, SSA. «Estática» se refiere a los nombres en la descripción del programa; no significa que un bucle solo se ejecute una vez. LLVM describe instrucciones y valores siguiendo esta organización. MLIR ofrece infraestructura para representar y transformar programas en distintos niveles. [@llvm-langref;@mlir-paper]

```text
Programa de partida:         Nombres separados:
x = entrada                 x0 = entrada
x = x + 1                   x1 = x0 + 1
y = x * x                   y0 = x1 * x1
```

Ahora se ve sin ambigüedad que las dos entradas del producto proceden de la suma. En un DAG puro de Lumbre, la referencia al nodo cumple una función similar: no se pregunta «¿qué valor tiene actualmente la variable?», sino «¿qué resultado produjo este nodo?». La dificultad adicional aparece cuando hay bifurcaciones, bucles y memoria mutable.

## Una bifurcación necesita reunir resultados

Considera una función que devuelve el valor absoluto. Una rama conserva el número; la otra cambia su signo. Al reunirse las ramas, hay que indicar qué resultado procede del camino tomado.

```text
entrada(x):
    si x >= 0: ir a positivo
    si no:     ir a negativo
positivo:
    p = x
    ir a union
negativo:
    n = -x
    ir a union
union:
    resultado = elegir_segun_predecesor(p, n)
```

La selección no calcula arbitrariamente ambos valores y escoge uno por su contenido. Asocia cada entrada con una arista de control. LLVM puede representarla mediante `phi`; en otras IR se usan argumentos de bloque. No se deben traducir todos esos mecanismos como una llamada ordinaria a una función de dos argumentos: su semántica depende del predecesor. [@llvm-langref;@mlir-toy]

**Dominancia** significa que para llegar a un punto hay que haber pasado por otro. Si una definición domina un uso, ese uso no puede alcanzarse sin que la definición esté disponible. El nodo de suma del ejemplo lineal domina el producto. En la bifurcación, la definición `p` no domina por sí sola todos los caminos de la unión; de ahí la necesidad de la selección asociada al control.

## Un bucle es una recurrencia, no un círculo arbitrario

Para sumar cuatro valores, la versión matemática dice «acumular contribuciones». Una IR con control necesita el valor inicial, la condición de continuación y el valor que vuelve por la arista del bucle.

```text
inicio:
    ir a cabecera(i=0, acumulado=0)
cabecera(i, acumulado):
    si i == N: ir a salida(acumulado)
    valor = A[i]
    siguiente = acumulado + valor
    ir a cabecera(i+1, siguiente)
salida(resultado):
    devolver resultado
```

La pareja de argumentos de la cabecera cambia en cada iteración, aunque sus nombres estáticos sean los mismos. El ciclo tiene una estructura controlada y una condición. No equivale a insertar un nodo que se referencia a sí mismo en un DAG de expresiones puras, donde la evaluación topológica dejaría de estar definida.

::: ejemplo Invariante completo del bucle
Antes de cada visita a la cabecera, `acumulado` es la suma de los elementos con índices menores que `i`, e `i` está entre 0 y `N`. Al principio se han sumado cero elementos. Una iteración añade exactamente `A[i]` y aumenta `i` en uno. Al terminar, `i=N`, por lo que se han incorporado todos los elementos y ninguno dos veces.

Para coma flotante, este invariante debe interpretarse como la secuencia de sumas redondeadas del programa. No autoriza automáticamente a sustituirla por cualquier árbol de reducción.
:::

## Dialectos: conservar información mientras sea útil

Una representación de alto nivel puede saber que una operación es una convolución. Después puede conocer bucles e índices afines. Más tarde, instrucciones vectoriales y accesos a memoria. Si se baja demasiado pronto a instrucciones escalares, reconocer otra vez la convolución resulta difícil. Si se conserva demasiado tiempo una abstracción que oculta la arquitectura, no se pueden tomar decisiones de registros o sincronización.

El diseño de MLIR permite dialectos y conversiones entre representaciones. El tutorial Toy desarrolla un lenguaje pequeño para enseñar ese recorrido. El objetivo de la lectura no es copiar todo MLIR en Lumbre, sino aprender que una transformación debería declarar qué información exige y qué información elimina. [@mlir-toy;@mlir-paper]

Nuestro contrato de conversión imaginario será: «antes hay operaciones tensoriales con formas conocidas; después hay bucles y buffers; ninguna operación tensorial no legalizada puede quedar». El verificador revisa ese contrato. Si aparece una operación desconocida, debe producir un diagnóstico, no transformarla en un comentario de C ni ignorarla.

## Alias: dos nombres pueden señalar la misma caja

SSA da nombres distintos a resultados, pero no elimina por magia el alias de memoria. Dos punteros pueden señalar la misma región. En ese caso, escribir mediante uno cambia lo que se lee mediante el otro. Reordenar un `load` alrededor de un `store` necesita demostrar que no interfieren, o representar la dependencia.

Por ejemplo, una función recibe `A` y `B`; calcula `t=A[0]`, escribe `B[0]=7` y vuelve a leer `A[0]`. Si los argumentos son alias, las dos lecturas pueden dar valores distintos. Una optimización que reutilice `t` como si la memoria fuera inmutable sería incorrecta. Este es uno de los motivos por los que el runtime educativo de Lumbre mantiene reglas simples de propiedad y actualización de parámetros.

**Análisis** y **transformación** deben distinguirse. El análisis obtiene una propiedad, como «estas dos regiones son disjuntas». La transformación utiliza esa propiedad, por ejemplo para adelantar una carga. Si la propiedad no se ha demostrado, la transformación no puede fingir que existe. En una primera implementación, ser conservador es preferible a aceptar resultados silenciosamente corruptos.

## Diseñar diagnósticos para alguien que empieza

«IR inválida» es un mensaje pobre. «La suma esperaba operandos compatibles, pero recibió formas `(2,3)` y `(4,3)`; el eje exterior 2 no coincide con 4» permite corregir el programa. Lo mismo ocurre con un error de dominancia: señala el uso, la definición y el camino por el que se alcanza el primero sin pasar por la segunda.

En un compilador educativo, cada fase puede devolver además una representación textual pequeña. El estudiante compara antes y después y busca la primera divergencia, igual que en una ejecución de un autómata. Una traza de compilación no es ruido de depuración: es la explicación concreta de cómo una promesa de alto nivel se convirtió en un programa ejecutable.

# Triton, CuTe, TileLang y bibliotecas: elegir una abstracción {#ch:atlas-lenguajes}

## La pregunta no es quién tiene la sintaxis más corta

Tres programas pueden calcular la misma matriz y repartir el trabajo de maneras distintas. Uno expresa el resultado tensorial completo. Otro describe un tile y deja al compilador distribuirlo. Otro determina fragmentos y layouts cercanos al hardware. Comparar solo el número de líneas oculta qué decisiones ha tomado el programador y cuáles ha delegado.

Triton documenta programas de multiplicación matricial organizados por bloques; CUTLASS y CuTe ofrecen herramientas de composición y layouts; TileLang desarrolla un modelo de programación por tiles; ThunderKittens se centra en primitivas para construir kernels. Estas descripciones no implican que todos produzcan el mismo código ni soporten el mismo hardware. [@triton-tutorial;@cutlass;@tilelang-paper;@thunderkittens]

## Un contrato común para comparar soluciones

Definamos la tarea antes del lenguaje: $A$ tiene forma $(M,K)$, $B$ forma $(K,N)$, ambas contiguas por filas; acumulamos en FP32 y devolvemos $C$ de forma $(M,N)$. Permitimos tamaños no múltiplos de tile, por lo que hay que proteger cargas y escrituras. La primera versión no admite alias entre entradas y salida.

Con ese contrato, una implementación debe contestar las mismas preguntas: cómo elige su tile de salida, cómo recorre $K$, qué sucede en los bordes, dónde conserva acumuladores y cuándo puede escribir. El lenguaje cambia la forma de expresar esas decisiones, no hace que desaparezcan.

El siguiente esquema **no es una API ejecutable de ningún proyecto**. Es una ficha de correspondencia para leer implementaciones reales sin confundir sus nombres.

```text
programa(tile_fila, tile_columna):
    filas = tile_fila * BM + [0, ..., BM-1]
    columnas = tile_columna * BN + [0, ..., BN-1]
    acumulador = ceros(BM, BN)
    para inicio_k en 0, BK, 2*BK, ...:
        A_tile = cargar_con_mascara(A, filas, inicio_k)
        B_tile = cargar_con_mascara(B, inicio_k, columnas)
        acumulador += producto(A_tile, B_tile)
    guardar_con_mascara(C, filas, columnas, acumulador)
```

## Triton: un programa procesa un bloque de datos

Al leer el tutorial oficial de GEMM de Triton, identifica primero los identificadores de programa y después los vectores de índices. Una expresión vectorial de índices no significa que haya un hilo de Python ejecutando cada elemento. Describe trabajo que el compilador bajará al dispositivo. Los parámetros de bloque afectan reutilización, ocupación y cantidad de programas. [@triton-tutorial]

El error inicial más frecuente es interpretar el código como NumPy ejecutándose en la GPU. La sintaxis familiar no elimina el carácter compilado ni los requisitos de los punteros y máscaras. Otro error es copiar un tamaño de bloque óptimo para una forma grande y utilizarlo en matrices pequeñas: el lanzamiento puede crear demasiado trabajo inútil o demasiado pocos programas para ocupar la máquina.

Para estudiar un kernel, cambia solo una dimensión. Mantén $M$ y $N$, aumenta $K$ y observa qué bucle se alarga. Después deja $K$ fijo y aumenta $M$: debería crecer el número de tiles de salida. Si ambas modificaciones cambian de la misma manera todos los recursos, revisa si has entendido qué dimensión se mapea al grid y cuál al bucle interno.

## CuTe y CUTLASS: el layout también es un programa

CUTLASS no es solo una colección de funciones GEMM finales. Su organización permite componer componentes, y CuTe hace explícitas relaciones entre coordenadas y almacenamiento. La versión del README consultada incluye CuTe DSL en Python y secciones específicas de arquitecturas recientes. Eso no convierte las capacidades de una GPU de centro de datos en capacidades de cualquier tarjeta que comparta el nombre de familia. [@cutlass]

Para comprender un layout, empieza con una tabla de cuatro elementos. El layout lineal `[0,1,2,3]` y el layout `[0,2,1,3]` contienen los mismos valores, pero asignan coordenadas a posiciones distintas. Al componer un layout lógico con un reparto entre hilos, una coordenada termina en un par «hilo, registro». Una instrucción colectiva exige que ese reparto coincida con su contrato. No basta con que cada hilo posea el número correcto de elementos.

La pregunta útil es: «para la coordenada $(i,j)$, ¿quién la posee y en qué posición local?». Cuando puedes responderla para una matriz diminuta, un tipo de layout deja de parecer una fórmula ornamental. En hardware moderno, ciertos layouts además expresan descriptores, memoria compartida y operandos de instrucciones matriciales; esos detalles deben consultarse en la documentación de la arquitectura concreta. [@ptx;@cutlass]

## TileLang: separar la intención del tile y su planificación

El trabajo de TileLang propone un modelo composable por tiles. Su interés didáctico está en poder razonar sobre regiones de datos y organización del cómputo sin escribir primero todas las instrucciones escalares. Las APIs evolucionan: la documentación consultada distingue interfaces dinámicas actuales y nombres usados por ejemplos antiguos. Un ejemplo encontrado en una publicación debe ejecutarse con las versiones que declara, no mezclarse con la última instalación suponiendo compatibilidad. [@tilelang-paper;@tilelang-doc]

Un port correcto conserva la semántica de bordes y reducción. Si una versión GPU usaba relleno cero para el borde de $K$, una traducción que lea memoria no inicializada puede fallar únicamente en tamaños impares. Si el lenguaje ofrece primitivas de pipeline, el estudiante todavía debe comprender las vidas de los buffers: reutilizar una etapa antes de que termine su consumidor no se vuelve seguro por estar escrito con una abstracción de alto nivel.

## Bibliotecas de kernels frente a compiladores completos

DeepGEMM aporta implementaciones especializadas de multiplicación, incluidas variantes útiles para modelos con expertos. DeepEP se ocupa de comunicación para expertos. DeepJIT aborda piezas de compilación, carga y lanzamiento. No son nombres diferentes de una misma biblioteca: pertenecen a problemas complementarios. Los ports Ascend deben leerse con sus restricciones, versiones y estados de implementación. [@deepgemm;@deepep;@deepjit;@deepgemm-ascend;@deepep-ascend]

Un compilador puede decidir llamar a una biblioteca en lugar de generar toda la operación. Eso no es hacer trampa si la llamada se declara. Debe controlar el contrato de tipos, layouts, workspace, stream y propiedad de memoria. Para evaluar el compilador, conviene informar por separado qué operaciones se generan y cuáles se delegan. Una mejora atribuida al renderer podría proceder, en realidad, de una nueva versión de la biblioteca llamada.

En AMD, los avisos de migración de Composable Kernel y hipBLASLt remiten a `ROCm/rocm-libraries`; RCCL remite a `ROCm/rocm-systems`. El repositorio histórico sigue siendo útil para entender referencias antiguas, pero no se debe construir una guía de contribución actual ignorando esos avisos. [@ck;@hipblaslt;@rccl]

## Decisión de proyecto: tres rutas legítimas

**Compilador desde cero:** implementa el subconjunto semántico y genera C/CUDA/HIP propio. Aprendes cada frontera, pero el rendimiento inicial será modesto.

**Frontend propio con backend existente:** construye una IR didáctica y bájala a MLIR, Triton o TileLang. Aprendes integración y contratos de conversión; debes reconocer qué optimizaciones realiza el backend externo.

**Especialización de un kernel:** conserva el modelo y sustituye una operación por una implementación propia. El alcance es menor, pero permite una evaluación rigurosa y una contribución útil. Un kernel bien estudiado suele ser un proyecto más defendible que una promesa de reemplazar toda la pila sin pruebas.

# Leer artículos de 2026 como investigador, no como espectador {#ch:articulos-2026}

## Una afirmación necesita su dominio

Cuando un artículo dice que una solución es más rápida, pregunta: ¿para qué operaciones, formas, tipos, arquitecturas y presupuesto de ajuste? Una mejora sobre un conjunto no se transforma en una garantía para cualquier programa. Tampoco se puede comparar un candidato afinado durante días con una referencia tomada sin sus opciones habituales y llamarlo una comparación neutral.

El artículo **AI as a Compiler**, presentado el 29 de septiembre de 2026, estudia traducción mediante agentes de kernels Triton a PTX. Sus autores reportan resultados que incluyen tanto casos inferiores a la referencia como aceleraciones, y apoyan el proceso con un entorno de evaluación y extensiones de un verificador de PTX para capacidades recientes. Es evidencia sobre un método y un conjunto de experimentos, no una demostración de que cualquier LLM sustituya correctamente cualquier compilador. [@taic]

Nuestra interpretación para el curso es separar **proponer** de **aceptar**. Un generador puede ser creativo y producir candidatos poco obvios. La aceptación exige un contrato, pruebas, restricciones de recursos y, donde sea posible, verificación de propiedades. Cambiar el generador no elimina la obligación de explicar qué significa correcto.

## Tres preguntas complementarias de la literatura reciente

**TileSight** aborda modelado analítico del rendimiento centrado en tiles. **Correct but Slow** estudia la distancia entre corrección y rendimiento en kernels de lenguajes específicos de dominio. **Characterizing Real-World Bugs in Tile Programs** examina fallos de programas por tiles para orientar su detección. Estas lecturas se incluyen como líneas de investigación de 2026; no se convierten aquí en una clasificación exhaustiva ni en benchmarks repetidos por el autor del libro. [@tilesight;@correct-slow;@tile-bugs]

Podemos conectar esas tres preguntas con Lumbre sin copiar sus experimentos. Primera: ¿predice un modelo de bytes y operaciones qué versión conviene? Segunda: ¿cuántos candidatos pasan la comparación numérica, pero siguen siendo lentos? Tercera: ¿qué clases de errores detectan nuestras pruebas y cuáles podrían quedar ocultas? Son preguntas distintas. Tener una respuesta satisfactoria para la primera no resuelve las otras dos.

## Diseñar una tabla que no oculte fracasos

Supón que se proponen diez kernels. Seis no compilan, uno produce resultados incorrectos, uno excede memoria y dos son correctos. Si se publica únicamente el más rápido de esos dos, falta información sobre el coste y fiabilidad del procedimiento. Una tabla útil conserva el estado de cada intento.

| Estado | Qué se ha observado | Qué no se puede concluir |
|---|---|---|
| No compila | El toolchain rechaza el candidato | Que la idea matemática sea imposible |
| Compila, falla prueba | Un caso contradice el contrato | Que otros casos compensen el error |
| Correcto en la batería | Pasa las comprobaciones realizadas | Corrección universal sin más argumento |
| Tiempo excedido | No termina dentro del presupuesto | Su tiempo exacto ni su corrección |
| Más rápido | Mejora la métrica del ensayo | Mejora de todo el modelo o toda GPU |

Esta tabla permite calcular tasas de éxito y costes de búsqueda, además de velocidad final. El presupuesto debe incluir compilaciones, pruebas fallidas y ajuste. Si el objetivo es amortizar una búsqueda durante millones de ejecuciones, el análisis puede favorecer un método caro; si la forma cambia cada segundo, quizá no. La conclusión depende de la carga de uso.

## Separar prueba formal, comprobación diferencial y muestreo

Una **prueba formal** deriva una propiedad bajo un modelo y unas hipótesis. Si el modelo no incluye una característica del hardware, la garantía no puede extenderse silenciosamente a esa característica. Una **prueba diferencial** compara dos implementaciones para entradas concretas. Un **muestreo aleatorio** explora entradas, pero puede pasar por alto fronteras raras.

Las tres herramientas se complementan. Para una transposición, podemos demostrar la biyección de índices y además comparar resultados. Para una reducción, podemos justificar que cada elemento contribuye una vez, comprobar tolerancias y medir la sensibilidad al orden. Para un pipeline GPU, necesitamos razonar sobre estados y sincronización; comparar únicamente una salida en una ejecución puede no revelar una carrera intermitente.

::: ejemplo Un resultado negativo que sí enseña
En las mediciones locales de este libro, la atención online evita materializar una matriz de puntuaciones, pero resulta más lenta que la alternativa densa bloqueada en dos tamaños examinados. No invalida FlashAttention. Muestra que una versión escalar por fila, una CPU y tamaños pequeños tienen un equilibrio diferente al de un kernel GPU optimizado por tiles.

La conclusión defendible es sobre esas implementaciones y condiciones. La siguiente hipótesis será reducir overhead, vectorizar o cambiar el reparto; no «el artículo es falso» ni «la reducción de memoria garantiza velocidad».
:::

## Elaborar una réplica de alcance honesto

Una réplica puede ser semántica, arquitectónica o de rendimiento. La réplica semántica implementa la misma operación o recurrencia y comprueba sus resultados. La arquitectónica reproduce organización relevante del cómputo, como tiles y pipeline. La de rendimiento intenta repetir condiciones y métricas publicadas. No deben usarse las tres etiquetas como sinónimos.

Lumbre implementa una recurrencia online de atención y prueba su equivalencia numérica con la versión densa en un dominio finito. Su backward recompone operaciones densas. Por eso no es una réplica integral del entrenamiento FlashAttention con memoria intermedia lineal. La etiqueta precisa permite saber qué está aprendido y cuál es el siguiente trabajo real.

Para tu informe, termina cada lectura con cuatro frases: qué problema define; qué mecanismo propone; qué evidencia presenta; qué experimento propio podría refutar tu interpretación. La última frase obliga a transformar admiración en una hipótesis comprobable.
