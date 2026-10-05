#* Programa del semestre: Compilers for Machine Learning {#curso:programa}

## El encargo y la forma de trabajar

Esta ampliación, incorporada el **5 de octubre de 2026**, desarrolla el syllabus completo facilitado por el lector: un semestre práctico en el que cada estudiante construye su compilador desde programas elemento a elemento hasta el entrenamiento de modelos de lenguaje en GPU. La relación entre representación intermedia, hardware y estructura del modelo guía las decisiones. Se estudian reescritura de términos, generación de código, movimientos, fusión de kernels, jerarquías de memoria, arquitectura GPU, autodiferenciación y FlashAttention.

El proyecto se realiza individualmente o en equipos de dos. El lenguaje de implementación es libre. El syllabus establece la especificación y propone un formato sencillo de intercambio de UOps; no entrega código inicial ni impone cómo implementar el compilador. Lumbre y los tutoriales de este repositorio son material de consulta y contraste. Para seguir la modalidad de construcción desde cero, crea tu propio proyecto y resuelve cada entrega antes de consultar su solución en Lumbre.

Aunque no hay prerrequisitos formales, el curso original presupone dominio de sistemas equivalente a 15-213, soltura en C, cierta experiencia GPU, dominio del lenguaje elegido y capacidad para mantener un proyecto durante meses. El itinerario preparatorio del libro sirve para detectar y cubrir lagunas; no sustituye automáticamente esa experiencia. La modalidad presencial del syllabus incluye cuestionarios frecuentes y clases que no se publican en línea. Estas condiciones describen el curso de referencia; el repositorio no anuncia sesiones presenciales propias.

La entrega semanal se realiza **el miércoles antes de las 10:00, antes de clase**. El orden de las sesiones es miércoles, viernes y lunes. No se proporciona zona horaria ni fechas del calendario académico: deben fijarse al organizar una edición concreta. Cada entrega conserva las capacidades de las anteriores. Una regla de simplificación incorrecta en la primera semana puede invalidar gradientes y modelos meses después.

::: idea Gestionar el trabajo que todavía no se entiende
La advertencia del syllabus sobre el «slop» se convierte aquí en una práctica observable: toda transformación debe tener un contrato, un contraejemplo que podría hacerla fallar y una prueba de regresión. Poder generar mucho código no sustituye poder explicar qué hace. Un integrante debe poder defender cambios escritos por el otro y cualquier fragmento producido con ayuda automática.
:::

```tikz
\needspace{12\baselineskip}
```

## El semestre de un vistazo

| Semanas | Trabajo acumulado | Hito del syllabus |
|---|---|---|
| 1–2 | UOps, operaciones elementales, simbólico, reescrituras, renderer y runtime | Compilar y simplificar programas elemento a elemento a C |
| 3–4 | Bucles, movimientos, broadcasting, reducción y rangeify | Compilar grafos de modelos expresables a C correcto, inicialmente lento |
| 5–6 | Fisión, fusión, autodiff, memoria, hilos, upcasting, GEMM y convolución | Entrenar MNIST y aspirar a código CPU competitivo con el estado del arte |
| 7–8 | Kernels, GPU, mapeo de ejes, aceleradores y tensor cores | Aspirar a código CUDA/HIP competitivo con PyTorch |
| 9–10 | Modelos reales, formas simbólicas y entrenamiento | Implementar y entrenar un LLM |
| 11 en adelante | Proyecto propio sobre el compilador | Implementar un artículo, portar a otro hardware o desarrollar una extensión |

La tabla final del syllabus agrupa autodiff con modelos reales; el desarrollo semanal lo introduce en la **semana 5**. Lo aprenderemos entonces y lo utilizaremos de nuevo en las semanas 9–10. Igualmente, `CALL` aparece en la semana 1 aunque las llamadas a kernels de dispositivo se estudien después. Las jerarquías de memoria y la aceleración CPU culminan en la semana 6; no se pospone autodiff para acomodarlo al resumen.

## Semana 1. UOps y reescrituras

Construye nodos inmutables con interfaz conceptual `UOp(op, src:tuple[UOp, ...], arg)` y hash-consing. Introduce la propiedad recursiva `dtype` con valores `bool`, `i32` y `f32`. El conjunto inicial es `PARAM`, `CALL`, `CONST`, `CAST`, `NEG`, `RECIP`, `EXP2`, `LOG2`, `ADD`, `MUL`, `CMPLT`, `CMPNE`, `MAX`, `FLOORDIV`, `FLOORMOD` y `WHERE`. Para booleanos, `MUL`, `MAX` y `CMPNE` significan AND, OR y XOR.

El miércoles se presentan curso, completitud de Turing, UOp, formato y primera tarea. El viernes se trabajan ALU, reescritura voraz de abajo arriba y plegado de constantes. El lunes se estudian llamadas, parámetros, simplificación simbólica y sistemas de álgebra computacional. El plegador debe poder evaluar programas constantes del subconjunto soportado; mencionar completitud de Turing en clase no convierte por sí solo ese subconjunto en un lenguaje universal.

Entrega un constructor, impresor y lector del subconjunto acordado, un verificador, un evaluador de referencia y un motor de reescrituras. Demuestra que una subexpresión pura repetida se comparte, que los parámetros mantienen su ámbito y que cada regla termina. Lectura: «UOps: construir un grafo de intenciones», simplificación simbólica y laboratorios 1–2. El contrato ampliado al final del libro concreta casos límite.

## Semana 2. Renderer, runtime y memoria

Añade `void`, la propiedad `shape:tuple[int, ...]` y `addrspace` con `MEM` y `ALU`. Introduce `BUFFER`, `ALLOC`, `STACK`, `INDEX`, `LOAD`, `STORE`, `AFTER` y `LINEAR`. `BUFFER` representa memoria persistente global y no se define dentro de una llamada; `ALLOC` reserva almacenamiento local a la llamada. Un buffer externo puede pasarse como argumento de la función sin volver a declararlo como global dentro de su cuerpo.

El miércoles se estudian orden topológico, renderer y repaso de C. El viernes, runtime y gestión de memoria. El lunes, sistemas de tipos para forma, dtype y espacio de direcciones. La entrega genera, compila y ejecuta funciones C **sin bucles** que combinan inmediatos y memoria. No es necesario anticipar `RANGE` para demostrar una carga, una suma y una escritura.

La prueba decisiva llama dos veces a la función con datos distintos y comprueba memoria y valores. Incluye una dependencia de lectura después de escritura y rechazo de índices inválidos. Lectura: «Del grafo al C», planificación de memoria y laboratorio 3. No basta con imprimir C: la firma, los tamaños y la vida de los buffers forman parte de la entrega.

## Semana 3. Bucles, GEMM y convolución

`RANGE` recibe como fuente el límite superior y como argumento el nombre del rango. Un límite de tres recorre 0, 1 y 2. `END` recibe `(passthrough, range)` y cierra el rango correspondiente. El renderer debe preservar anidamiento, acumuladores y valores disponibles al salir.

Escribe una reducción, una multiplicación de matrices rectangular y una convolución pequeña usando índices explícitos. Prueba rangos de longitud cero y uno donde el contrato los permita. Una suma vacía conserva su identidad; una reducción sin identidad definida requiere un diagnóstico. Lectura: bucles, reducción, GEMM y laboratorio de convolución. La semántica de bucles aparece aquí; la transformación automática de formas a bucles llegará en la semana siguiente.

## Semana 4. Movimientos y rangeify

Incorpora `RESHAPE`, `EXPAND`, `PERMUTE`, `FLIP`, `PAD`, `SHRINK` y `REDUCE`, parametrizada por número de ejes y operación. Obtén GEMM y convolución componiendo movimientos y reducciones. El alumno debe poder seguir una coordenada desde la salida hasta cada entrada sin materializar todas las vistas.

Rangeify elimina las formas de las operaciones del grafo de cálculo bajado: el trabajo queda expresado con rangos, índices y operaciones escalares. El runtime sigue necesitando tamaños y metadatos de almacenamiento. La entrega conserva el grafo de alto nivel para contrastar valores y comprueba que el backend recibe su representación escalar esperada. Lectura: movimientos, broadcasting, layouts y laboratorio 3.

## Semana 5. Fisión, fusión y autodiferenciación

`STAGE` representa un buffer temporal indexable. Úsalo para decidir cuándo materializar un valor compartido y cuándo fusionar su cálculo en el consumidor. Contar kernels sin contar cómputo repetido puede premiar el plan equivocado.

Autodiff actúa sobre **grafos puros con formas**, antes de introducir efectos. No se aplica a `STORE`, `CALL`, `AFTER`, `RANGE` ni `END`. Genera el grafo de gradientes y bájalo después con los mismos mecanismos que el forward. El hito es entrenar el primer clasificador de MNIST con tu compilador. Los datos, la pérdida y el protocolo experimental deben documentarse; este hito es una entrega propuesta, no una ejecución MNIST ya incluida entre las evidencias de Lumbre.

Lectura: planificación de kernels, autodiff y laboratorio 8. La guía ampliada desarrolla el backward del broadcasting, un clasificador y las pruebas necesarias para evitar una falsa sensación de éxito basada solo en la pérdida.

## Semana 6. Memoria, multihilo y kernels rápidos en CPU

Clasifica los rangos como `LOOP`, `UPCAST` o `THREAD`. Upcasting transforma trabajo de un eje en varias operaciones explícitas que el backend puede combinar o vectorizar; no significa convertir FP32 a FP64. Añade `GROUP`, que agrupa salidas `void` sin ordenar sus fuentes y tiene forma escalar `()`.

Trabaja localidad, tiling, acumuladores, hilos, conteo de FLOPs, GEMM, reducciones, convoluciones y atención con memoria acotada. La CPU sigue siendo el destino. La meta del syllabus es código C competitivo con el estado del arte: una afirmación así exige suite, referencia fuerte, precisión, hilos, hardware y medidas reproducibles. Lectura: CPU rápida, benchmarks, atención eficiente y proyecto C.

## Semanas 7–8. GPU y aceleradores

Introduce kernels de dispositivo, runtime GPU, mapeo de ejes a bloques e hilos, jerarquía de memoria y tensor cores. Mantén un camino correcto para formas no compatibles con el kernel especializado. CUDA e HIP son destinos del programa; validar uno no valida automáticamente el otro.

La meta es competir con PyTorch en cargas acordadas. Compara operaciones equivalentes y declara si incluyes compilación, transferencias, conversiones y sincronización. Lectura: GPU básica, aceleradores, laboratorio 7 y el nuevo puente del tutorial CUDA. Las prácticas existentes de vectores, reducción, GEMM y streams ofrecen referencias para revisar el código que genere el compilador propio.

## Semanas 9–10. Modelos reales y dimensiones simbólicas

Integra un modelo de lenguaje con embeddings, atención causal, normalización, bloque feed-forward, pérdida, gradientes, optimizador y checkpoints. Lleva las dimensiones simbólicas desde las restricciones del frontend hasta las comprobaciones de ejecución y la clave de caché. Un tamaño variable no puede invalidar silenciosamente un tile o una reserva.

Entrena primero una configuración pequeña que puedas depurar y después escala dentro del presupuesto disponible. Registra tokens, arquitectura, memoria, tiempo por paso y evaluación. La aspiración a LLM de última generación del syllabus requiere además datos, recursos y comparación externa; los resultados existentes del repositorio acreditan un decoder pequeño. Lectura: modelo de lenguaje, entrenamiento, escalado y laboratorios 9–11.

## Semanas 11 en adelante. Proyecto y defensa

Escoge una contribución sobre el compilador: reproducir una idea de un artículo, portar a hardware distinto, implementar backward de atención por tiles, mejorar selección de kernels o ampliar el entrenamiento. Los proyectos A–E del libro dan ejemplos. El syllabus deja abierto el número de semanas finales; cualquier calendario de catorce semanas de este material es una propuesta organizativa.

Entrega código propio, pruebas, instrucciones, resultados, límites y una comparación que permita activar y desactivar la contribución. Para equipos de dos, ambos deben poder explicar un caso nuevo desde el tensor hasta la ejecución. Una defensa útil modifica una forma o una dependencia y pide predecir el efecto antes de ejecutar.

## Qué se conserva entre entregas

Cada miércoles etiqueta una versión reproducible con especificación vigente, gramática admitida, ejemplos de IR, pruebas acumuladas y un informe breve. Al añadir una operación, registra sus propiedades, aridad, precondiciones y fase legal. Al añadir una optimización, conserva un programa mínimo que detecte su fallo más probable. Al añadir un backend, mantén los mismos valores de entrada y el mismo contrato numérico.

Los capítulos «Contrato de UOps» y «De rangeify al entrenamiento» amplían el syllabus con decisiones didácticas explícitas. El formato del mensaje original es un ejemplo y no una gramática completa. No se atribuyen al curso reglas de ABI, literales o evaluación que su texto no fija.
