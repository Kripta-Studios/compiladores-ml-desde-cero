# Tokenización por bytes y BPE: una extensión completa {#ch:bpe}

## El carácter desconocido revela una decisión pendiente

Nuestro vocabulario de caracteres se obtiene del corpus didáctico. ¿Qué ocurre si después aparece un símbolo nuevo? No basta con cambiar su identificador a cero sin explicarlo: eso sustituye información por otro token. Para una demostración cerrada es una simplificación visible; para una aplicación general necesitamos un contrato mejor.

Una opción es empezar por los 256 valores posibles de un byte. Todo texto UTF-8 se representa como una secuencia de esos bytes. El vocabulario básico puede codificar cualquier texto válido sin inventar un token desconocido para cada carácter nuevo.

El precio es que un carácter puede ocupar varios tokens. La longitud en caracteres, bytes y tokens deja de ser la misma. Comparar pérdidas por token entre tokenizadores distintos requiere cuidado: cada uno está resolviendo unidades diferentes.

## Una caja de piezas que aprende a pegar

BPE añade piezas que representan parejas frecuentes de piezas anteriores. Empezamos con bytes. Contamos parejas adyacentes en los datos de entrenamiento, elegimos una y la sustituimos por un identificador nuevo. Repetimos hasta alcanzar el límite o no encontrar una pareja suficientemente repetida.

Para entender la mecánica usamos símbolos visibles. En `ababab`, la pareja ab aparece tres veces. Creamos X=ab y la secuencia queda XXX. Después podríamos crear Y=XX, que representa abab, y quedaría YX. El texto no ha cambiado; ha cambiado su segmentación.

La implementación `lumbre/tokenizer.py` es original, pequeña y deliberadamente simple: no aplica normalización, expresiones regulares de pretokenización ni tokens especiales. Su entrenamiento recuenta parejas en cada iteración, por lo que no es una implementación eficiente para corpus masivos.

## Parejas solapadas no pueden sustituirse dos veces

En `aaa`, la pareja aa aparece en posiciones 0–1 y 1–2, pero ambas ocurrencias comparten el carácter central. Si reemplazamos de izquierda a derecha sin solapamiento, obtenemos Xa, no XX.

El código `replace_pair` avanza dos posiciones cuando sustituye y una cuando no. Esa pequeña regla evita contar un mismo byte dos veces en la representación resultante.

```python
from lumbre.tokenizer import replace_pair
assert replace_pair([1,1,1], (1,1), 256) == [256,1]
```

Contar frecuencias de parejas y ejecutar sus sustituciones son pasos diferentes. El contador puede observar solapamientos; la sustitución debe producir una secuencia válida sin reutilizar posiciones.

## Un orden reproducible para los empates

Si dos parejas tienen la misma frecuencia, necesitamos una regla de desempate. El ejemplo elige de forma determinista según la pareja de identificadores. Sin esa decisión, dos entrenamientos del tokenizador podrían producir vocabularios distintos aunque los datos fueran iguales.

Cada nueva pieza solo puede referirse a piezas ya definidas. El constructor comprueba esa propiedad y rechaza referencias hacia el futuro o parejas duplicadas. Así, reconstruir los bytes de una pieza termina siempre siguiendo una lista ordenada.

::: definicion El vocabulario también es un programa
Los primeros 256 tokens representan bytes. Cada token posterior contiene una regla que concatena dos tokens anteriores. Guardar únicamente el número de tokens no basta: necesitamos las reglas y su orden para interpretar los identificadores.
:::

## Codificar y decodificar

Para codificar, se transforma el texto a bytes y se aplican las fusiones aprendidas en orden. Para decodificar, se concatenan los bytes de cada token y se interpreta el resultado como UTF-8.

Una secuencia generada arbitrariamente por un modelo de bytes puede no ser UTF-8 válido. El ejemplo de muestreo usa sustitución de secuencias inválidas para poder mostrar una cadena. Esa sustitución pertenece a la presentación; no demuestra que el modelo haya generado caracteres correctos.

```python
from lumbre.tokenizer import ByteBPE

text = "la memoria guarda datos. la memoria guarda ideas."
codec = ByteBPE.fit(text, max_merges=20)
ids = codec.encode(text)
assert codec.decode(ids) == text
assert codec.decode(codec.encode("un texto nuevo")) == "un texto nuevo"
copy = ByteBPE.from_state(codec.state())
assert copy.encode(text) == ids
```

El segundo `assert` comprueba que no dependemos de haber visto todos los caracteres durante el entrenamiento del tokenizador. La base de bytes hace posible representar textos nuevos, aunque su compresión en tokens sea peor.

## Ajustar el tokenizador solo con entrenamiento

La ruta BPE separa primero el texto en entrenamiento y validación y aprende las fusiones con la primera parte. Después codifica ambas con el mismo vocabulario. Así no se utilizan frecuencias de validación para elegir piezas.

Esto no resuelve automáticamente la contaminación del corpus. Si el texto repite las mismas frases en ambos lados, sigue habiendo solapamiento de contenido. Separar correctamente las etapas de preparación es necesario, pero también hay que diseñar una división que responda a la pregunta experimental.

Para documentos, una división por documento suele ser más informativa que cortar a mitad de una repetición. Para autores, periodos o dominios distintos, la separación debe reflejar qué generalización queremos medir.

## Usarlo con el compilador

El programa de entrenamiento acepta bytes y fusiones BPE mediante estas opciones:

```bash
python examples/train_decoder.py --byte-tokens --bpe-merges 30 --steps 20 --gemm blocked --output reports/mi_bpe
```

Con cero fusiones y `--byte-tokens` se usa el vocabulario de 256 bytes. Con un número positivo se añaden tantas fusiones como permita la regla de entrenamiento, hasta ese máximo. El número final no tiene por qué ser exactamente 256 más el límite solicitado: el algoritmo puede detenerse antes.

La dimensión de salida del modelo cambia con el vocabulario. El constructor, la pérdida, el muestreo y el checkpoint deben usar la misma tabla. El estado del tokenizador y el hash del corpus forman parte de la configuración guardada.

## Qué demuestra la prueba incluida

Las pruebas comprueban ida y vuelta con cadenas vacías, repeticiones, texto nuevo y caracteres multibyte; validan el orden de las reglas y el caso de solapamiento. Además, se ejecutó un entrenamiento corto con BPE que produjo pérdidas finitas.

Ese entrenamiento no produjo texto útil. Es una comprobación de integración: el pipeline puede cambiar de representación sin romper el compilador. La calidad lingüística exige una escala y evaluación distintas.

::: practica Diseñar una prueba que falle de verdad
¿Por qué `decode(encode(text)) == text` no basta para demostrar que dos tokenizadores asignan los mismos identificadores? Construye una explicación antes de seguir.
:::

Dos vocabularios pueden codificar el mismo texto de formas diferentes y decodificarlo correctamente. La ida y vuelta comprueba conservación del contenido, no identidad de segmentación. Para cargar pesos entrenados con otro tokenizador necesitamos igualdad del mapeo de identificadores, reglas, normalización y tokens especiales, no solo que ambos sepan leer español.

# Ejecutar el entrenamiento completo y leer sus resultados {#ch:train}

## El primer comando tiene un objetivo pequeño

Desde la carpeta del paquete, después de preparar el entorno descrito al principio, ejecuta:

```bash
python examples/train_decoder.py --steps 60 --gemm blocked --output reports/primer_decoder
```

El programa construye el modelo, crea el grafo de pérdida y AdamW, compila, inicializa parámetros y ejecuta actualizaciones. Después guarda un checkpoint, calcula una pérdida de validación sobre cuatro batches y genera una muestra.

Si no has exportado `PYTHONPATH` ni instalado el paquete, ejecuta el comando con `PYTHONPATH=.` en una terminal POSIX. El README incluye la instalación editable para evitar depender de esa variable. No cambies varias opciones a la vez durante la primera ejecución.

## Lo que ocurre en una iteración

Se eligen posiciones de inicio en los datos de entrenamiento. Se construyen entradas y etiquetas desplazadas. Se cargan las entradas y escalares de la actualización. El programa compilado calcula pérdida, gradientes, norma global, momentos nuevos y propuestas de pesos. Se comprueba finitud y se aplican las propuestas.

La preparación de índices y la selección del batch ocurren en Python. El cálculo tensorial del entrenamiento ocurre en el C generado. NumPy sirve para almacenar datos, inicializar pesos, comparar resultados y presentar muestras; no ejecuta en secreto el avance ni el retroceso del Transformer.

```diagram
Batch y estado actual | C compilado: loss, VJP, AdamW | Estado siguiente y registro
```

## Una configuración ejecutada durante la preparación

El siguiente comando reproduce la configuración principal de la edición. Los tiempos exactos dependerán del entorno y de si existe caché de compilación.

```bash
python examples/train_decoder.py --dim 64 --layers 2 --context 32 --steps 600 --gemm blocked --online --output reports/decoder_final
```

Se usaron 24 caracteres, batch 2, dos cabezas de consulta, una cabeza KV, dimensión intermedia 128 y 75,584 parámetros. La atención online se usa en el avance; su retroceso reconstruye la versión densa. Todo el experimento se ejecutó en CPU.

| Medida | Resultado observado | Interpretación |
|---|---|---|
| Actualizaciones | 600 | Pasos de AdamW, no épocas. |
| Primera pérdida de entrenamiento | 3.21643686 | Batch del primer paso. |
| Última pérdida de entrenamiento | 0.27465653 | Batch del paso 600. |
| Pérdida de validación | 0.33215784 | Media de cuatro batches del split descrito. |
| Tiempo de bucle | 3.43718 segundos | Excluye compilación, evaluación y muestreo. |
| Kernels del grafo de entrenamiento | 447 | Fronteras del plan generado. |
| Arena | 2,100,160 bytes | No es memoria total del proceso. |

La compilación reportó cero segundos porque se reutilizó un binario en caché. Eso no significa que el compilador sea instantáneo. El reporte conserva el hash de la unidad compilada y distingue esa fase del bucle.

## La muestra no autoriza una afirmación de calidad

La muestra observada comienza así:

```text
el sol rga yla rila laa la l miria lardiorgana dajreniza el trabajo.
el sol sala y la l
```

Hay fragmentos reconocibles y errores evidentes. La pérdida ha descendido, pero el modelo no produce lenguaje general fiable. Mostrar el texto real, en lugar de seleccionar una frase artificialmente corregida, permite juzgar qué se obtuvo.

El corpus contiene seis frases originales repetidas veinte veces. La división contigua 90/10 deja contenido repetido en ambos lados. La cifra de validación es útil como comprobación del pipeline, pero no como evidencia de generalización a documentos independientes.

::: cuidado No llamar benchmark a un corpus de depuración
Un resultado sobre repeticiones puede demostrar que la cadena de entrenamiento funciona y memoriza patrones. No demuestra comprensión, cobertura lingüística ni superioridad frente a otro modelo. La advertencia se guarda también en el JSON del experimento.
:::

## Una ejecución anterior también enseña algo

Se conserva un reporte de 600 pasos con GEMM de referencia y atención densa. Obtuvo una pérdida final aproximada 0.31436 y validación 0.30030, con 8.14276 segundos de bucle en aquella ejecución. No debe compararse con el nuevo tiempo como si fuera un benchmark controlado definitivo: cambian kernels y orden numérico, y las mediciones no forman una campaña estadística emparejada.

Los reportes históricos no se reescriben para que coincidan con el último código. Se identifican como experimentos anteriores. Para reanudar se usa el checkpoint del formato actual, que incluye configuración, vocabulario, tokenizador, hash del corpus, RNG y estado de AdamW.

## Reanudar significa añadir pasos

La opción `--steps` indica cuántos pasos ejecutar en esta invocación, también cuando se reanuda. Para continuar diez actualizaciones desde el checkpoint anterior:

```bash
python examples/train_decoder.py --dim 64 --layers 2 --context 32 --steps 10 --gemm blocked --online --resume reports/decoder_final/checkpoint.npz --output reports/decoder_continuado
```

La historia nueva empieza en 601. No se reinicia en uno. La configuración del modelo y el corpus deben coincidir con los del checkpoint; un desacuerdo produce un error.

La prueba incluida compara una ejecución continua de 10 pasos con otra de 6 más 4. En el mismo entorno CPU, todos los campos guardados coincidieron bit a bit, incluidos pesos, momentos, RNG y metadatos. El archivo `reports/resume_check.json` conserva esa comprobación.

## Entrenar con texto propio

La opción `--text` lee un archivo UTF-8. No descarga material ni verifica sus derechos de uso. Prepara un corpus cuya procedencia y permiso conozcas. El experimento debe registrar cómo se construyó, cuánto ocupa, cómo se dividió y qué transformaciones recibió.

```bash
python examples/train_decoder.py --text datos/corpus.txt --byte-tokens --bpe-merges 256 --dim 128 --layers 4 --heads 4 --kv-heads 2 --context 64 --steps 1000 --gemm blocked --output reports/corpus_propio
```

Esta configuración es una propuesta de experimento, no un resultado ejecutado en esta edición. Antes de lanzarla, estima memoria y coste. No añadas ceros al número de pasos para compensar un corpus pequeño o contaminado.

El entrenador mantiene una división sencilla de un único texto. Para una evaluación seria por documentos, la extensión de datos del proyecto final reemplaza esa política por manifiestos explícitos. El libro no presenta el cargador de depuración como una plataforma de datos de producción.

## La ruta GPU del mismo modelo

El mismo grafo puede enviarse al backend CUDA o HIP. En CUDA se exige declarar la arquitectura real. Un ejemplo para la RTX 5070 Ti Laptop validada, con destino `sm_120`, sería:

```bash
export LUMBRE_CUDA_ARCH=sm_120
python examples/train_decoder.py --backend cuda --steps 60 --gemm blocked --output reports/decoder_cuda
```

No copies ese destino sin comprobar el dispositivo. El código GPU implementado incluye operaciones elementwise, reducciones y matmuls genéricos, así como una ruta de GEMM compartido. No utiliza automáticamente el ejemplo WMMA ni se presenta como un backend tensor-core de producción.

La revisión local ejecutó el decoder de depuración de 2.736 parámetros durante 20 pasos en CUDA, con atención online y GEMM bloqueada. La pérdida pasó de 3,1800122 a 2,9162602; se conservan checkpoint, evaluación y registro en `code/reports/validation_20261001/decoder_cuda/`. HIP sigue pendiente. La aceptación del laboratorio exige volver a comprobar los resultados y medir en el dispositivo del estudiante.

También se ejecutó el modelo principal de 75.584 parámetros durante 600 actualizaciones en CUDA. Una campaña posterior repite esa configuración en CPU y CUDA en el mismo host, con tareas secuenciales y carpetas de salida nuevas. El capítulo «CPU y CUDA en la misma máquina» incluye las órdenes completas, los resultados y el análisis de por qué 447 kernels pequeños pueden limitar la ventaja de una GPU.

## Qué medir al aumentar el tamaño

Registra por separado construcción del grafo, compilación, primera ejecución, ejecución estable, copia del batch, checkpoint y evaluación. Cuenta tokens procesados y actualizaciones; no compares solo segundos entre configuraciones con contextos o batches distintos.

La arena estática de Lumbre informa de buffers administrados por el compilador. No incluye toda la memoria del compilador C, arrays de Python, bibliotecas cargadas o memoria interna del controlador. Presentarla como consumo total del ordenador sería incorrecto.

El primer objetivo del escalado es conservar corrección y observabilidad. Después se optimizan los cuellos de botella que revelan las medidas.

# Evaluar sin engañarse: datos, métricas y experimentos {#ch:evaluacion}

## Entrenamiento, validación y prueba responden preguntas distintas

Los datos de entrenamiento ajustan parámetros. La validación ayuda a escoger configuraciones y detener o comparar experimentos. Un conjunto de prueba reservado permite una evaluación final que no haya guiado repetidamente esas decisiones.

Si miramos el resultado de prueba después de cada cambio y elegimos el mejor, ya estamos utilizando ese conjunto como validación. El nombre de la carpeta no impide esa dependencia.

En un curso de compiladores, hay además otro eje: pruebas de corrección de operaciones frente a calidad del modelo. Un kernel correcto puede entrenar un modelo mediocre. Un modelo que parece bueno puede ocultar una operación incorrecta en casos que el corpus no visita.

## Pérdida y perplejidad

Cuando la pérdida es una media de negativos logaritmos naturales por token, su exponencial se llama perplejidad. Una pérdida de log(V) corresponde a la predicción uniforme sobre V alternativas bajo ese planteamiento.

No compares perplejidades de tokenizadores diferentes como si tuvieran la misma unidad. Un token largo puede representar más texto que uno corto. También importan las máscaras, los tokens especiales y qué posiciones entran en la media.

Para comparar compiladores, usa exactamente el mismo modelo, pesos, datos, tokenizador, semántica numérica y reducción de la pérdida. Para comparar sistemas completos, explica qué diferencias forman parte de la propuesta.

## Una pérdida finita puede estar mal normalizada

Supón dos secuencias: una tiene cuatro posiciones válidas y otra una. Si sumamos todas sus pérdidas y dividimos entre ocho posiciones de un tensor rellenado, la pérdida parece artificialmente pequeña. El divisor correcto depende de la definición: cinco tokens válidos si buscamos una media por token.

Las posiciones de padding necesitan una máscara de pérdida distinta de la máscara causal. Una impide consultar el futuro; la otra decide qué etiquetas cuentan en la evaluación. Usar una no crea automáticamente la otra.

El entrenador didáctico usa ventanas completas sin padding de entrenamiento. Al extenderlo a documentos variables, se debe añadir esta distinción y probarla con ejemplos donde las cantidades válidas sean diferentes.

## Contaminación y duplicados

Separar por filas al azar puede dejar copias del mismo documento en varios splits. También puede dejar fragmentos casi idénticos o versiones de un mismo texto. Una métrica favorable puede reflejar ese solapamiento.

Una práctica reproducible guarda identificadores de documentos y hashes, y define reglas de deduplicación antes de evaluar. El hash exacto detecta copias idénticas, pero no paráfrasis ni pequeñas ediciones. No se le atribuye una capacidad semántica que no tiene.

En nuestro corpus repetido el solapamiento es intencional y se declara. Convertirlo en un test de generalización exige cambiar el diseño de datos, no solo renombrar el campo `validation_loss`.

## Comparación de implementaciones con los mismos pesos

Para aislar un cambio del compilador, primero compara un avance con pesos fijos. Después compara gradientes. Después una actualización. Finalmente compara una trayectoria corta de entrenamiento.

Pequeñas diferencias de redondeo pueden amplificarse tras muchos pasos. Que dos trayectorias de 600 pasos no sean bit a bit iguales no demuestra por sí solo un error. Pero tampoco autoriza a ignorar una discrepancia grande en el primer avance.

La estrategia consiste en localizar el primer nivel donde aparece una diferencia y cuantificarla. Las tolerancias se justifican por operaciones y tipos; no se aumentan indefinidamente hasta que cualquier resultado pase.

## Ablaciones: cambiar una cosa con una razón

Una ablación elimina o sustituye un componente para investigar su efecto. En un compilador, podemos desactivar fusión, reutilización de memoria o GEMM bloqueado. En el modelo, podemos cambiar atención online por densa manteniendo la misma operación matemática.

Una tabla útil conserva varias dimensiones del resultado: error, tiempo, memoria y coste de compilación. Es posible mejorar una y empeorar otra. El proyecto de atención online mostrará precisamente ese caso.

::: practica Una conclusión que los datos no permiten
«La versión B es mejor porque usa menos memoria». ¿Qué falta para que esa frase responda a un objetivo de sistema?
:::

Falta definir qué significa mejor: quizá el usuario necesita latencia, throughput, capacidad para un contexto mayor o menor coste total. Menos memoria puede permitir una configuración antes imposible, pero también puede aumentar el tiempo de una que ya cabía. La conclusión debe nombrar el objetivo y las condiciones.

## Semillas y variabilidad

Una semilla fija facilita repetir un caso, pero no convierte una única ejecución en una estimación completa de calidad. Para comparar recetas de entrenamiento conviene usar varias semillas y presentar dispersión, especialmente cuando las diferencias son pequeñas.

Para microbenchmarks, repeticiones dentro del mismo proceso estiman variación temporal local; procesos o sesiones distintas capturan otros efectos. Las dos clases de repetición no son intercambiables. Un millón de medidas casi idénticas de la misma condición no cubre un cambio de frecuencia o una actualización del controlador.

## Informe final de un experimento

El informe debe permitir responder qué se cambió, por qué, con qué datos y entorno, qué se esperaba, qué se observó y qué sigue sin comprobarse. Incluye casos negativos y fallos: son información sobre el dominio de la implementación.

La última frase debe ser proporcional a la evidencia. «En esta CPU, para estas formas y esta precisión, la variante redujo el tiempo mediano» es una conclusión útil. «Hemos superado todas las bibliotecas de IA» no se deduce de una tabla de tres tamaños.
