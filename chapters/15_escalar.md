# Escalar un modelo: primero contar memoria y trabajo {#ch:escalar}

::: idea El tamaño de los pesos no es el tamaño del entrenamiento
Guardar un libro ocupa una estantería. Trabajar sobre él puede necesitar otra para anotaciones, borradores y copias temporales. Un entrenamiento mantiene pesos, gradientes, estados del optimizador, activaciones y buffers de ejecución.
:::

## Un presupuesto por categorías

Para P parámetros en FP32, los pesos ocupan 4P bytes. Si guardamos gradientes del mismo tipo, añadimos 4P. Dos momentos FP32 de Adam añaden 8P. Esa suma simple da 16P bytes antes de activaciones y otros temporales.

En precisión mixta puede haber pesos de trabajo de menos bytes, copias maestras FP32 y gradientes de un tipo elegido. No existe una cifra universal de bytes por parámetro para todas las implementaciones. Hay que enumerar lo que realmente se almacena y cuándo está vivo.

Lumbre conserva los parámetros y calcula propuestas de actualización como salidas separadas antes de asignarlas. Por tanto, un presupuesto basado solo en pesos y momentos no representa su pico completo. El planificador de arena da una cuenta más cercana de sus buffers, pero tampoco incluye toda la memoria del proceso.

| Categoría | Pregunta de dimensionamiento | Estrategia posible |
|---|---|---|
| Pesos | ¿Cuántos parámetros y qué tipo? | Tipos compactos o partición. |
| Gradientes | ¿Todos simultáneos o por grupos? | Acumulación y liberación planificada. |
| Optimizador | ¿Qué estados por parámetro? | Partición u offload. |
| Activaciones | ¿Qué necesita el backward? | Recomputation o checkpointing. |
| Temporales | ¿Qué kernels materializan? | Fusión, arena y algoritmos con menos memoria. |

## Una cuenta que descarta un plan antes de ejecutarlo

Con siete mil millones de parámetros y un presupuesto de 16 bytes por parámetro, el estado básico suma 112 GB decimales. Eso ya supera con mucho una tarjeta de 12 GB, antes de añadir activaciones.

No se arregla reduciendo únicamente el batch a uno. El batch afecta a activaciones, pero no elimina el estado de todos los parámetros entrenables. Cuantizar pesos para inferencia o entrenar solo un adaptador responde a tareas distintas de preentrenar todos los pesos.

La cuenta no impide trabajar con modelos grandes mediante partición, descarga a otra memoria o entrenamiento parcial. Obliga a nombrar esas decisiones y sus costes, en lugar de prometer un entrenamiento completo a partir del tamaño del archivo de inferencia.

## Una arquitectura intermedia concreta

Entre 75,584 parámetros y un modelo de miles de millones hay etapas útiles. Una configuración hipotética con V=32768, D=768, L=12, H=12, HK=4, F=3072 tiene 128,994,048 parámetros con nuestra fórmula y pesos compartidos de entrada y salida.

```python
from lumbre.nn import Config

config = Config(vocab=32768, batch=1, context=512,
                dim=768, heads=12, kv_heads=4,
                layers=12, hidden=3072, online=True)
```

Este bloque define una configuración; no informa de un entrenamiento ejecutado. Antes de construir todos sus arrays o compilar, estima el estado y prueba una versión menor. La implementación BPE didáctica tampoco está diseñada para aprender decenas de miles de fusiones con eficiencia industrial.

El objetivo de esta etapa es convertir el escalado en un proceso con puertas de aceptación: formas correctas, memoria dentro del presupuesto, paso numérico verificado, rendimiento estable y checkpoint probado. No se salta directamente a una escala enorme para descubrir errores básicos allí.

## Activaciones: el contexto puede dominar

Un tensor de activaciones B×T×D en FP32 ocupa 4BTD bytes. Si guardamos varios por capa, el factor del número de capas importa. La atención densa añade términos proporcionales a BHT², que pueden dominar al aumentar el contexto.

Para B=1, T=2048, D=768, una sola activación B×T×D ocupa aproximadamente 6 MiB. Multiplicar esa cifra por todas las activaciones vivas de varias capas produce una cantidad mucho mayor. No basta con contar la salida de cada bloque: el backward puede necesitar proyecciones, normalizaciones, puertas y estadísticas internas.

La fusión elimina algunos intermediarios; el checkpointing evita conservar otros; la atención online evita ciertas matrices cuadráticas. Son mecanismos diferentes y sus ahorros no deben sumarse sin analizar posibles solapamientos.

## Checkpointing de activaciones no es guardar el entrenamiento en disco

El mismo término se usa para dos trabajos. Un checkpoint persistente permite recuperar un entrenamiento tras interrumpirlo. El checkpointing de activaciones guarda solo algunos estados del avance y recalcula los demás durante el retroceso.

Imagina una cadena de doce bloques. Podemos guardar la entrada de cada grupo de tres y recomputar su interior cuando toque derivarlo. Reducimos activaciones retenidas, pero repetimos cómputo. La política óptima depende de costes desiguales, memoria disponible y posibilidades de solapamiento.

Un compilador puede representar esa decisión insertando subgrafos de recomputación o cambiando la planificación. Debe conservar los valores aleatorios y efectos necesarios para repetir la misma función. No se puede recomputar libremente una operación que consume un estado externo distinto cada vez.

## Cuánto trabajo aproxima un entrenamiento denso

Para una capa lineal grande, el avance cuesta aproximadamente dos operaciones por multiplicación-acumulación y el backward requiere productos adicionales. De ahí surge una estimación habitual del orden de 6·P·N operaciones para entrenar un modelo denso sobre N tokens, bajo supuestos simplificados.

No es una ley exacta. La atención dependiente del contexto, embeddings, normalizaciones, activaciones, recomputación, comunicación y estructuras dispersas cambian la cuenta. En un MoE no se usa sin pensar el total de parámetros como si todos participaran en cada token.

Para una planificación inicial, escribe la estimación y sus términos omitidos. Después calibra con un paso real de una configuración pequeña o intermedia. Si la estimación y la observación difieren, investiga el cuello de botella antes de extrapolar varios órdenes de magnitud.

## Datos y parámetros compiten por el presupuesto

Aumentar el modelo sin aumentar adecuadamente datos y entrenamiento puede no ser la mejor utilización de un presupuesto. Los trabajos de escalado compute-optimal estudian precisamente esa distribución de recursos; no ofrecen una regla universal independiente del dominio, la receta y los objetivos posteriores. [@chinchilla]

Un curso de compiladores no necesita entrenar todos los puntos de una ley de escalado para aprender de ella. Sí necesita evitar la conclusión de que más parámetros implican automáticamente mejor experimento. Un modelo menor bien entrenado y evaluado puede ser una referencia mucho más útil para optimizar el sistema.

## Puertas de aceptación antes de un entrenamiento largo

Primero, un paso completo produce valores finitos y gradientes contrastados. Segundo, varias decenas de pasos sobre un batch fijo reducen una pérdida controlada. Tercero, una ejecución con datos reales conserva throughput y memoria. Cuarto, guardar y reanudar produce el estado esperado. Quinto, el evaluador utiliza un split independiente y una métrica definida.

Solo después se reserva una ejecución larga. Si la preparación de datos tarda más que el kernel, optimizar el kernel no resuelve el sistema. Si el checkpoint tarda mucho y bloquea el dispositivo, debe medirse como parte de la operación real.

# Precisión mixta y cuantización: ahorrar sin cambiar a ciegas el cálculo {#ch:precision-mixta}

## Almacenar, multiplicar y acumular pueden usar tipos distintos

Un GEMM puede leer valores de baja precisión, multiplicarlos mediante una unidad especializada y acumular en una precisión mayor. El resultado puede volver a almacenarse en otro formato. Esos son tres puntos de redondeo potencialmente diferentes.

El dtype de un tensor de entrada no describe toda la semántica numérica del kernel. Para comparar implementaciones necesitamos tipos de almacenamiento, multiplicación, acumulación, epílogo y salida, además de escalas y modos de redondeo.

En Lumbre el camino principal es FP32. El ejemplo WMMA ilustra entradas FP16 y acumulación FP32, pero no está integrado como política automática de entrenamiento. El estudiante debe añadir una representación explícita de tipos antes de extender esa ruta de forma general.

## Por qué el backward puede necesitar otro rango

Un valor de activación puede ser moderado mientras un gradiente es muy pequeño. Redondear ambos con el mismo formato puede tener efectos distintos. Una actualización menor que la separación entre números representables del peso puede desaparecer al almacenarla.

Una copia maestra de mayor precisión puede conservar actualizaciones pequeñas y producir pesos de trabajo de menor precisión para el avance. Eso aumenta memoria respecto a guardar solo los pesos compactos, pero evita perder sistemáticamente cambios.

La decisión debe verificarse con un problema donde las actualizaciones pequeñas importen. Un test que usa únicamente números enteros y pasos grandes no evalúa esa propiedad.

## Escalado de la pérdida

Multiplicar la pérdida por un factor también multiplica sus gradientes. Podemos calcular esos gradientes escalados y dividirlos antes de aplicar la actualización. La idea ayuda cuando un formato tiene un rango insuficiente para gradientes pequeños, pero puede introducir desbordamiento si el factor es excesivo.

Un escalado dinámico necesita detectar valores no finitos, decidir si omite la actualización y ajustar el factor. El contador del optimizador y el scheduler de tasa deben tener un comportamiento definido en un paso omitido. No se limita a multiplicar una línea de código.

BF16 y FP16 tienen rangos y precisiones diferentes, por lo que no debe trasladarse mecánicamente una receta de uno a otro. Tampoco se supone que todo kernel de un dispositivo admita ambos tipos con la misma eficiencia.

## Cuantización simétrica con una cuenta pequeña

En un esquema propio simple, aproximamos x por $s q$, donde q es un entero limitado y s una escala positiva. Elegimos q redondeando x/s y saturando al rango permitido.

Con s=0.1, el valor 0.26 se aproxima por 0.3 si el redondeo produce 3. Con s=1, se aproxima por 0. Ese ejemplo muestra el compromiso: una escala grande cubre magnitudes mayores, pero pierde detalle en valores pequeños.

Si un grupo contiene un valor 100 y muchos valores 0.01, una única escala puede representar mal la mayoría. Escalas por canal o por bloque reducen ese conflicto a cambio de metadatos y operaciones adicionales.

## La calibración no debe mirar el examen

Un esquema puede elegir escalas a partir de estadísticas de datos de calibración. Esos datos deben estar definidos y separados de la evaluación final. Elegir la escala que produce mejor métrica en el conjunto de prueba es utilizarlo para ajustar el sistema.

El test del kernel cuantizado compara primero con una referencia que aplique exactamente la misma cuantización. Después se compara el modelo cuantizado con el modelo de mayor precisión para medir degradación de calidad. Son dos preguntas distintas: corrección de la implementación y efecto de la aproximación.

## Formatos muy compactos y escalas por bloques

Los repositorios recientes de kernels incluyen formatos compactos y esquemas de escala por bloques. La compatibilidad depende de arquitectura, layout y operación; no se deduce únicamente del nombre FP8 o FP4. [@cutlass;@deepgemm;@deepgemm-ascend]

Para diseñar una IR, un tensor cuantizado debe transportar más que un array. Necesita identificar valores, escalas, ejes de agrupación, posible zero-point, formato y reglas de reconstrucción. Si se pierde esa información al hacer un reshape, el resultado puede seguir ocupando los mismos bytes y representar números diferentes.

Un movimiento es gratuito solo si conserva la interpretación completa del dato. Reorganizar escalas junto con valores puede requerir trabajo real o impedir una fusión.

## Una política de error por operación

No todas las operaciones toleran el mismo error. Una suma larga acumula redondeo; una comparación cercana al umbral puede cambiar una decisión; un softmax puede amplificar diferencias de logits; una normalización tiene divisiones sensibles a magnitudes pequeñas.

Una política razonable empieza por conservar mayor precisión en reducciones y estadísticas críticas, y evalúa dónde reducir almacenamiento o multiplicación. El resultado final se comprueba en el modelo, no solo en kernels aislados.

::: practica Un falso ahorro
Un tensor baja de cuatro a un byte por elemento, pero el kernel lo convierte completo a FP32 en un buffer temporal y después ejecuta el mismo GEMM. ¿Qué ha mejorado y qué puede no haber mejorado?
:::

Puede haber disminuido el almacenamiento persistente o el tráfico desde una fuente externa. Pero el pico temporal puede seguir siendo alto, y el coste de conversión puede empeorar la latencia. Para aprovechar cómputo de baja precisión necesitamos una ruta que lo use realmente, no solo un archivo comprimido.

# Entrenamiento distribuido: repartir estado y reunir contribuciones {#ch:distribuido}

## Dos estudiantes calculan mitades de un lote

Supón que un lote contiene ocho ejemplos y dos dispositivos procesan cuatro cada uno con los mismos pesos. Cada uno obtiene una media de gradientes local. Para reproducir la media de los ocho, promediamos ambas contribuciones porque los tamaños coinciden.

Si uno procesa seis y otro dos, las medias locales deben ponderarse 6/8 y 2/8. Promediarlas con peso un medio cambia el objetivo. La matemática de la reducción debe definirse antes de escoger una biblioteca de comunicación.

En paralelismo de datos, los parámetros suelen replicarse y los gradientes se combinan. La memoria de activaciones por dispositivo puede bajar al dividir el lote, pero los pesos replicados siguen ocupando espacio en cada uno.

## All-reduce, reduce-scatter y all-gather

All-reduce combina contribuciones y entrega el resultado completo a todos los participantes. Reduce-scatter combina y reparte partes del resultado. All-gather reúne partes para que cada participante obtenga la colección completa.

Imagina dos vectores [1,2] y [3,4]. Una suma all-reduce entrega[4,6] a ambos. Una reduce-scatter puede entregar[4] al primero y [6] al segundo. Un all-gather posterior vuelve a construir[4,6] en ambos.

Estas operaciones no son solo funciones matemáticas puras desde el punto de vista del runtime: los participantes deben ejecutarlas de forma compatible y sus buffers tienen dependencias de finalización. NCCL y RCCL son referencias de infraestructura colectiva; el repositorio de RCCL consultado remite a `rocm-systems`. [@nccl;@rccl]

## Particionar estados para no repetirlos

Si todos los dispositivos guardan los mismos momentos de Adam, parte de la memoria está duplicada. La familia de ideas de ZeRO reparte estados y, en distintas etapas, gradientes y parámetros, coordinando comunicación para conservar el entrenamiento. [@zero]

La cuenta ideal de memoria puede mejorar con el número de dispositivos, pero no desaparece el coste de mover datos ni los temporales necesarios. Un plan de partición debe explicar cuándo cada dispositivo necesita una parte que no posee y cómo la obtiene.

La elección también afecta al checkpoint: guardar únicamente el fragmento de un dispositivo no produce un modelo completo salvo que el formato y el manifiesto describan el resto. La reanudación con otro número de dispositivos puede necesitar redistribución.

## Tensor parallelism con una multiplicación pequeña

Para Y=XW, podemos repartir columnas de W. Cada dispositivo calcula columnas distintas de Y. Si la operación siguiente acepta esa partición, quizá no sea necesario reunir Y inmediatamente.

También podemos repartir el eje de reducción de W y las columnas correspondientes de X. Cada dispositivo calcula una suma parcial de Y y hay que combinar esas parciales. Son particiones diferentes, con comunicaciones diferentes.

El trabajo de Megatron-LM es una referencia histórica importante para organizar paralelismo dentro de capas Transformer. La derivación de los dos casos anteriores es suficiente para empezar a razonar sobre dónde se necesitan colectivas en nuestro propio grafo. [@megatron]

## No reunir un tensor solo para volver a partirlo

Un compilador consciente de particiones puede conservar layouts distribuidos entre operaciones. Por ejemplo, una activación elementwise puede aplicarse a cada fragmento sin reunir todo el tensor. Una normalización sobre un eje partido necesita estadísticas globales o una estrategia equivalente.

Por eso el tipo de una IR distribuida debería incluir qué dimensiones están particionadas, qué valores son parciales y qué grupo de dispositivos participa. Un tensor local y una suma parcial de un tensor global no son intercambiables aunque tengan el mismo shape local.

::: ejemplo Una suma parcial que parece una salida completa
Dos dispositivos calculan cada uno la mitad del eje K de un GEMM. Ambos producen arrays M×N. Ninguno es todavía la salida Y: falta sumar las contribuciones. La forma por sí sola no expresa esa obligación.
:::

## Pipeline parallelism y burbujas

Podemos repartir capas en etapas. El primer dispositivo procesa las primeras y envía activaciones al segundo. Si solo hay un microbatch, algunos esperan mientras otros trabajan. Al introducir varios microbatches se puede llenar la tubería, pero aparecen más estados vivos y una planificación de avance y retroceso.

Una burbuja es un intervalo donde una etapa no tiene trabajo útil listo. Reducir burbujas puede aumentar memoria o complejidad de comunicación. La planificación debe respetar que un backward utiliza parámetros coherentes con el forward correspondiente según la receta elegida.

No basta con dibujar flechas diagonales bonitas. El calendario debe especificar qué microbatch, fase y versión de parámetros ocupa cada celda. Ese calendario puede comprobarse con una simulación antes de desplegarlo.

## Solapar comunicación y cómputo

Cuando un grupo de gradientes está listo, puede iniciarse su comunicación mientras se calculan otros. Para que el solapamiento sea correcto, el buffer no puede sobrescribirse ni leerse como resultado global antes de finalizar.

Agrupar muchos gradientes amortiza costes de lanzamiento de comunicación, pero retrasa el momento en que el primer grupo puede empezar. Agrupar muy poco aumenta overhead. La política de buckets es una decisión de scheduling con un compromiso medible.

El tiempo ideal de dos fases solapables se aproxima al máximo de sus duraciones, no a su suma. Sin embargo, comparten recursos y pueden interferir; un solapamiento dibujado en el grafo no garantiza el mismo solapamiento físico.

## Fallos y orden colectivo

Si un participante ejecuta dos colectivas en un orden diferente al resto, puede haber bloqueo o resultados inválidos. Si un proceso falla, los demás necesitan detectar que la operación no completará normalmente. Un timeout es una señal de fallo, no una prueba automática de qué nodo lo causó.

Las pruebas distribuidas incluyen tamaños pequeños, participantes con datos vacíos según el contrato, orden de colectivas y reanudación. Se registran todos los ranks y no solo el proceso que imprime la pérdida.

Lumbre no incluye un runtime distribuido. Este capítulo define su diseño y criterios de aceptación como ampliación avanzada. Conectar una biblioteca colectiva sería una integración declarada, no una funcionalidad que exista por aparecer en la bibliografía.

# Mezclas de expertos y modelos de frontera en 2026 {#ch:moe}

## Más especialistas, no todos trabajando a la vez

En una mezcla de expertos, un router selecciona qué subredes procesan cada token. El modelo puede contener muchos parámetros totales y activar solo una parte por token. Eso reduce el cómputo respecto a ejecutar todos los expertos, pero no elimina la necesidad de almacenar o distribuir sus pesos.

Imagina una biblioteca con varios especialistas. Cada consulta se envía a dos de ellos y después se combinan sus respuestas. Hay tres trabajos: elegir destinos, transportar consultas y ejecutar especialistas. Optimizar solo uno puede dejar los otros dominando la latencia.

## Un ejemplo de dispatch completo

Tenemos cuatro tokens t0,t1,t2,t3 y dos expertos A,B. El router decide: t0 va a A; t1 a B; t2 a A; t3 a B. Podemos formar dos lotes compactos, [t0,t2] y[t1,t3], ejecutar cada experto y devolver resultados a las posiciones originales.

Si cada token usa dos expertos con pesos distintos, necesita dos contribuciones y una combinación ponderada. El manifiesto de dispatch debe conservar identificador del token, experto y peso. Perder el orden original produce una salida del shape correcto con contenido asignado a ejemplos equivocados.

```diagram
Tokens y rutas | Lotes compactos por experto | Restaurar orden y combinar
```

## Expertos vacíos y tamaños irregulares

Un experto puede no recibir tokens. Otro puede recibir casi todos. Un kernel de grouped GEMM debe tratar esos tamaños sin leer datos inexistentes ni gastar un gran bloque en trabajo vacío.

Las decisiones de capacidad y balance afectan al modelo, no solo al rendimiento. Descartar tokens que exceden una capacidad cambia el cálculo y debe formar parte de la receta de entrenamiento. No es una optimización transparente.

El routing top-k es discreto. El entrenamiento de un router necesita una definición de qué dependencias se derivan y qué funciones auxiliares se usan. No se obtiene una derivada de la decisión de selección simplemente porque el resto del grafo sea diferenciable.

## Comunicación y cómputo son bibliotecas distintas por una razón

DeepEP se centra en mover y combinar trabajo de expertos; DeepGEMM en el cómputo matricial. Sus variantes Ascend muestran que el cambio de hardware afecta tanto al transporte y sincronización como a las multiplicaciones. [@deepep;@deepgemm;@deepep-ascend;@deepgemm-ascend]

Una evaluación de MoE debe incluir el coste de empaquetado, dispatch, expertos y combine. Un GEMM muy rápido en un experto aislado no demuestra un paso completo rápido cuando los tokens se distribuyen mal o atraviesan una red.

## Un mapa de diferencias respecto al decoder del curso

El decoder de Lumbre es denso en su feed-forward y utiliza atención causal estándar con GQA. No implementa routing de expertos, atención comprimida especializada, optimizador Muon ni una receta de entrenamiento de frontera.

El informe público de DeepSeek-V4 describe variantes con contexto de un millón de tokens y arquitecturas MoE; comunica, entre otros elementos, atención comprimida, mHC y uso de Muon. La ficha de Pro distingue 1.6 billones de parámetros totales y 49 mil millones activos; son cifras del autor, no resultados reproducidos aquí. [@deepseek-v4]

Ese contraste evita llamar «réplica de DeepSeek» a un pequeño decoder que comparte solo algunas piezas. La utilidad del libro es que permite identificar qué habría que añadir al lenguaje de operaciones, al scheduler, al runtime y a la evaluación para acercarse a esas arquitecturas.

## Qué exigiría una reproducción de calidad competitiva

Primero se fija un modelo y una versión de referencia, no una etiqueta genérica SOTA. Después se reproduce su tokenización, arquitectura, inicialización, datos y receta de entrenamiento. Se verifican operaciones y gradientes con pesos comparables. Se implementan los mecanismos de memoria y distribución necesarios. Finalmente se evalúa con protocolos y tareas definidos.

Una mejora de compilador puede evaluarse manteniendo la calidad constante y reduciendo coste. Una mejora de modelo necesita evaluar calidad y coste conjuntamente. Mezclar ambas sin control permite atribuir al compilador una ganancia que vino de cambiar el objetivo o usar más datos.

La escala de frontera requiere recursos y evidencia que no aporta el experimento local de este libro. El temario incluye sus técnicas y un itinerario de implementación; no se presenta un entrenamiento no realizado como si existiera.

# Generación eficiente: la caché KV es parte del sistema {#ch:kv-cache}

## Recalcular todo es correcto, pero caro

El ejemplo de muestreo vuelve a ejecutar una ventana completa cada vez que añade un token. Es sencillo de comprobar porque utiliza el mismo avance que ya conocemos. Pero repite proyecciones de tokens anteriores.

En generación causal, las claves y valores de posiciones pasadas pueden conservarse y reutilizarse. Para una nueva posición calculamos su consulta y sus nuevos K,V, añadimos estos al historial y atendemos sobre lo disponible. Esa estructura se llama caché KV.

La caché no contiene respuestas finales ni sustituye los pesos. Guarda activaciones de un historial concreto, por capa y cabeza KV. No puede reutilizarse entre secuencias diferentes sin una política de prefijos compartidos que demuestre compatibilidad.

## Una cuenta de memoria

Para B secuencias, L capas, T posiciones almacenadas, HK cabezas KV y dimensión d, K y V ocupan aproximadamente $2BLTH_Kd$ elementos, antes de metadatos o padding. Multiplica por bytes del formato usado.

GQA reduce HK respecto al número de consultas, lo que puede disminuir esa memoria. Pero si una implementación materializa copias de K,V para cada cabeza de consulta, puede perder parte del beneficio de almacenamiento. La arquitectura y el layout de la caché deben diseñarse juntos.

## Posiciones y desplazamientos

Al procesar solo el token nuevo, su posición no vuelve a cero. RoPE debe usar la posición correspondiente dentro de la secuencia o la política de ventana definida. Reiniciar el ángulo en cada llamada cambia la atención.

También hay que gestionar el límite de contexto. Un buffer circular necesita traducir posición lógica a posición física y conservar el orden semántico de las claves. Una máscara construida para memoria contigua puede ser incorrecta después de envolver el buffer.

Un test compara generación incremental con un avance completo de la misma secuencia y pesos. Se comprueban logits de cada posición, no solo el token muestreado: dos distribuciones distintas pueden producir el mismo token por casualidad.

## Páginas para longitudes variables

Reservar el máximo contexto para todas las solicitudes desperdicia memoria cuando muchas son cortas. Una organización por bloques o páginas asigna capacidad conforme crece cada secuencia y mantiene una tabla que traduce posiciones lógicas a bloques físicos.

PagedAttention es una referencia de esta conexión entre gestión de memoria y servicio de LLM. Su contribución no equivale a sustituir la atención por una función de sistema operativo; reorganiza cómo se almacena y utiliza la caché. [@paged]

Un kernel que lee una caché paginada debe seguir esos índices correctamente. La indirección añade trabajo y puede cambiar la localidad, pero permite utilizar mejor la capacidad total y organizar más solicitudes simultáneas.

## Prefill y decode necesitan métricas separadas

Prefill procesa el contexto inicial. Decode genera tokens nuevos de forma incremental. Una aplicación puede tener un prefill largo y muchos pasos de decode, o solicitudes cortas con pocos tokens de salida.

La latencia hasta el primer token, el tiempo por token posterior y el throughput agregado miden aspectos diferentes. Un cambio que mejora throughput mediante lotes mayores puede aumentar la espera de una solicitud individual.

Un compilador orientado a servicio debe conservar esa distinción en su selección de kernels. La configuración óptima de una matriz grande del entrenamiento no representa necesariamente el producto vector-matriz de decode.

## Caché y entrenamiento no son la misma ruta

La caché KV de inferencia suele guardar valores para reutilizarlos sin construir un grafo de retroceso de todo el historial. El entrenamiento necesita dependencias y gradientes diferentes. Activar una API de caché de forward no garantiza soporte de backward.

Por eso el proyecto de caché del libro se plantea como una ampliación de inferencia con pruebas de equivalencia. No se utiliza para afirmar que el entrenamiento de contexto largo ya está resuelto.
