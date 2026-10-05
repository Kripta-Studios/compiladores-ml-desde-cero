# Semanas 6–8: del rango del compilador al kernel {#cuda:syllabus}

## Conectar las prácticas con el syllabus

Compilers for Machine Learning introduce memoria, multihilo y rangos `LOOP`, `UPCAST` y `THREAD` en CPU durante la semana 6. Las semanas 7–8 añaden GPU, mapeo de ejes y tensor cores. Este capítulo, incorporado el 5 de octubre de 2026, propone ejercicios para integrar esas ideas en el compilador propio. Las mediciones del cuaderno conservan el alcance de su campaña original.

Los programas de vectores, reducción, GEMM y streams son referencias ejecutables. Los ejercicios siguientes son trabajo de ampliación, con soluciones razonadas; no se presentan como nuevas ejecuciones de tensor cores, HIP o comparaciones con PyTorch.

## Un eje lógico y tres decisiones

Para `y[i]=(x[i]+2)*3`, el grafo puro contiene forma y operaciones aritméticas. Rangeify convierte la forma en rangos y accesos escalares. El planificador decide después qué trabajo es secuencial, se desenrolla o se distribuye. El renderer recibe esa decisión y sus límites; no debería inferir paralelismo a partir de nombres de variables.

| Decisión | Pregunta antes de generar CUDA | Evidencia |
|---|---|---|
| Dominio | ¿Qué posiciones debe producir el programa? | Intervalo lógico `[0,N)` |
| Reparto | ¿Qué hilo calcula cada posición? | Fórmula de índice y cobertura |
| Dependencias | ¿Qué debe terminar antes de usar cada dato? | Grafo de efectos y sincronización |
| Recursos | ¿Qué almacenamiento y bloque requiere? | Validación frente al dispositivo |

En CUDA, bloques e hilos identifican el trabajo y los hilos de un bloque pueden cooperar mediante memoria compartida y sincronización de bloque. Estas garantías no establecen un orden global entre bloques. Consulta el contrato del modelo de ejecución antes de escoger una bajada concreta. [@cuda-guide]

## Ejercicio resuelto: 257 salidas

Una primera política usa un hilo por salida y 128 hilos por bloque. Para N positivo, la cantidad de bloques es el techo de N dividido entre 128. Para N=257 son tres bloques, 384 posiciones de hilo y 127 posiciones que no escriben. El índice es `bloque*128+hilo`; la guarda de escritura es `i<N`.

**Cobertura.** Cada i válido tiene cociente y resto únicos al dividir por 128, que identifican su bloque e hilo. **Límites.** La guarda elimina índices entre 257 y 383. **Caso vacío.** Para N=0, el runtime puede omitir el lanzamiento y devolver la salida vacía según su contrato.

Compara kernel manual, C escalar y CUDA generado. Usa N igual a 0, 1, 127, 128, 129 y 257. Cambia los datos entre dos llamadas: acertar una única entrada no demuestra que el runtime actualice sus argumentos.

## UPCAST no cambia la precisión

Supón que cada hilo calcula cuatro posiciones. Una política usa `i=base+hilo+q*128`, con q entre cero y tres y `base=bloque*512`. A q fijo, hilos consecutivos acceden a posiciones consecutivas. Otra asigna `i=4*(bloque*128+hilo)+q`; cubre el dominio, pero cambia el patrón simultáneo de accesos.

El upcasting puede exponer las cuatro operaciones como expresiones separadas y mantener varios acumuladores. Su beneficio depende de registros, instrucciones y memoria. Compara ambas políticas con el mismo contrato, sin asumir que cuatro elementos por hilo será siempre mejor.

**Solución para N=513.** En la primera política hacen falta dos bloques. El primero cubre 512 posiciones. En el segundo solo el hilo cero con q=0 produce un elemento válido. Cada escritura potencial necesita su propia condición; comprobar únicamente la base del hilo puede permitir accesos fuera de rango.

## THREAD en un eje de reducción

Cambiar un rango de `LOOP` a `THREAD` no basta si todos los participantes actualizan el mismo acumulador. Diseña parciales privados, cooperación dentro del bloque y combinación de bloques. La práctica existente combina parciales en CPU; una extensión puede usar un segundo kernel.

Para N=257 y bloques de 128, la primera fase produce tres parciales y la segunda los reduce. Usa la identidad cero para posiciones enmascaradas y prueba valores positivos, negativos y cancelación. Si cambia el orden FP32, compara según la tolerancia del contrato.

Una barrera de bloque coordina producción y consumo dentro del bloque. Para participantes aún activos, ponerla en una condición que algunos cumplen y otros no puede dejar la sincronización sin un contrato válido. Los hilos de borde deben seguir la estrategia de participación y enmascarado prevista. [@cuda-guide]

## STAGE, AFTER y vida de temporales

`STAGE` representa un temporal indexable. El plan debe identificar productor, consumidores, almacenamiento y última lectura. Una arena solo puede reutilizar el espacio cuando las dependencias lo permitan.

Imagina un productor P y consumidores Q y R. Que Q termine no permite sobrescribir el temporal mientras R siga pendiente. Una planificación secuencial en una cola puede establecer suficiente orden; varias colas necesitan dependencias explícitas entre ellas. La vida del objeto del host no determina la finalización de una lectura del dispositivo.

`GROUP` reúne resultados `void` sin ordenar sus fuentes; `AFTER` establece dependencia. Prueba dos salidas independientes y otra pareja donde la segunda consume lo escrito por la primera. El primer caso admite varios órdenes; el segundo exige observar el valor actualizado.

## GEMM irregular: tres máscaras diferentes

Propón M=17, N=19, K=23 y tiles de lado 16. Hay cuatro tiles de salida y dos pasos por K. En el segundo paso solo siete posiciones de K son válidas. En la última fila de tiles existe una fila de salida; en la última columna, tres columnas.

**Solución.** Cada carga de A y B comprueba sus propias coordenadas y aporta cero fuera de dominio. La acumulación utiliza el tile con esos ceros; la escritura final comprueba M y N. Debe existir la sincronización requerida antes de consumir el tile y antes de sobrescribirlo en la siguiente iteración.

Reutilizar la misma condición para cargar A, cargar B y escribir C suele ser incorrecto porque sus límites difieren. Predice un elemento de borde con la referencia escalar y compara después toda la salida. Haz que el compilador conserve un dump del mapa de índices para esta prueba.

## Tensor cores y selección por precondiciones

Una ruta matricial especializada exige tipos de entrada y acumulación, tiles, layouts, alineación y requisitos de participación de hilos. Los detalles dependen de la interfaz y arquitectura y deben consultarse antes de integrar el kernel. La selección comprueba esas condiciones y dispone de una alternativa correcta para los casos restantes. [@cuda-guide]

Como entrega, conserva el kernel general y añade una variante para un dominio explícito. Si necesita conversión, padding o reordenación, registra esos costes y su error numérico. No atribuyas una mejora exclusivamente a la instrucción cuando también cambia la precisión.

La demostración WMMA del libro principal no integra automáticamente tensor cores en el generador ni en el modelo. Acredita esa integración mostrando que una operación compilada selecciona la variante esperada y produce el resultado correcto.

# Semanas 9–11: medir el compilador dentro del modelo {#cuda:modelos-syllabus}

## Del tiempo de kernel al tiempo de entrenamiento

La ejecución residente mide trabajo con datos disponibles en dispositivo. La medida extremo a extremo incorpora preparación y transferencias pertinentes. El paso de entrenamiento incluye forward, backward y optimizador. Conserva esas fronteras separadas.

Si un kernel ocupa el 20 por ciento del paso y se acelera por dos sin cambiar nada más, el nuevo coste relativo es `0.8+0.2/2=0.9`: la aceleración ideal del paso es aproximadamente 1,11. La mejora del kernel no implica duplicar tokens por segundo. Este cálculo es una predicción; el experimento puede mostrar otras interacciones.

## La meta de competir con PyTorch

El syllabus fija competitividad CUDA/HIP frente a PyTorch como objetivo. Compara mismas formas, entradas, dtypes, precisión de acumulación y función matemática. Declara versión, dispositivo, modo de ejecución y sincronización. Si una ruta incluye transferencias y otra no, informa métricas distintas.

Guarda muestras tras calentamiento, mediana y dispersión. Incluye formas del modelo y casos irregulares que activen kernels alternativos. Omitir las formas que fallan no establece cobertura. Los informes publicados de este cuaderno no contienen esa campaña: será evidencia adicional del estudiante.

## Un paso y su memoria

Enumera pesos, activaciones, gradientes, estados del optimizador y temporales. Anota qué kernels los producen y consumen. Un tensor que el forward ya no necesita puede seguir siendo necesario para backward.

Al introducir recomputación o fusión, compara pérdida, gradientes y actualización con el plan anterior antes de medir. Para reanudar, conserva todos los estados necesarios del entrenamiento; descargar solo pesos no cumple ese contrato.

## Proyecto y aceptación

Una ampliación acotada puede implementar reducción residente, upcasting, selección de GEMM o reutilización de temporales. El informe contiene hipótesis, precondiciones, pruebas, medidas y casos desfavorables.

| Hito | Evidencia |
|---|---|
| Semana 7 | Mismo grafo correcto en CPU y GPU; índices y vida de buffers revisados |
| Semana 8 | Ruta optimizada con dominio explícito y alternativa correcta |
| Semanas 9–10 | Paso del modelo con valores, gradientes, memoria y tiempo comprobados |
| Semana 11 en adelante | Contribución activable y desactivable; reproducción y defensa |

Los kernels manuales conservan su papel como referencias pequeñas. La entrega muestra además el programa generado, su plan y la relación entre cada decisión del compilador y la ejecución.
