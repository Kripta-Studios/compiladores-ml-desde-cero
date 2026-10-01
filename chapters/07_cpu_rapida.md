# La jerarquía de memoria: por qué multiplicar no es todo el trabajo {#ch:jerarquia}

## La cocina y el almacén

Imagina que cada vez que cortas una verdura debes caminar hasta un almacén situado a cien metros. Un cuchillo diez veces más rápido apenas resolvería el problema. Primero necesitas organizar qué ingredientes mantienes cerca y cuántas veces los reutilizas.

Una CPU tiene una jerarquía de almacenamiento: registros muy próximos a las unidades de ejecución, varios niveles de caché y memoria principal. La analogía de distancias no representa literalmente sus tiempos, pero ayuda a separar dos recursos: capacidad de calcular y capacidad de suministrar datos.

## Latencia y ancho de banda

La **latencia** es cuánto tarda en completarse una petición. El **ancho de banda** es cuánto volumen puede trasladarse por unidad de tiempo cuando existe suficiente trabajo en curso. Un sistema puede tener alta latencia y alto ancho de banda: una petición aislada tarda, pero muchas peticiones simultáneas entregan gran cantidad de datos.

Un kernel de aprendizaje automático puede estar limitado por cualquiera de ellos, por dependencias, por ocupación o por costes de lanzamiento. «La memoria es lenta» es una explicación insuficiente si no identificamos qué comportamiento concreto domina.

## Caché y localidad

Una caché conserva datos que probablemente volveremos a usar. La **localidad temporal** aparece cuando reutilizamos pronto un valor. La **localidad espacial** aparece cuando accedemos a posiciones cercanas. Los arrays contiguos ayudan a explotar esta segunda propiedad.

En una suma de vectores, cada entrada suele utilizarse una vez. En GEMM, un elemento de A puede contribuir a varias columnas y uno de B a varias filas. Esa reutilización potencial explica por qué GEMM puede beneficiarse tanto del blocking.

No es suficiente que los datos sean reutilizables matemáticamente. El orden del programa debe acercar esos usos en el tiempo y mantener un conjunto de trabajo que quepa en el nivel de memoria relevante.

## Líneas de caché y accesos dispersos

Las cachés mueven datos en unidades que contienen varios bytes consecutivos. Leer un único valor de cada región muy distante puede desperdiciar gran parte de los datos trasladados. Leer posiciones contiguas aprovecha mejor cada transferencia.

El tamaño exacto de líneas y cachés depende del procesador. No debemos usar cifras de un modelo como si describieran todos los portátiles. En un informe se registra la CPU y, cuando sea necesario, sus propiedades consultadas mediante herramientas del sistema.

## Intensidad aritmética

La intensidad aritmética compara operaciones de cálculo con bytes movidos. Una suma de vectores FP32 realiza aproximadamente una operación por dos lecturas y una escritura, unos doce bytes solicitados por elemento. Su intensidad es baja.

Una GEMM de tamaño grande puede reutilizar datos muchas veces, aumentando operaciones por byte trasladado desde memoria lejana. La intensidad depende de qué nivel de la jerarquía estamos midiendo y de la implementación, no solo de la fórmula del operador.

::: traduccion Un cociente con unidades
```math
I=\frac{\text{operaciones de coma flotante}}{\text{bytes transferidos}}.
```
Si I vale 0,1 FLOP/byte y el ancho de banda sostenible es 100 GB/s, un techo idealizado asociado a esa transferencia es 10 GFLOP/s. Son cifras inventadas para practicar unidades, no especificaciones de una máquina de esta entrega.
:::

## El modelo roofline

Roofline compara un techo de capacidad de cómputo con otro impuesto por ancho de banda e intensidad. La cota idealizada de rendimiento es el mínimo entre ambos. El modelo ayuda a decidir si tiene sentido reducir operaciones o, antes, reducir tráfico. [@roofline]

```math
P\leq \min(P_{\text{cómputo}}, B_{\text{memoria}}\,I).
```

Una cota no es una predicción exacta. Ignora o simplifica latencias, dependencias, instrucciones, cachés, paralelismo disponible y comportamiento del compilador. Si un kernel queda muy por debajo de ambos techos, el modelo no queda refutado: falta explicar qué otros límites aparecen.

## Contar tráfico con cuidado

Leer A y B y escribir C una vez da una estimación mínima para GEMM, pero una implementación ingenua puede leer los mismos valores muchas veces desde distintos niveles. Algunas escrituras provocan tráfico adicional. El resultado de un contador de hardware tampoco coincide necesariamente con el número de accesos del código fuente.

Un informe debe distinguir bytes lógicos solicitados, bytes estimados por un modelo y bytes observados por contadores. Mezclarlos puede producir intensidades imposibles o comparaciones engañosas.

## Blocking: trabajar por porciones que se reutilizan

En lugar de calcular una salida completa y pasar a la siguiente, podemos calcular un bloque de salidas. Un pequeño conjunto de A y B alimenta varias multiplicaciones antes de abandonar la caché o los registros.

Para un bloque de cuatro filas y ocho columnas, un valor de A puede multiplicarse por ocho valores de B; una fila parcial de B puede utilizarse para cuatro filas de salida. El trabajo se organiza para conservar esos datos cerca de las unidades de cálculo.

La elección de tamaños depende de registros, caché, SIMD, número de hilos y dimensiones reales. Un tile mayor no siempre es mejor: puede expulsar datos que se querían reutilizar o producir demasiados acumuladores.

## Prefetch y predicción de accesos

Algunos procesadores intentan anticipar patrones de acceso. Un recorrido regular puede facilitar esa anticipación. Los programas también pueden incluir instrucciones de prefetch, pero usarlas sin medir puede empeorar el tráfico o traer datos demasiado pronto.

Antes de añadir prefetch manual, consigue un recorrido claro, un conjunto de trabajo razonable y un benchmark estable. La complejidad de bajo nivel no es un indicador de calidad. Un kernel sencillo que vectoriza bien puede superar uno lleno de instrucciones explícitas mal organizadas.

## NUMA y afinidad

En sistemas con varios dominios de memoria, el lugar donde se reserva o se toca primero un buffer puede influir en qué procesador accede a él con menor coste. Esto se conoce como NUMA. También importa dónde se ejecutan los hilos y si migran entre núcleos.

No todos los equipos del curso tienen varios nodos NUMA, pero el concepto explica por qué un benchmark de servidor puede cambiar al modificar afinidad. Registra número de hilos, política de colocación y configuración; no atribuyas automáticamente toda variación al compilador.

## Una investigación guiada

Ejecuta una suma de vectores con tamaños que aumenten por potencias de dos. Para cada tamaño mide varias repeticiones calientes y una política separada de datos fríos. Convierte tiempo a bytes lógicos por segundo y anota dónde cambia el comportamiento.

No etiquetes cada cambio como «L1», «L2» o «RAM» únicamente por su forma visual. Utiliza información del procesador y, si está disponible, contadores. Una curva sugiere hipótesis; no identifica por sí sola un componente físico.

::: practica Decidir qué optimizar
Un kernel lee dos arrays FP32, suma sus elementos y escribe uno. Has eliminado una operación entera de cálculo de índice, pero el tiempo apenas cambia. Propón una explicación coherente con roofline y otra que no dependa de ancho de banda.
:::

**Solución.** Puede estar limitado por tráfico de memoria, de modo que ahorrar una operación pequeña no cambia el techo dominante. También puede ser tan pequeño que domine la llamada o el lanzamiento. Hay que medir varios tamaños y separar costes antes de concluir cuál explicación corresponde.

::: comprueba Antes de continuar
Debes manejar unidades de FLOP, byte y segundo; distinguir localidad temporal y espacial; y explicar por qué la reutilización matemática necesita un orden de ejecución adecuado para convertirse en reutilización física.
:::

# Upcasting, unrolling y SIMD: varias casillas por instrucción {#ch:simd}

## Ocho lápices que avanzan juntos

Imagina que puedes escribir ocho resultados de la misma clase a la vez, utilizando ocho parejas de entradas. SIMD significa «una instrucción, múltiples datos». La operación es común, pero cada carril trabaja con valores distintos.

No confundas SIMD con varios núcleos independientes. Un núcleo puede ejecutar instrucciones vectoriales, y varios núcleos pueden hacerlo a la vez. Tampoco confundas SIMD con una GPU completa: comparten ideas de paralelismo de datos, pero sus modelos de ejecución y memoria tienen diferencias importantes.

## Del bucle escalar al vector

Un bucle escalar produce una casilla por vuelta. Un bucle vectorizado puede procesar grupos de cuatro u ocho, según tipo e ISA. Para N que no sea múltiplo del ancho, hace falta tratar la cola mediante máscaras o un bucle escalar.

```c
int i = 0;
for (; i + 3 < N; i += 4) {
    y[i]   = x[i]   + 2.0f;
    y[i+1] = x[i+1] + 2.0f;
    y[i+2] = x[i+2] + 2.0f;
    y[i+3] = x[i+3] + 2.0f;
}
for (; i < N; i++) y[i] = x[i] + 2.0f;
```

El código muestra un desenrollado de cuatro posiciones. No garantiza por sí solo que el compilador produzca una instrucción SIMD concreta. La generación real depende del compilador, las opciones, el target y los obstáculos de aliasing o dependencias.

## Unrolling no es necesariamente vectorización

Desenrollar copia el cuerpo del bucle para varias iteraciones. Puede reducir control de bucle y exponer independencia de instrucciones. Vectorizar agrupa operaciones en instrucciones de múltiples carriles. Podemos desenrollar sin vectorizar, vectorizar sin un desenrollado visible en el fuente o combinar ambas técnicas.

Un exceso de desenrollado aumenta tamaño de código y valores vivos. El compilador puede necesitar más registros o perder eficacia de la caché de instrucciones. Por eso la cantidad de copias es un parámetro de scheduling, no una mejora automática.

## Qué significa upcasting en este contexto

En el vocabulario de ciertos compiladores de tensores, upcasting puede referirse a convertir parte de un eje iterativo en un conjunto de valores procesados dentro de una misma instancia, a menudo representados como carriles o vectores. No debe confundirse con promover FP16 a FP32, que es otro uso habitual de la palabra en programación.

En el commit estudiado de tinygrad aparecen ejes UPCAST y UNROLL y transformaciones que expanden rangos y vectores. El sentido concreto se entiende leyendo el pase y sus invariantes, no trasladando una definición de «casting de tipos» sin mirar el código. [@tinygrad-codegen]

## Acumuladores múltiples

Una reducción secuencial tiene una dependencia: cada suma necesita el acumulador anterior. Utilizar varios acumuladores independientes puede exponer más paralelismo. Por ejemplo, sumar posiciones de cuatro clases de resto por separado y combinar al final.

```c
float s0=0, s1=0, s2=0, s3=0;
int i=0;
for (; i+3<N; i+=4) {
    s0 += x[i];   s1 += x[i+1];
    s2 += x[i+2]; s3 += x[i+3];
}
float s = (s0+s1)+(s2+s3);
for (; i<N; i++) s += x[i];
```

El orden numérico ya no coincide con la cadena original. No ocultes ese cambio bajo la palabra «vectorización». Debe estar permitido por el contrato y probado con tolerancias apropiadas.

## Alineación

Una dirección alineada cumple un múltiplo requerido o conveniente para una instrucción. Algunas instrucciones admiten cargas no alineadas; otras exigen condiciones más estrictas. Incluso cuando una carga no alineada es legal, cruzar ciertas fronteras puede cambiar el coste.

No añadas una promesa de alineación al compilador sin garantizar que el allocator y los offsets la cumplen. En Lumbre la arena y sus regiones tienen una política explícita; las vistas arbitrarias pueden romper alineaciones de subregiones, por lo que necesitarían análisis adicional.

## Aliasing como obstáculo y como contrato

Si el compilador C no sabe si x e y se solapan, puede generar comprobaciones o evitar ciertos reordenamientos. `restrict` ofrece una promesa que puede ayudar, pero su uso incorrecto viola el contrato del lenguaje.

Una API de kernels puede declarar que entradas y salida no se solapan. Entonces el runtime debe hacer cumplir esa condición o elegir un camino seguro. La solución no consiste en añadir palabras clave hasta que el benchmark mejore, sino en conectar garantías de alto nivel con supuestos de bajo nivel.

## Auto-vectorización y ensamblador

La primera estrategia razonable es escribir bucles simples, compilar para el target y consultar informes de vectorización. Después inspecciona ensamblador de un caso pequeño. Busca cargas vectoriales, operaciones aritméticas y tratamiento de colas, pero no deduzcas rendimiento únicamente contando instrucciones.

```bash
cc -O3 -S -fverbose-asm kernel.c -o kernel.s
```

Las opciones exactas de diagnóstico dependen del compilador. Guarda su versión. Una recomendación de GCC no se convierte automáticamente en una opción válida de Clang ni de un compilador de acelerador.

## Intrinsics y despacho por ISA

Un intrinsic permite expresar una operación ligada a una ISA desde C o C++. Puede ofrecer control útil, pero reduce portabilidad y exige gestionar capacidades. Ejecutar instrucciones no soportadas puede fallar aunque el programa compile en otra máquina.

Una biblioteca portable puede compilar varias versiones y escoger en ejecución según la CPU. El fallback no debe desaparecer. El compilador educativo puede empezar con C escalar y auto-vectorización; el proyecto avanzado añade versiones específicas con guardias y tests sobre el mismo contrato.

## Reducir presión de registros

Cada acumulador ocupa recursos. Un microkernel de cuatro por ocho salidas mantiene treinta y dos acumuladores escalares o su equivalente vectorial. Aumentar a ocho por dieciséis puede multiplicar necesidades y causar spills.

El tamaño óptimo depende de cuántos registros existen, qué otras variables están vivas y cómo el compilador asigna instrucciones. Por eso una búsqueda de tiles debe incluir medición y descartar configuraciones que superen recursos, en lugar de escoger siempre el bloque de mayor superficie.

## Un ejercicio de predicción de colas

Con N=19 y ancho ocho, hay dos grupos completos de ocho y una cola de tres. Una versión enmascarada puede lanzar un tercer grupo con cinco carriles inactivos. Una versión escalar puede procesar tres iteraciones finales. Ambas deben evitar leer más allá de la entrada.

El relleno de la cola depende de la operación. Para suma parcial, cero es identidad. Para máximo, cero sería incorrecto si todos los datos válidos son negativos; hay que usar una identidad compatible, como menos infinito bajo el contrato adoptado.

::: practica Qué cambió realmente
Una optimización transforma una reducción de un acumulador en ocho acumuladores y luego reduce esos ocho. El tiempo mejora y el último bit cambia. ¿Es necesariamente un bug? ¿Qué información necesitas para responder?
:::

**Solución.** Necesitas conocer la equivalencia exigida. Si se requiere reproducibilidad bit a bit con el recorrido original, el cambio incumple el contrato. Si se permite una reducción reordenada dentro de una tolerancia y las pruebas la cumplen, puede ser válida. La mejora de tiempo no decide por sí sola la cuestión.

::: comprueba Antes de continuar
Distingue upcasting de ejes y conversión de tipos, unrolling y SIMD, legalidad de una carga y su eficiencia. Explica por qué cada promesa de alineación o ausencia de alias debe respaldarse desde el runtime.
:::

# GEMM rápida: del triple bucle al microkernel {#ch:gemm}

## Empezar por una salida correcta

La multiplicación matricial es un caso central porque aparece en capas lineales, atención y otros operadores. La primera implementación debe ser transparente: para cada salida, sumar K productos en orden. Esa versión sirve como referencia y como instrumento para localizar errores de índices.

Optimizar antes de disponer de esa referencia complica el diagnóstico. Cuando un microkernel tiene tiles, packing, colas y vectorización, un resultado incorrecto puede tener muchas causas. Una implementación lenta no es un fracaso: es una pieza del sistema de verificación.

## Reutilizar una entrada de A

En el orden `i,j,k`, un valor A[i,k] vuelve a cargarse lógicamente para varias columnas j. Cambiar a un bloque de columnas permite usarlo con varios valores contiguos de B. El microkernel mantiene varias salidas en acumuladores.

Para cuatro filas y ocho columnas, cada paso k carga cuatro valores de A y ocho de B, y actualiza treinta y dos productos. Esta cuenta idealizada ilustra reutilización; el número real de instrucciones y transferencias depende de cómo se implemente y compile.

```tikz
\begin{center}
\begin{tikzpicture}
\node[draw,fill=verde!5,minimum width=1.3cm,minimum height=2.3cm,align=center](a) at (0,0){A\\$4\times K$};
\node[draw,fill=azul!5,minimum width=3.2cm,minimum height=1.1cm,align=center](b)at(3,1.3){B: $K\times8$};
\node[draw,fill=naranja!7,minimum width=3.2cm,minimum height=2.3cm,align=center](c)at(3,-.7){32 acumuladores\\C: $4\times8$};
\draw[flow](a.east)--(c.west);\draw[flow](b.south)--(c.north);
\node[font=\small,align=center]at(3,-2.4){Cada valor de k actualiza\\todo el bloque de salida.};
\end{tikzpicture}
\end{center}
```

## Tres niveles de blocking

Un microkernel organiza registros. Un bloque mayor organiza caché. Una partición entre hilos organiza paralelismo de CPU. Son escalas diferentes. Un tamaño adecuado para registros no tiene por qué llenar una caché, y un bloque de caché no tiene por qué ser la unidad de reparto entre núcleos.

Las bibliotecas de álgebra lineal de alto rendimiento combinan estos niveles con packing, kernels específicos y selección por forma. Nuestro objetivo docente es entender sus razones y construir una versión medible, no afirmar que unas decenas de líneas reemplazan todos sus años de ingeniería.

## Packing

Empaquetar copia un panel de datos a un formato que facilita el microkernel: contigüidad, alineación y un orden de acceso regular. La copia tiene un coste inicial. Compensa cuando el panel se reutiliza lo suficiente o cuando evita accesos muy ineficientes.

El packing debe incluir o tratar las colas de forma segura. Si rellenamos un panel con ceros, el microkernel puede ejecutar un tamaño fijo sin leer fuera de rango, siempre que las posiciones de salida y la semántica numérica se controlen. No debemos escribir fuera de C aunque las entradas estén correctamente rellenadas.

Una matriz pequeña puede perder rendimiento por el coste de packing. Por eso los sistemas maduros suelen tener caminos distintos para formas pequeñas, matrices estrechas y grandes GEMM cuadradas.

## El microkernel incluido en Lumbre

La opción `gemm="blocked"` genera una implementación CPU con bloques de cuatro filas, ocho columnas y tramos de K de hasta sesenta y cuatro. Mantiene un array local de acumuladores, recorre K en orden creciente y trata explícitamente las colas de M y N.

No utiliza BLAS ni delega el producto a NumPy. Tampoco implementa packing completo, despacho por ISA ni paralelismo multicore. Es una mejora original y comprobada frente al backend de referencia del propio compilador, no una afirmación de liderazgo frente a BLIS, oneDNN, MKL u otra biblioteca.

```python
A = param("A", (128, 128))
B = param("B", (128, 128))
C = A @ B
programa = Program([C], gemm="blocked")
```

El scheduler materializa los operandos de la llamada para garantizar buffers contiguos con un contrato sencillo. Esa materialización puede costar cuando una entrada es una vista o expresión. Una versión avanzada puede integrar movimientos o packing en la carga, pero debe medir el coste de extremo a extremo.

## Colas y dimensiones cero

Para M=9 y N=17 aparecen bloques incompletos en ambas direcciones. Cada acceso y cada escritura debe comprobar su posición. K=0 produce una suma vacía: las salidas deben valer cero bajo este contrato.

Un test que solo utiliza 128 por 128 no examina estas fronteras. La batería de la entrega incluye formas no múltiplos del microkernel y un caso K=0. Las pruebas de broadcasting de lote comprueban además que una misma matriz B puede reutilizarse en varios lotes.

## Orden de acumulación

El microkernel conserva el orden creciente de k dentro de cada salida. La política CPU desactiva contracción FMA. Esto facilita comparar con la referencia, aunque un compilador de bajo nivel puede seguir elegir instrucciones y optimizaciones legales dentro de sus reglas.

Para un proyecto SIMD con FMA, el contrato puede permitir diferencias pequeñas. Guarda métricas de error máximo y distribución, no solo una bandera de aprobado. Si un cambio reduce error respecto a FP64 pero rompe una exigencia bit a bit, sigue siendo una modificación de contrato que debe declararse.

## Matrices por lotes y transpuestas

Los ejes de lote se convierten en offsets independientes. Si B tiene lote de tamaño uno y A varios, B puede reutilizarse. Los ejes de matrices siguen teniendo su contrato `(M,K)` y `(K,N)`. No conviene mezclar broadcasting de lote con una transposición implícita de los dos últimos ejes.

Un backend de biblioteca puede aceptar flags de transposición. Un microkernel propio puede preferir materializar o empaquetar. El compilador decide qué convención entrega a cada implementación y conserva una referencia que compruebe el resultado.

## Medidas locales de esta edición

El script `examples/bench_kernels.py` compara los dos caminos CPU con datos residentes, después de calentamiento y sin incluir compilación ni copias de salida. En la ejecución registrada, la mediana para 128×128×128 fue aproximadamente 1,566 ms en la referencia y 0,157 ms en la versión bloqueada. Para 256×256×256 fue aproximadamente 15,247 ms y 1,371 ms, respectivamente.

Estas cifras pertenecen a la CPU virtual del entorno de elaboración y a esa configuración concreta. No predicen tiempos de un portátil ni demuestran superioridad frente a bibliotecas de producción. El interés didáctico es que una transformación visible de organización produjo una mejora medible sin cambiar la operación solicitada.

## Cuándo decir «competitivo»

Para justificar ese término, define competidor, versión, hardware, forma, tipo, layout, número de hilos, calentamiento, modo numérico y alcance temporal. Compara varias formas representativas, no solo la que favorece al microkernel. Incluye costes de packing o conversiones que el usuario real deba pagar.

Una curva de speedup por forma es más informativa que un único máximo. También lo es informar de casos donde se pierde. La meta del temario es aprender a producir y evaluar código competitivo; la palabra no debe sustituir al experimento que la demostraría.

## Progresión de ingeniería

Después de la versión bloqueada, una extensión razonable incorpora packing y un microkernel vectorial. La siguiente añade selección por forma y varios hilos. Más tarde se pueden estudiar prefetch, división de K y layouts específicos. Cada paso debe conservar tests y aislar qué cambio explica el rendimiento.

No cambies a la vez algoritmo, tipo, tolerancia y número de hilos para luego atribuir todo el speedup a «mi compilador». La ablación, es decir, activar y desactivar componentes de forma controlada, permite separar sus efectos.

::: practica Contar la reutilización
Un microkernel produce un bloque de 2×4 salidas. Para cada k, ¿cuántos valores distintos de A y B necesita idealmente y cuántos productos calcula? ¿Por qué esa cuenta no basta para conocer su tiempo?
:::

**Solución.** Necesita dos valores de A y cuatro de B y calcula ocho productos. El tiempo depende además de cargas reales, registros, instrucciones, dependencias, alineación, caché, colas y overhead. La cuenta muestra una oportunidad de reutilización, no un benchmark.

::: comprueba Antes de continuar
Explica registros, caché y reparto de hilos como tres niveles distintos. Identifica dónde aparecen packing, epílogo y colas. Después lee el helper `lm_gemm` generado por Lumbre y relaciona cada bucle con una de esas decisiones.
:::
