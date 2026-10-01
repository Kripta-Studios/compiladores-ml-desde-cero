# GPU: repartir posiciones entre miles de tareas {#ch:gpu}

## Una GPU no adivina el paralelismo

El compilador debe decidir qué trabajo corresponde a cada hilo, qué datos lee y dónde escribe. La GPU ejecuta esa organización bajo su modelo de hardware. Una fórmula con muchas operaciones no se vuelve paralela automáticamente por enviarla a un dispositivo.

Empezamos con una suma elemento a elemento porque sus salidas son independientes. Un hilo calcula una posición. Más adelante varios hilos cooperarán para una salida o un bloque de salidas. La dificultad crece cuando comparten memoria o dependen de resultados parciales.

## Hilo, bloque y grid

Un **hilo** es una instancia del programa de kernel con índices propios. Un **bloque** agrupa hilos que pueden cooperar con recursos y sincronizaciones de bloque. Una **grid** reúne los bloques de un lanzamiento. CUDA proporciona identificadores para cada nivel. [@cuda]

Para N=10 y bloques de cuatro hilos, lanzamos tres bloques. En el bloque b, el hilo t calcula `q=b*4+t`. Los primeros diez q son válidos; los dos últimos deben abstenerse de acceder a la salida. Esta es exactamente la división de rangos que ya hicimos en CPU.

```cuda
__global__ void sumar(const float* x, float* y, long long n) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q < n) y[q] = x[q] + 2.0f;
}
```

El keyword `__global__` marca una función lanzable como kernel. No indica que todas sus variables vivan en memoria global. Los nombres de CUDA requieren estudiar el contexto en que aparecen.

## La guardia de la cola

Si un hilo tiene q fuera de rango, no debe leer x[q] ni escribir y[q]. Poner la condición alrededor de ambas operaciones es la solución inicial. La expresión del índice utiliza un tipo suficientemente ancho antes de multiplicar índices, evitando que el producto se haga primero en una anchura menor.

El número de bloques se calcula con división redondeada hacia arriba: `(N+B-1)//B` para B positivo. Para N=0, el runtime puede no lanzar nada. Lanzar una grid vacía o permitir tamaños negativos debe manejarse mediante un contrato explícito del runtime.

## Mapeo bidimensional

Para una matriz, podemos asignar una coordenada de fila y otra de columna a partir de bloque e hilo. Un bloque de 16×16 contiene 256 hilos lógicos y cubre una región de 16×16 salidas. Las colas de ambas dimensiones requieren máscaras.

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
if (row < M && col < N) C[row*N+col] = A[row*N+col] + B[row*N+col];
```

El producto de dimensiones del bloque debe respetar los límites del dispositivo. No copies un bloque de 1024×1024 porque la matriz tenga ese tamaño: bloque de ejecución y tamaño total del problema no son lo mismo.

## SIMT y grupos de ejecución

En NVIDIA, los hilos se ejecutan en grupos llamados warps. Comparten una organización de ejecución que hace relevante si siguen el mismo camino de instrucciones. AMD utiliza wavefronts con características que dependen de arquitectura y modo. No se debe imponer un tamaño único de grupo a todos los backends. [@cuda;@hip]

Para un principiante, la intuición útil es que instrucciones similares sobre posiciones vecinas suelen aprovechar mejor el hardware que decisiones completamente divergentes. Pero la semántica no autoriza a omitir sincronizaciones por suponer que todos los hilos avanzan exactamente juntos.

## Divergencia

Si unos hilos toman una rama y otros otra, el hardware debe gestionar los caminos. Una condición de borde suele ser inevitable y afecta solo a ciertos grupos. Una condición irregular basada en datos puede producir más divergencia.

Divergencia no significa automáticamente incorrección. El problema de corrección aparece cuando una operación colectiva o una barrera exige participación y algunos hilos no llegan. El rendimiento y la validez de la sincronización son cuestiones distintas.

## Registros y memoria local

Las variables privadas de un hilo pueden residir en registros si el compilador y los recursos lo permiten. Si no, pueden almacenarse en memoria asociada a ese hilo. En terminología CUDA, «local memory» no significa necesariamente una memoria física cercana y rápida como shared memory.

Esta terminología confunde porque distintos niveles del stack usan «local» con sentidos diferentes. En un informe conviene escribir si hablamos de almacenamiento privado, memoria compartida del bloque, caché o memoria global del dispositivo.

## Ocupación

La ocupación se relaciona con cuántos grupos de ejecución pueden estar activos respecto a una capacidad del dispositivo. Registros por hilo, memoria compartida por bloque y tamaño del bloque pueden limitar esa residencia.

Una ocupación mayor no garantiza un kernel más rápido. Un microkernel con más registros puede reutilizar mejor datos y ganar aunque reduzca el número de grupos residentes. La ocupación es una señal para analizar latencia y recursos, no un objetivo que deba maximizarse aislado.

## Memoria del host y del dispositivo

Una CPU y una GPU pueden tener espacios de memoria y mecanismos de transferencia distintos. En el runtime inicial reservamos buffers de dispositivo y copiamos explícitamente entradas y salidas. No suponemos que un puntero de NumPy sea directamente utilizable por cualquier kernel GPU.

La memoria unificada y los mecanismos de acceso compartido ofrecen otras posibilidades, pero también tienen costes y restricciones. Empezar con transferencias explícitas permite observar el movimiento de datos y no confundirlo con cálculo.

## Primer hito GPU

El objetivo inicial no es alcanzar el techo del hardware. Es ejecutar una operación simple con tipos, tamaños, máscaras, errores y sincronización correctos. Después se incorporan reducciones y GEMM de referencia. Solo entonces tiene sentido optimizar accesos y cooperación.

Lumbre proporciona una ruta CUDA y una HIP que generan kernels a partir del mismo IR. La revisión del 1 de octubre ejecuta CUDA en una RTX 5070 Ti Laptop bajo WSL2, con toolkit 12.9 y destino `sm_120`; sus evidencias se guardan en `code/reports/validation_20261001/`. HIP sigue pendiente de hardware compatible. La validación en el dispositivo del estudiante sigue siendo una tarea concreta del laboratorio.

## CUDA frente a HIP

HIP ofrece un modelo y APIs cercanos a CUDA para facilitar portabilidad, especialmente hacia AMD. La proximidad sintáctica permite compartir parte del renderer y del runtime, pero no garantiza rendimiento portable ni disponibilidad idéntica de instrucciones. [@hip]

Una suma independiente puede portarse con cambios pequeños. Una GEMM que depende de instrucciones matriciales, layouts de fragmentos, barreras y tamaños de grupo necesita un análisis específico. La portabilidad tiene varios niveles: fuente, compilación, corrección y rendimiento.

::: practica Cobertura de una grid
Una matriz tiene 33 filas y 17 columnas. Usas bloques de 16×16. ¿Cuántos bloques lanzas en cada eje? ¿Cuántos hilos del bloque situado al final de ambos ejes escriben una salida válida?
:::

**Solución.** Hacen falta tres bloques en filas y dos en columnas. El último bloque empieza en fila 32 y columna 16. Solo existe una fila y una columna válidas en esa región: un hilo escribe una salida. Los demás deben respetar las máscaras. Esto no significa que deban saltarse barreras colectivas si el kernel las utiliza.

::: comprueba Antes de continuar
Calcula el índice de un hilo, el tamaño de grid y una máscara de borde. Explica por qué una GPU necesita un plan de trabajo y por qué portabilidad de sintaxis no equivale a portabilidad de rendimiento.
:::

# Call y runtime: hacer que el código llegue al dispositivo {#ch:runtime}

## Un kernel compilado todavía no se ha ejecutado

Entre la fórmula y un resultado GPU hay varias etapas: generar fuente o IR, compilar para un target, cargar un módulo, reservar memoria, transferir datos, resolver una función, preparar argumentos, lanzar y sincronizar. Un fallo en cualquiera puede impedir el resultado aunque la aritmética del kernel sea correcta.

Una operación **call** representa una invocación con un contrato. Puede llamar a un kernel generado, a una biblioteca especializada o a una función del runtime. El compilador necesita conocer argumentos, salidas, workspace, efectos y condiciones de validez.

## Un contrato de llamada

Para GEMM, la llamada debe especificar dimensiones, tipos, layouts, alineación, dispositivo y si entradas y salida pueden solaparse. Para una operación asíncrona, también necesita definir qué evento indica finalización. Para una biblioteca, puede exigir un workspace y una configuración que dependan de la forma.

La firma «tres punteros» no contiene por sí sola toda esa información. El compilador puede mantener metadatos separados y validar antes del lanzamiento. Una llamada especializada no es una excepción a la corrección: es una frontera donde el contrato debe resultar especialmente claro.

## Cargar un módulo y resolver una función

En el backend CPU, una biblioteca compartida se carga mediante `ctypes`. En GPU, una ruta puede utilizar APIs de módulo y funciones del driver, o compilar una biblioteca anfitriona que incluya lanzamientos. Lumbre utiliza esta segunda vía como solución didáctica.

El código generado contiene wrappers de reserva, copia, sincronización y liberación. El runtime Python declara sus firmas y comprueba códigos de retorno. La arquitectura concreta se indica para CUDA mediante `LUMBRE_CUDA_ARCH`; no se elige una capacidad arbitraria por el nombre comercial del ordenador.

## Comprobar el target

El nombre de una familia de GPU no identifica todas sus instrucciones. La tabla de NVIDIA distingue, por ejemplo, las capacidades de H100/H200, B200/B300 y GeForce RTX 50. Una tarjeta Blackwell de consumo no debe tratarse como si fuera idéntica a una GPU Blackwell de centro de datos. [@cuda-gpus]

La práctica correcta consulta el dispositivo real y el soporte del compilador instalado. Las variantes de target con sufijos y características específicas tienen reglas de compatibilidad propias. Compilar para un target avanzado no añade al silicio instrucciones que no posee.

## Streams

Un stream organiza una secuencia de trabajo en un dispositivo. Operaciones dentro de una secuencia pueden respetar un orden, mientras varias secuencias permiten potencialmente solapamiento. Para coordinar streams se utilizan eventos y dependencias explícitas.

El solapamiento no es gratuito: requiere recursos disponibles y transferencias o kernels que realmente puedan ejecutarse a la vez. Dos tareas que compiten por el mismo ancho de banda pueden no mejorar al superponerse. La ausencia de sincronización global puede revelar carreras si los buffers se comparten sin dependencias.

## Eventos

Un evento marca un punto de progreso. Otro stream o el host puede esperar a que se alcance. Los eventos también pueden servir para medir intervalos en el dispositivo, bajo las reglas de la API.

Un evento debe representar exactamente el trabajo del que depende el consumidor. Esperar a un evento grabado antes de una copia no garantiza que la copia haya terminado. Los diagramas de runtime deben indicar orden temporal, no solo conexiones visuales entre nombres.

```diagram
Copiar entradas | Lanzar kernels en orden | Esperar y leer salidas
```

Esta secuencia es el punto de partida. Una versión avanzada puede superponer la copia del siguiente batch con el cálculo del actual, manteniendo buffers separados y dependencias correctas.

## Errores asíncronos

Un error de lanzamiento puede detectarse al encolar. Un acceso inválido dentro del kernel puede aparecer al sincronizar más tarde. Por eso un runtime no debe comprobar únicamente la llamada de lanzamiento y asumir que la ejecución fue correcta.

En Lumbre se comprueba el estado de lanzamiento y se sincroniza al terminar `run`. Para depurar, conviene reducir el caso a una forma pequeña y conservar la fuente generada. Después se utiliza la herramienta de diagnóstico disponible para el dispositivo, sin afirmar una validación que no se ha ejecutado.

## Vida de buffers y módulos

Un buffer debe permanecer vivo hasta que todos sus consumidores terminen. Un módulo no debe descargarse mientras existan llamadas que dependan de él. Un objeto Python destruido por el recolector no debería liberar memoria que todavía utiliza un stream asíncrono.

La gestión mediante context manager ayuda a cerrar recursos de forma explícita, pero no elimina la necesidad de sincronización. La ruta del curso prioriza un ciclo de vida sencillo y visible. Un runtime de producción puede utilizar pools y referencias asociadas a eventos.

## Caché del JIT y capacidades

Una caché GPU debe incluir arquitectura, opciones, versión del compilador y otras dependencias relevantes. Algunas bibliotecas generan variantes para formas y tipos concretos. Reutilizar un módulo con una clave incompleta puede producir un fallo o un resultado sutilmente incorrecto.

DeepJIT, publicado por DeepSeek, aborda compilación, caché, carga y lanzamiento para CUDA y Ascend desde una interfaz C++20. En el estado consultado, ciertas APIs Python y funciones de precalentamiento figuran como trabajo pendiente; no se utilizan en el libro como si fueran interfaces ya disponibles. [@deepjit]

## Bibliotecas externas y fallback

Un compilador puede escoger una implementación de biblioteca cuando resulte adecuada y un kernel propio en otros casos. Debe registrar esa decisión. Una medida de un backend que llama a cuBLAS o hipBLASLt no se presenta como si todo su código aritmético hubiera sido generado desde cero por el estudiante.

El fallback es valioso para cobertura, pero debe ser observable en logs y métricas. Un programa que mezcla kernels propios y bibliotecas puede ser una solución excelente; ocultar esa mezcla impide interpretar sus resultados.

## Un pasaporte de llamada

Registra nombre del operador, hash del código o biblioteca, dimensiones, tipo, layout, argumentos de lanzamiento, memoria compartida, workspace, stream y política numérica. Ese pasaporte permite reproducir un fallo y comparar implementaciones sin depender de nombres ambiguos.

El proyecto final de «pasaportes de kernels» extiende esta idea a verificación de precondiciones y selección automática. La originalidad no exige inventar una nueva instrucción de GPU: puede estar en unir contratos, evidencia y observabilidad de forma útil.

::: practica Una carrera entre copia y cálculo
Copias nuevos pesos en un stream y lanzas un kernel que los usa en otro, sin evento. A veces funciona. ¿Qué falta? ¿Basta con que ambas llamadas aparezcan en ese orden en Python?
:::

**Solución.** Falta una dependencia de finalización entre la copia y el consumidor. El orden de llamadas del host no establece por sí solo todas las dependencias entre streams distintos. Debe usarse una espera o una organización de stream que garantice el orden requerido. Que algunas ejecuciones funcionen no demuestra ausencia de carrera.

::: comprueba Antes de continuar
Enumera las etapas entre fuente y resultado. Distingue error de lanzamiento y error de ejecución, y explica qué información pertenece a la clave de caché y qué información pertenece a los datos de una llamada.
:::

# Coalescing, shared memory y sincronización {#ch:gpu_memoria}

## Una fila de hilos y una fila de datos

Cuando hilos vecinos acceden a posiciones vecinas, el hardware puede agrupar solicitudes de memoria de forma eficiente. A esta propiedad se la llama coalescing en el contexto GPU. El patrón exacto de transacciones depende de arquitectura, alineación y tamaño de acceso. [@cuda]

Si cada hilo salta una gran distancia, puede requerirse mover muchas regiones para utilizar pocos valores de cada una. La fórmula matemática no cambia, pero sí el coste de suministrar datos. La relación entre layout y mapeo de hilos es por tanto una decisión central del compilador.

## Elegir qué eje sigue al hilo x

En un array por filas, las columnas contiguas suelen asignarse a hilos vecinos del eje x. Si asignamos filas a esos hilos, cada uno puede saltar un stride grande. En otros layouts o patrones de reutilización puede convenir otra organización.

El criterio no es «x siempre representa columnas» por una ley del lenguaje. Es «qué mapeo produce accesos útiles para la operación y el hardware». Las APIs ofrecen índices; el compilador les da significado.

## Shared memory como mesa de trabajo del bloque

La memoria compartida de bloque permite que varios hilos carguen datos, los reutilicen y cooperen. No se rellena automáticamente: el programa debe decidir quién carga qué y cuándo los demás pueden leerlo.

En GEMM, un bloque puede cargar un tile de A y otro de B, sincronizar, calcular productos y repetir con el siguiente tramo de K. La reutilización dentro del bloque reduce accesos repetidos a memoria global.

## Dos barreras por iteración

Después de cargar un tile, todos los participantes deben ver datos completos antes de calcular. Después de calcular, nadie debe sobrescribir ese almacenamiento con el siguiente tile mientras otro hilo todavía lee el anterior. Por eso un diseño sencillo utiliza una barrera tras la carga y otra antes de reutilizar el buffer.

```cuda
cargar_tile_A_y_B();
__syncthreads();
acumular_productos_del_tile();
__syncthreads();
```

El fragmento muestra el orden, no una función ejecutable por sí solo. El kernel completo generado por Lumbre contiene ambas barreras y cargas enmascaradas que escriben cero en posiciones fuera de rango.

## La trampa del retorno temprano

En una suma sin cooperación, un hilo fuera de rango puede salir. En un kernel con barreras de bloque, hacer que solo algunos hilos retornen antes de una barrera puede violar sus requisitos. Los hilos de borde pueden no producir una salida válida y aun así tener que participar cargando ceros y alcanzando las barreras.

::: cuidado Una máscara de salida no es una máscara de participación
No encierres todo el cuerpo de una GEMM cooperativa en `if (row<M && col<N)` si dentro hay barreras que deben alcanzar todos los hilos del bloque. Enmascara accesos y escritura final; conserva la participación requerida en las operaciones colectivas.
:::

## GEMM compartida de 16×16

El backend bloqueado GPU de Lumbre utiliza dos arrays compartidos de 16×16. Cada hilo carga una posición de A y una de B. Tras sincronizar, acumula dieciséis productos del tile en su salida. Después sincroniza de nuevo y avanza K.

La implementación es pedagógica: cada hilo mantiene un acumulador escalar FP32. No utiliza tensor cores ni una canalización asíncrona avanzada. Permite estudiar coalescing, reutilización y colas con un código que se corresponde directamente con los capítulos anteriores.

## Bancos de shared memory

La memoria compartida se organiza en bancos. Ciertos patrones de acceso pueden hacer que varios hilos compitan por un mismo banco, salvo casos soportados como difusión de un mismo dato. Padding y swizzling pueden cambiar el mapeo de direcciones para reducir conflictos.

El patrón óptimo depende del tamaño de elemento y de las instrucciones que consumen los datos. Una disposición que parece ideal para cargas escalares puede no ser la que necesita una instrucción matricial. Por eso los layouts de bibliotecas como CuTe se tratan como funciones de coordenadas a ubicaciones, no como simples nombres «row-major» y «column-major».

## Swizzling

Un swizzle reorganiza direcciones mediante una transformación estructurada, a menudo utilizando combinaciones de bits de índices. La intención es distribuir accesos entre bancos o adaptar los datos a un consumidor. No cambia qué valor corresponde a cada coordenada lógica si el mapa y su uso son consistentes.

Para comenzar, dibuja una tabla pequeña de coordenadas y direcciones. Comprueba que no hay colisiones indebidas y que la transformación es invertible sobre el dominio requerido. Después evalúa el patrón de bancos. No copies una fórmula XOR de otro kernel sin entender sus dimensiones y alineaciones.

## Registros compartidos por una operación colectiva

Algunas instrucciones de grupo distribuyen fragmentos entre carriles. Un fragmento lógico de matriz no reside necesariamente como una submatriz intuitiva en cada hilo. El contrato de carga, cómputo y almacenamiento define cómo cooperan.

No uses el contenido de un fragmento de otro target suponiendo que el reparto interno es idéntico. Cuando una API documenta un layout opaco, debe tratarse como opaco. Un backend de bajo nivel que maneja registros explícitos necesita especificaciones más detalladas y pruebas por arquitectura.

## Double buffering

Mientras se calcula con un tile, se puede preparar el siguiente en otra región. Esto intenta solapar movimiento y cálculo. Se necesitan dos buffers y un protocolo que indique cuál está listo y cuál puede reutilizarse.

Un error típico confunde «la copia ha sido emitida» con «la copia ha terminado». Otro sobrescribe un buffer que todavía consume una instrucción asíncrona. El número de stages no es una mejora aislada: cambia recursos y dependencias.

## Profundidad de pipeline

Más stages pueden ocultar latencia, pero consumen más memoria compartida y estado de control. Eso puede reducir residencia de bloques. Un pipeline profundo puede perder frente a uno corto si el kernel ya está limitado por otro recurso.

El estudio de programas tile de 2026 incluye fallos relacionados con sincronización y vida útil. Su utilidad para el curso es recordar que un DSL no elimina automáticamente todas las obligaciones de cooperación; la evaluación debe cubrir el protocolo, no solo una salida fácil. [@tile-bugs]

## Pruebas de concurrencia

Repite kernels con formas de borde, pocos y muchos bloques, diferentes tamaños de tile y datos cambiantes. Utiliza herramientas de detección de carreras y accesos cuando estén disponibles. Conserva un caso reducido para cada fallo.

Una prueba que pasa una vez no demuestra ausencia de carrera. Aun así, pruebas de estrés y comprobaciones estructurales son evidencia útil. Deben complementarse con el razonamiento de productores, consumidores y barreras.

::: practica Encontrar la barrera que falta
Un bloque carga un tile, sincroniza, calcula y comienza inmediatamente a cargar el siguiente en el mismo buffer. ¿Qué carrera puede existir aunque todos hayan superado la primera barrera?
:::

**Solución.** Un hilo rápido puede sobrescribir datos del tile anterior mientras uno lento todavía los lee durante el cálculo. La primera barrera protege la disponibilidad inicial, no la reutilización posterior. Hace falta una segunda sincronización o un protocolo equivalente que garantice finalización de todos los consumidores.

::: comprueba Antes de continuar
Explica quién produce y quién consume cada región compartida. Distingue máscara de acceso, máscara de salida y participación colectiva. Después justifica las dos barreras del kernel GEMM de referencia.
:::
