# Del compilador al programa CUDA {#cuda:inicio}

Ya has construido grafos, seguido índices y leído kernels en el libro de Lumbre. Ahora escribirás tú el programa que reserva memoria, copia datos, lanza un kernel y comprueba su salida. El objetivo es poder explicar cada acceso y cada espera antes de intentar acelerar una operación.

No necesitas conocer C++ avanzado. Usaremos funciones, punteros, bucles y una pequeña clase que libera memoria al salir de su ámbito. Introduciremos la sintaxis nueva cuando aparezca. Las prácticas trabajan con entradas deterministas y caben en la GPU del equipo de validación.

## El recorrido y sus entregas

| Práctica | Programa | Evidencia que producirás |
|---|---|---|
| Vectores | `01_vectors.cu` | Índices válidos, incluidos tamaños irregulares. |
| Reducción | `02_reduce.cu` | Suma por bloques y combinación en CPU. |
| GEMM | `03_gemm.cu` | Dos kernels con la misma salida y medidas repetidas. |
| Stream | `04_stream.cu` | Copia, kernel y copia ordenados en una cola. |

Los archivos viven en `tutorials/cuda/`. El apéndice incorpora sus fuentes completos. Para trabajar, abre el archivo correspondiente, predice un resultado y ejecuta el programa. Conserva la versión correcta antes de modificarla.

::: idea Conectar con Lumbre
En Lumbre, el renderer produce el kernel y el runtime gestiona la memoria y el lanzamiento. Aquí separarás esas dos responsabilidades en el mismo archivo. Si la salida es incorrecta, podrás preguntar primero si falla el cálculo de índices o si el host entregó un buffer equivocado.
:::

## Qué se ha ejecutado en esta edición

La validación usa Ubuntu 24.04 bajo WSL2 y una NVIDIA GeForce RTX 5070 Ti Laptop, capacidad de cómputo 12.0. El compilador es `nvcc` 12.9.86; los programas se construyen para `sm_120`. El controlador observado es 591.86. El informe de cada programa conserva su hash y su código de salida.

Una arquitectura de compilación identifica las instrucciones que puede contener el binario. La cantidad de memoria, la potencia disponible y el tamaño de las matrices siguen condicionando la ejecución. El tutorial no necesita una GPU de centro de datos ni descarga modelos.

::: comprueba Antes de continuar
Localiza en el libro principal la diferencia entre forma y layout. Explica por qué una matriz de 17 por 19 no cabe en un único tile de 16 por 16. Si puedes dibujar los cuatro tiles y sus bordes, ya tienes la base para estas prácticas.
:::

# Preparar el taller y entender la compilación

## Tres piezas del entorno

El controlador permite que el proceso se comunique con el dispositivo. El toolkit proporciona `nvcc`, cabeceras y bibliotecas. El ejecutable resultante contiene código del host y código que se ejecutará en la GPU. Esas piezas tienen versiones distintas.

Abre una terminal de Linux o WSL y comprueba:

```bash
nvidia-smi
nvcc --version
```

La versión CUDA que aparece en `nvidia-smi` describe la compatibilidad del controlador; no demuestra que `nvcc` esté instalado. Si la segunda orden falla, localiza el toolkit y añade su carpeta `bin` a `PATH`. En este equipo se utiliza una instalación aislada de CUDA 12.9.1; en otro equipo la ruta puede ser distinta.

Desde la raíz del repositorio:

```bash
nvcc -O2 -lineinfo -std=c++17 -arch=sm_120 \
  tutorials/cuda/01_vectors.cu -o /tmp/cuda_vectors
/tmp/cuda_vectors
```

`-O2` solicita optimización, `-lineinfo` conserva información de líneas para diagnóstico y `-std=c++17` fija el dialecto del host. Cambia `sm_120` si tu GPU requiere otro destino que el toolkit admita. No elijas la arquitectura por el número comercial de la tarjeta.

## Ejecutar la campaña completa

```bash
python3 tutorials/validate_cuda.py --arch sm_120 \
  --output tutorials/reports/mi_cuda
```

El ejecutor usa la biblioteca estándar de Python. Compila los cuatro archivos en una carpeta temporal, guarda las salidas de los programas y elimina sus binarios al terminar. Si falla una compilación o una comparación, termina con error. Las carpetas de referencia de la edición no deben reutilizarse para tus experimentos.

## C++ que aparece en los ejemplos

`std::vector<float>` posee un array en memoria del host. `data()` entrega un puntero a sus elementos y `size()` su longitud. Un `float*` contiene una dirección, pero su tipo por sí solo no dice en qué dispositivo vive.

`DeviceBuffer<float>` es una clase del tutorial. Su constructor reserva memoria de dispositivo; su destructor la libera. El destructor se llama al salir del bloque delimitado por llaves. Se prohíbe copiar el objeto porque dos propietarios que liberasen el mismo puntero provocarían un error. Esta técnica se suele llamar RAII: ligar la duración de un recurso a la de un objeto.

`auto launch = [&]() { ... };` crea una función local que utiliza las variables del ámbito que la rodea. En la práctica de GEMM sirve para repetir exactamente el mismo lanzamiento al calentar y al medir. No compila un kernel nuevo en cada llamada.

# Una operación elemento a elemento completa

## El contrato antes del kernel

Queremos calcular $z_i=2x_i-y_i$ para índices de cero a $n-1$. Los tres arrays tienen longitud $n$, sus elementos son FP32 y la salida tiene almacenamiento propio. El programa admite $n=0$ como caso vacío y no lanza ningún kernel en ese caso.

El kernel recibe punteros que ya deben ser válidos en la GPU. `const float*` impide escribir a través del puntero de entrada dentro de esta función. La palabra `__global__` identifica una función que el host puede lanzar en el dispositivo.

```cuda
__global__ void axpby(const float* x, const float* y,
                     float* z, int n) {
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = 2.0f*x[i] - y[i];
}
```

La expresión del índice combina el número del bloque, el tamaño del bloque y la posición del hilo. Con 128 hilos por bloque y $n=257$, el host lanza tres bloques. Hay 384 hilos, pero solo 257 escriben resultados. El `if` protege también las dos lecturas. La guía de programación documenta la jerarquía de hilos. [@cuda-guide]

## Seguir un elemento de principio a fin

En el host llenamos `x` e `y`, calculamos una referencia y reservamos tres buffers. Copiamos las entradas con `cudaMemcpyHostToDevice`. Después lanzamos el kernel y copiamos `z` al host con `cudaMemcpyDeviceToHost`. La comparación ocurre cuando el trabajo de dispositivo ha terminado.

```diagram
Host: entradas y referencia | GPU: buffers y kernel | Host: salida y comparación
```

Los argumentos entre `<<<` y `>>>` son la configuración de lanzamiento. Los argumentos entre paréntesis pertenecen al kernel. Intercambiar un puntero de entrada con uno de salida puede compilar y producir una respuesta incorrecta; el compilador no conoce nuestro contrato matemático.

## Dos momentos distintos para comprobar errores

`cudaGetLastError()` después del lanzamiento detecta errores que CUDA puede identificar al enviar el trabajo. `cudaDeviceSynchronize()` espera y puede comunicar un error ocurrido durante su ejecución. El helper `CUDA(...)` imprime la operación y la línea antes de terminar si el estado es incorrecto. [@cuda-runtime]

No uses `printf` de miles de hilos como prueba principal. Una salida de depuración puede ayudar a seguir un caso pequeño, pero la comprobación numérica debe recorrer todo el resultado.

::: practica Predicción con 129 elementos
Calcula cuántos bloques se lanzan y cuántos hilos quedan fuera del dominio. Busca el último hilo que escribe. Solución: dos bloques, 256 hilos, 127 inactivos; el índice 128 corresponde al hilo cero del segundo bloque.
:::

# Memoria, accesos y resultados numéricos

## Un puntero no representa una copia automática

La variable `dx.ptr` está en el host, pero su valor identifica almacenamiento de dispositivo. Leer `dx.ptr[0]` desde C++ del host no equivale a leer el primer dato mediante CUDA. En estos programas usamos reservas y transferencias explícitas para que la propiedad de cada array sea visible.

No liberamos una entrada mientras un kernel pueda utilizarla. En la primera práctica la sincronización termina el trabajo antes de salir del ámbito. En la práctica de streams esperaremos a la cola correspondiente antes de leer o liberar la memoria fijada del host.

## Accesos contiguos

Los hilos vecinos del kernel vectorial acceden a elementos vecinos. Ese patrón permite que el dispositivo agrupe accesos a memoria. Un stride grande puede obligar a transportar más segmentos para obtener la misma cantidad de datos útiles. Para estudiar este efecto necesitas un experimento con el mismo número de operaciones y una salida verificable.

La contigüidad es una propiedad de las direcciones generadas, no del nombre de una dimensión. En una matriz almacenada por filas, el índice `row*k+q` recorre una fila cuando cambia `q`. En `q*n+col`, mantener `q` y variar `col` recorre posiciones contiguas de B.

## Errores que una tolerancia no debe ocultar

El comparador verifica que la salida sea finita y aplica:

```math
|z_i-r_i|\leq \mathrm{atol}+\mathrm{rtol}|r_i|.
```

La tolerancia absoluta importa cerca de cero; la relativa escala con el tamaño del resultado. Las entradas vectoriales son múltiplos de potencias de dos, elegidos para facilitar el seguimiento manual. Una campaña posterior debe incluir otros valores y escalas.

Para una reducción larga, cambiar el orden de suma cambia el redondeo. Si acumulas una referencia en FP64 y el kernel en FP32, una discrepancia pequeña puede ser esperable. Antes de ampliar el margen, comprueba que no faltan elementos, que la salida no contiene NaN y que las diferencias crecen de una forma coherente con el problema.

::: comprueba Diseñar un fallo observable
En una copia de trabajo, cambia el límite a `i < n-1`. El caso de un elemento debería fallar. Explica por qué un conjunto de pruebas formado solo por ceros podría no detectar esa escritura ausente. Restaura el límite antes de medir.
:::

# Cooperación dentro del bloque: reducir una suma

## Dividir el problema

Una reducción combina muchos valores en uno. El programa `02_reduce.cu` asigna hasta 128 entradas a cada bloque. El bloque produce una suma parcial y el host combina esas sumas en FP64. Así evitamos introducir a la vez una segunda reducción GPU y la gestión de sus buffers.

La memoria `__shared__` pertenece al bloque. Otro bloque tiene su propio array `tile`, aunque la declaración sea la misma. Cada hilo escribe una posición; los hilos que quedan fuera de la entrada escriben cero, el elemento neutro de la suma.

## Por qué la barrera está fuera del condicional

Todos los hilos del bloque llegan a la primera barrera después de cargar. Luego la mitad suma pares, una cuarta parte suma pares de parciales, y así hasta obtener un valor. Tras cada nivel, una barrera garantiza que las lecturas del siguiente nivel vean las escrituras anteriores.

```cuda
tile[t] = i<n ? x[i] : 0.0f;
__syncthreads();
for (int stride=64; stride>0; stride/=2) {
  if (t<stride) tile[t] += tile[t+stride];
  __syncthreads();
}
```

El contrato de este kernel exige exactamente 128 hilos. La longitud del array y la secuencia de strides dependen de ese número. Cambiar el lanzamiento a 256 hilos sin cambiar el kernel escribiría fuera de `tile`.

Un retorno temprano de los hilos que no tienen entrada rompería la estructura de participación en las barreras. En una operación con cooperación, estar fuera del dominio de salida no significa que un hilo pueda abandonar el algoritmo.

## Completar la reducción en el dispositivo

Para una segunda práctica, reserva un buffer para los parciales y vuelve a aplicar el mismo kernel hasta que quede un valor. En cada etapa, la longitud nueva es el número de bloques de la etapa anterior. Conserva ambas reservas hasta que termine la etapa que las usa y evita leer y escribir sobre regiones que se solapan.

El tamaño cero necesita una decisión del host: la suma vacía vale cero y no necesita una grid vacía. Un tamaño de un elemento sigue el mismo contrato que los demás.

::: practica Dibujar el árbol
Haz el dibujo para ocho entradas 1, 2, 3, 4, 5, 6, 7 y 8. Tras combinar posiciones separadas por cuatro obtienes 6, 8, 10 y 12; después 16 y 20; finalmente 36. Indica en qué puntos dos hilos podrían competir si quitaras una barrera.
:::

# GEMM: de la fórmula al tile

## La versión directa

Para A de forma [M,K] y B de forma [K,N], queremos C de forma [M,N]. Cada hilo calcula una posición de C. El eje x recorre columnas y el eje y filas. El bloque es de 16 por 16, por lo que contiene 256 hilos.

```math
C_{r,c}=\sum_{q=0}^{K-1} A_{r,q}B_{q,c}.
```

La versión directa repite lecturas: los hilos que producen una misma fila de C necesitan elementos comunes de A. Los que producen una misma columna necesitan elementos comunes de B. Esa reutilización motiva el tile.

## Cargar una pieza de cada entrada

El kernel compartido reserva dos arrays de 16 por 16. Cada hilo carga un elemento de A y uno de B, espera, y utiliza la fila de un tile y la columna del otro. Los 256 hilos cooperan para producir hasta 256 resultados.

Para el tile que comienza en `base`, el hilo `(tx,ty)` carga A en `(row,base+tx)` y B en `(base+ty,col)`. Si una coordenada cae fuera, escribe cero. La primera barrera publica ambas cargas. La segunda impide sobrescribir el tile mientras otro hilo lo sigue consumiendo.

El acumulador `sum` vive en cada hilo y persiste durante todas las piezas de K. La escritura final necesita máscara porque un bloque de borde también contiene hilos sin salida válida.

## Los bordes son parte del algoritmo

En la forma [17,19,23], el programa necesita dos bloques en M, dos en N y dos piezas de K. La última pieza solo contiene siete valores de reducción útiles. Las posiciones restantes de la memoria compartida deben contener cero en esa iteración, no los valores de la anterior.

El tutorial prueba [1,1,1], [17,19,23], [31,47,65], [128,128,128] y [256,256,256]. Cada forma se ejecuta con los dos kernels y se compara con una referencia que acumula en doble precisión.

## Qué se puede concluir de la comparación

Ambos programas calculan la misma GEMM, pero gestionan la memoria de manera distinta. El tile introduce cargas compartidas y dos barreras por iteración. En matrices pequeñas, ese trabajo puede superar el ahorro. Para matrices mayores, influyen el tráfico, las instrucciones y la ocupación.

Una GEMM FP32 escrita con estos bucles no se convierte en una operación Tensor Core por tener una GPU reciente. La demostración WMMA del libro principal tiene un contrato diferente de tipos, layout y participación por warp. Integrar esa ruta exige otra prueba de corrección.

::: practica Cambiar una sola decisión
Compara la forma irregular [31,47,65] con [32,48,64]. Explica qué máscaras siguen siendo necesarias y cuántos tiles de K ejecuta cada una. La segunda usa cuatro piezas completas; la primera necesita cinco. Mide ambas antes de atribuir toda diferencia a la irregularidad.
:::

# Medir trabajo terminado

## Definir el intervalo

El programa de GEMM reserva y copia antes de medir. Ejecuta tres calentamientos y después recoge treinta muestras. Cada muestra encierra cien lanzamientos entre dos eventos CUDA en la misma cola. Divide el intervalo entre cien y expresa el resultado en microsegundos por lanzamiento.

La muestra puede incluir huecos de envío entre kernels. Describe el tiempo de esa secuencia residente, no una latencia de instrucción ni el coste completo del proceso. La referencia CPU, las reservas y las transferencias iniciales quedan fuera.

```cuda
CUDA(cudaEventRecord(start));
for (int repeat=0; repeat<100; ++repeat) launch();
CUDA(cudaEventRecord(stop));
CUDA(cudaEventSynchronize(stop));
CUDA(cudaEventElapsedTime(&ms,start,stop));
```

Esperar al evento de fin hace que el intervalo esté disponible. Ordenamos las treinta muestras y promediamos las dos centrales para obtener la mediana. El registro conserva también mínimo y máximo. Si la dispersión cambia mucho entre ejecuciones, investiga temperatura, potencia y otros procesos antes de defender una mejora.

## No mezclar los relojes de los dos libros

Lumbre mide desde Python y sincroniza el runtime. Este tutorial usa eventos y agrupa cien lanzamientos. Sus tablas responden a preguntas diferentes. Aunque ambos programas calculen GEMM, dividir una latencia de un libro por otra del otro no produce una comparación controlada de compiladores.

Para medir extremo a extremo, inicia un reloj del host antes de la reserva o del trabajo que decidas incluir y detenlo después de una sincronización final. Escribe esa frontera en el informe. Conserva los resultados numéricos: un programa más rápido que omite parte de C no es una mejora.

::: comprueba Interpretar unidades
Si cien lanzamientos tardan 0,8 milisegundos, cada uno tarda 8 microsegundos en promedio dentro de ese lote. Para una GEMM [M,N,K], el recuento habitual es aproximadamente $2MNK$ operaciones. No presentes ese cociente como utilización del pico sin fijar también la precisión y la ruta de instrucciones.
:::

# Streams y vida de los buffers

## Una cola con dependencias claras

En `04_stream.cu`, el host crea un stream, encola una transferencia de entrada, el kernel y la transferencia de salida. Las operaciones del mismo stream respetan ese orden. Antes de leer la salida, el host espera a ese stream.

Los arrays del host se reservan con `cudaMallocHost`: CUDA mantiene esas páginas fijadas para las transferencias. Se liberan con `cudaFreeHost`, después de la espera. No se debe reutilizar el array de entrada mientras la copia pendiente aún lo consume.

```diagram
Copia host a GPU | Kernel en el stream | Copia GPU a host
```

El cuarto argumento de lanzamiento especifica el stream. El tercero es la cantidad de memoria compartida dinámica y aquí vale cero. Las prácticas anteriores usan memoria compartida de tamaño conocido en compilación.

## Asincronía y solapamiento

Una llamada asíncrona permite al host continuar antes de que termine el trabajo. El solapamiento exige además operaciones independientes y recursos disponibles. Este ejemplo utiliza una cola para enseñar dependencia y vida de buffers; no demuestra una aceleración mediante concurrencia.

Para extenderlo a dos trozos independientes, cada trozo necesitaría regiones separadas de entrada, salida y dispositivo. Una dependencia entre streams requiere una espera explícita, por ejemplo mediante un evento. La división de los datos no garantiza que la suma de los tiempos disminuya.

::: practica Encontrar la carrera del host
Describe qué puede pasar si lees `output[0]` antes de `cudaStreamSynchronize`. El host puede observar un valor anterior o todavía no escrito. Una ejecución que casualmente devuelve lo esperado no demuestra que el programa esté ordenado.
:::

# Depurar, ampliar y volver al compilador

## Un orden de investigación

Ante una salida incorrecta, reduce la forma, calcula una referencia y conserva el primer índice que difiere. Comprueba tamaños en bytes, orden de argumentos y máscaras. Después revisa barreras y duración de los buffers. Para una forma que solo falla en tiles incompletos, inspecciona primero las cargas enmascaradas.

Si dispones de Compute Sanitizer, puedes ejecutar una copia del binario con `compute-sanitizer --tool memcheck`. Las comprobaciones `racecheck` y `synccheck` estudian otros tipos de fallo. Esta edición no presenta esos análisis como ejecutados: la campaña incorporada compila, ejecuta y compara resultados.

Una prueba numérica y un analizador de memoria aportan evidencias distintas. Ni uno ni otro sustituye el razonamiento sobre el contrato y el dominio de índices.

## Proyecto de cierre

Añade un bias por columna a GEMM y aplica ReLU antes de escribir C. Construye una referencia CPU que haga exactamente esa operación, incluyendo valores negativos. Conserva las cinco formas de prueba y mide la variante fusionada con la misma frontera que la original.

En una segunda versión, ejecuta GEMM y un kernel de bias/ReLU por separado. Compara resultados antes de comparar tiempos. Cuenta las escrituras y lecturas adicionales del intermedio. Este experimento conecta con la fusión del compilador sin requerir que cambies su IR.

## Solución de diseño

En la variante fusionada, cada hilo calcula su acumulador, añade `bias[col]` y escribe el máximo con cero solo cuando su salida es válida. El bias tiene longitud N y debe copiarse al dispositivo. La versión separada recibe M por N elementos y obtiene la columna con el resto de dividir el índice lineal entre N.

La corrección incluye los bordes, la comparación con referencia y el rechazo de valores no finitos. El informe debe indicar si la medida incluye solo los kernels o también las transferencias. Conserva el código original para poder repetir la comparación.

::: comprueba Criterio de finalización
Has completado el tutorial cuando puedes compilar los cuatro programas, explicar los dos controles de error tras un lanzamiento, justificar ambas barreras de GEMM y distinguir una cola asíncrona de un experimento que demuestra solapamiento. Entrega también una modificación con referencia y resultados, no solo una captura del tiempo.
:::
