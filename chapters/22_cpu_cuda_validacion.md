# CPU y CUDA en la misma máquina: de la prueba a la medida {#ch:cpu-cuda-validacion}

Disponer de una GPU cambia las preguntas que podemos contestar. Ya podemos compilar el mismo grafo para dos destinos, comprobar sus resultados y observar cuánto cuesta ejecutarlo en cada uno. El objetivo de este capítulo es construir una comparación que puedas repetir y explicar. Los informes que acompañan al libro incluyen tanto la primera validación de CUDA como una campaña posterior con tareas ejecutadas en secuencia.

## El entorno que realmente ejecutó el código

La máquina de validación ofrece una NVIDIA GeForce RTX 5070 Ti Laptop de 12 GB. Dentro de Ubuntu 24.04 en WSL2 usamos Python 3.12.3, GCC 13.3.0, NumPy 2.3.5 y pytest 9.0.2. El controlador observado es 591.86; el compilador CUDA es `nvcc` 12.9.86 y el destino es `sm_120`. Puedes consultar el inventario completo en `code/reports/validation_20261001/environment.json`.

Hay tres piezas que conviene distinguir al diagnosticar una ejecución. El controlador permite utilizar el dispositivo. El toolkit aporta el compilador y sus cabeceras y bibliotecas. El runtime de Lumbre genera y carga una biblioteca anfitriona que reserva memoria, copia datos y lanza kernels. Ver una GPU en `nvidia-smi` solo comprueba una parte de esa cadena.

En esta máquina también existía un toolkit Windows 12.4. La validación de Lumbre utilizó el toolkit Linux 12.9 dentro de WSL, porque el runtime suministrado carga bibliotecas POSIX. Un `nvcc` encontrado en una terminal Windows no identifica por sí mismo el compilador que utilizará Python dentro de WSL. Registra el ejecutable que resuelve cada entorno.

Desde una terminal WSL y con el entorno Python del proyecto activado, comprueba:

```bash
python --version
cc --version
command -v nvcc
nvcc --version
nvidia-smi
```

Si usas el toolkit aislado de la validación local, su directorio es `~/.local/opt/cuda-12.9.1/bin`. En otra instalación utiliza la ruta de tu propio toolkit. El valor de `LUMBRE_CUDA_ARCH` debe corresponder a tu dispositivo; el valor siguiente describe la RTX validada.

```bash
export LUMBRE_CUDA_ARCH=sm_120
export LUMBRE_CACHE="$HOME/.cache/lumbre-comparacion"
```

::: comprueba Qué significa seleccionar un backend
La variable `LUMBRE_CUDA_ARCH` configura el destino del compilador CUDA. Para seleccionar el dispositivo también debes pasar `backend="cuda"` a `Program`, `--backend cuda` a los ejemplos o `--backend=cuda` a pytest. Una variable de arquitectura no convierte una ejecución CPU en una ejecución CUDA.
:::

## La escalera de aceptación

El primer escalón comprueba tamaños pequeños y bordes. Con bloques de 128 hilos, 257 elementos requieren tres bloques: se lanzan 384 hilos y 127 posiciones deben quedar enmascaradas. También interesa probar 127, 128 y 129 para cruzar el límite del bloque. El tamaño cero comprueba que el renderer omite el kernel cuando la salida está vacía; lanzar una grid de tamaño cero sería otra operación.

El segundo escalón comprueba matrices irregulares. En una GEMM por tiles, un hilo cuya salida queda fuera de la matriz puede seguir participando en cargas compartidas y barreras. Las pruebas deben ejercer bordes en M, N y K, broadcasting entre lotes y una dimensión de reducción vacía admitida por el contrato. El kernel compartido de Lumbre usa tiles de 16 por 16 y enmascara las cargas de los tiles incompletos.

El tercer escalón comprueba operaciones compuestas y gradientes. Las pruebas existentes incluyen movimientos, reducciones, gather/scatter, softmax, convolución por composición, atención online y causalidad del decoder. Finalmente, el entrenamiento debe mantener pérdidas y gradientes finitos, guardar sus pesos y reanudar el estado del optimizador y del generador de números aleatorios.

Desde `code/`, ejecuta cada backend de forma explícita:

```bash
python -m pytest -q --backend=cpu
python -m pytest -q --backend=cuda
python examples/validate_book.py --backend cpu \
  --output reports/comparacion/ejemplos_cpu.json
python examples/validate_book.py --backend cuda \
  --output reports/comparacion/ejemplos_cuda.json
```

La batería registrada superó 61 pruebas en cada selección de backend, sin omisiones. Esa cifra corresponde a la batería completa: el tokenizador, los mapas lógicos y la convolución C independiente mantienen su trabajo en el host. Los `Program` tensoriales seleccionan CUDA cuando se pide ese backend. El archivo `tests/conftest.py` implementa esta selección; un compilador o dispositivo ausente provoca fallos en las pruebas que lo necesitan.

## Comparar valores y gradientes antes de comparar segundos

Para un mismo grafo, utiliza las mismas entradas, dtypes y formas. La comprobación elemento a elemento de una referencia finita suele expresar dos márgenes:

```math
|y_{\mathrm{CUDA}}-y_{\mathrm{CPU}}|
\leq \mathrm{atol}+\mathrm{rtol}|y_{\mathrm{CPU}}|.
```

La tolerancia absoluta protege la comparación cerca de cero. La relativa escala con la magnitud de la referencia. Una tolerancia debe apoyarse en la operación, la precisión y la longitud de la reducción. Aumentarla hasta ocultar un índice incorrecto destruye el valor de la prueba.

En el decoder de depuración de 2.736 parámetros, las historias de veinte pasos registradas en CPU y CUDA difieren como máximo en `4.76837158203125e-7` en la pérdida. Ese resultado pertenece a esa configuración y esas entradas. Las reducciones y las transformaciones de coma flotante pueden producir diferencias que crezcan a lo largo de un entrenamiento de muchas actualizaciones.

La igualdad exacta tiene otro papel en la prueba de reanudación. Dentro de cada backend, compara diez actualizaciones continuas con seis seguidas de cuatro al cargar el checkpoint. Deben coincidir los pesos, los momentos de AdamW, el contador de pasos, la configuración y el estado del RNG. Los informes `resume_cpu.json` y `resume_cuda.json` de la primera revisión registran esa igualdad. No se exige identidad bit a bit entre los checkpoints CPU y CUDA de dos entrenamientos completos.

::: practica Localizar una divergencia
Un modelo guarda pesos finitos, pero su pérdida CUDA se separa mucho de la CPU desde el primer paso. Compara primero el forward con los mismos pesos y el mismo batch. Después compara la pérdida, los gradientes por parámetro y la actualización del optimizador. Si el forward ya diverge, inspeccionar el muestreo de texto no resolverá la primera diferencia.
:::

## Medir con una frontera explícita

El benchmark de Lumbre construye el grafo, compila el programa y carga entradas antes de medir. Ejecuta tres calentamientos y después cronometra `run(read=[])`. La medida incluye la llamada Python y la sincronización que hace el runtime; excluye la compilación, la carga inicial de entradas y la copia de los resultados al host. Su unidad de observación es una llamada residente al programa.

En CUDA, una medida de host que solo cronometra el lanzamiento puede terminar antes que el trabajo de la GPU. Lumbre sincroniza al final de `run`, de modo que esta medida incluye la finalización del programa. Aun así, no equivale al tiempo aislado de una instrucción del dispositivo ni al de todo el proceso Python.

Para repetir la campaña, desde `code/` y con la arquitectura configurada:

```bash
python examples/bench_kernels.py --backend cpu --repeat 30 \
  --output reports/comparacion/kernels_cpu.json
python examples/bench_kernels.py --backend cuda --repeat 30 \
  --output reports/comparacion/kernels_cuda.json
```

Espera a que termine una orden antes de lanzar la siguiente. La campaña incorporada a este capítulo ejecutó sus tareas en secuencia en el mismo host WSL. No fijó frecuencias, potencia ni carga del escritorio: las cifras son observaciones locales que conviene repetir antes de defender una mejora de rendimiento. Los JSON conservan las treinta muestras, mínimo, máximo, mediana, error numérico y metadatos del programa.

En GEMM, el campo `shape` usa el orden **[M,N,K]**: la entrada A tiene forma [M,K] y B tiene forma [K,N]. En atención la forma es [B,H,T,D]. La tabla utiliza microsegundos por llamada residente y redondea la mediana para facilitar la lectura.

| Operación y forma | Variante | CPU (us) | CUDA (us) |
|---|---|---|---|
| gemm 31x47x65 | reference | 329.94 | 464.43 |
| gemm 31x47x65 | blocked | 97.92 | 205.93 |
| gemm 128x128x128 | reference | 12840.66 | 364.82 |
| gemm 128x128x128 | blocked | 1644.77 | 423.72 |
| gemm 256x256x256 | reference | 106237.30 | 415.52 |
| gemm 256x256x256 | blocked | 12203.57 | 426.81 |
| attention 1x2x16x32 | dense | 49.18 | 1583.98 |
| attention 1x2x16x32 | online | 40.28 | 581.51 |
| attention 1x2x64x32 | dense | 828.06 | 954.86 |
| attention 1x2x64x32 | online | 671.03 | 550.33 |
| attention 1x2x128x32 | dense | 3143.98 | 1313.48 |
| attention 1x2x128x32 | online | 2369.11 | 896.29 |

Las formas pequeñas permiten estudiar cuánto pesa el coste de llamada y sincronización frente a la cantidad de aritmética. Al aumentar una dimensión, vuelve a medir: la relación entre memoria, paralelismo y trabajo cambia. Una variante denominada `blocked` tampoco tiene que ganar en todas las formas o en todos los dispositivos. La implementación CPU bloquea bucles; la CUDA utiliza memoria compartida y barreras. Comparten el contrato matemático, pero realizan trabajos de gestión distintos.

Para calcular un cociente comparable, divide la mediana CPU por la mediana CUDA de la misma fila. Un resultado mayor que uno indica menor tiempo CUDA en ese ensayo; uno menor que uno indica menor tiempo CPU. Conserva ambas latencias y las muestras originales. Un cociente aislado omite la escala del trabajo y la variación de las medidas.

## Entrenar la misma configuración en ambos destinos

La configuración principal tiene vocabulario 24, lote 2, contexto 32, dimensión 64, dos capas, dos cabezas de consulta, una cabeza de claves/valores y dimensión intermedia 128. Usa atención online y GEMM bloqueada: 75.584 parámetros, 447 unidades de kernel y una arena planificada de 2.100.160 bytes.

```bash
python examples/train_decoder.py --backend cpu --steps 600 \
  --dim 64 --layers 2 --context 32 --batch 2 \
  --heads 2 --kv-heads 1 --hidden 128 \
  --online --gemm blocked --output reports/comparacion/decoder_cpu
python examples/train_decoder.py --backend cuda --steps 600 \
  --dim 64 --layers 2 --context 32 --batch 2 \
  --heads 2 --kv-heads 1 --hidden 128 \
  --online --gemm blocked --output reports/comparacion/decoder_cuda
```

Los reportes de esta campaña están en `code/reports/book_gpu_extension/`. Cada ejecución conserva su historia de entrenamiento, configuración, checkpoint y evaluación. El tiempo del bucle excluye compilación, evaluación y generación posteriores; incluye el trabajo que realiza el entrenador para alimentar los lotes y actualizar los parámetros.

| Backend | Pasos | Pérdida final | Evaluación | Bucle (s) |
|---|---|---|---|---|
| cpu | 600 | 0.2746565 | 0.3321578 | 18.744 |
| cuda | 600 | 0.2508018 | 0.3293002 | 28.685 |

El plan contiene muchos kernels pequeños. Tras cada `run`, el runtime sincroniza; las asignaciones del optimizador también requieren operaciones explícitas. Una GPU puede resolver una GEMM grande con rapidez y dedicar una fracción considerable del paso a gestionar cientos de lanzamientos y transferencias pequeñas. Para explicar el tiempo total, inspecciona el plan, el tamaño de cada operación y las copias del entrenador. El número de parámetros por sí solo no identifica el cuello de botella.

La arena contabiliza los buffers que administra Lumbre. No es la memoria total que verás en `nvidia-smi`: ese inventario incluye otros usos del dispositivo y del runtime. Del mismo modo, un entrenamiento que reduce su pérdida sobre las seis frases repetidas del corpus de demostración verifica aprendizaje y ejecución, pero requiere otra evaluación para estudiar generalización lingüística.

## WMMA: una comprobación con alcance propio

La demostración `kernels/wmma_demo.cu` multiplica un tile de 16 por 16 con entradas FP16 y acumulación FP32. Un warp participa en las operaciones colectivas de carga, multiplicación y escritura. El programa comprueba errores de reserva, transferencia, lanzamiento y sincronización, y compara la salida con una referencia calculada en el host.

La RTX de validación ejecutó este programa con error absoluto máximo cero para los datos deterministas de la demostración. Esos valores tienen una estructura numérica sencilla; el cero observado no garantiza error cero para cualquier entrada FP16. Cambiar valores, distribuciones y escalas forma parte de la siguiente prueba.

El entrenamiento de Lumbre utiliza el renderer tensorial y su GEMM compartida. La demostración WMMA es un programa separado. Integrarla requiere reconocer la operación, adaptar formas y tipos, resolver los tiles de borde y verificar el error de la nueva ruta. Conserva una comparación contra el kernel anterior para detectar una conversión o un layout incorrectos.

## Qué entregar al terminar la comparación

Tu informe debe permitir que otra persona reconstruya el experimento. Incluye el commit del código, el entorno, el destino CUDA, las formas, dtypes y semillas, las órdenes exactas y la política de caché. Adjunta resultados numéricos y muestras de tiempo. Identifica qué incluye cada reloj y qué trabajo quedó fuera.

Un experimento siguiente puede estudiar una sola hipótesis: fusionar operaciones elemento a elemento, agrupar transferencias del optimizador o reducir sincronizaciones conservando las dependencias. Mantén el caso original, comprueba valores y gradientes y repite la medida. Una mejora que solo funciona al reutilizar por accidente un buffer no supera la prueba de corrección.

::: comprueba Puerta de salida
Puedes completar este capítulo cuando reproduces una prueba de bordes, explicas una comparación numérica CPU/CUDA, localizas las treinta muestras de un benchmark y distingues el tiempo de una llamada residente del tiempo de entrenamiento. También debes poder señalar qué evidencia valida WMMA y qué parte del renderer sigue utilizando la GEMM compartida.
:::
