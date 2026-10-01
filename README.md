# Compiladores de ML desde cero

Libro y código para construir un compilador de aprendizaje automático desde cero: **Lumbre**, CPU/CUDA, autodiferenciación y un decoder Transformer entrenable.

Empieza con programas, arrays y tensores. Aprende a construir una representación intermedia, generar código C y CUDA, planificar memoria, calcular gradientes y entrenar un modelo pequeño con tu propio compilador.

**Edición del 1 de octubre de 2026 · 305 páginas · 68 capítulos · 12 laboratorios · 5 proyectos finales.**

[Leer el libro en PDF](Compiladores_ML_Desde_Cero_2026.pdf) · [Guía del código](code/README.md) · [Resultados de validación](code/reports/validation_20261001/validation.json)

## Qué encontrarás

| Recorrido | Contenido |
| --- | --- |
| Fundamentos | Python, C, terminal, tensores, índices y coma flotante. |
| Construcción del compilador | UOps, simplificación simbólica, reescrituras, generación de código y planificación de memoria. |
| Kernels | Broadcasting, movimientos, reducciones, GEMM, convoluciones y mediciones de rendimiento. |
| GPU | Runtime CUDA/HIP, memoria compartida, sincronización, GEMM por tiles y una demostración WMMA. |
| Modelos | Autodiferenciación, atención online, decoder Transformer, AdamW, ByteBPE y checkpoints. |
| Práctica | Laboratorios, ejercicios y proyectos para ampliar y comprobar el compilador. |

Lumbre es un compilador didáctico original. Python construye y diferencia grafos; el código nativo generado ejecuta los cálculos tensoriales. NumPy proporciona almacenamiento, datos y referencias numéricas. Puedes estudiar y ejecutar el proyecto sin depender de PyTorch ni de tinygrad.

El libro incluye 15 diagramas vectoriales, 122 bloques de código, los 21 archivos completos del proyecto ejecutable y 69 referencias bibliográficas con enlaces. Empieza por **«Empieza aquí»** y el capítulo 1. Encontrarás el índice detallado al final y 663 marcadores navegables en el PDF.

## Ejecutar en CPU

Necesitas Linux o WSL, Python 3.11 o posterior y un compilador C disponible como `cc`. Las dependencias fijadas son NumPy 2.3.5 y pytest 9.0.2. En Windows, ejecuta Lumbre dentro de WSL: el runtime carga bibliotecas compartidas POSIX.

```bash
git clone https://github.com/Kripta-Studios/compiladores-ml-desde-cero.git
cd compiladores-ml-desde-cero/code

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'

python -m pytest -q
python examples/validate_book.py --output reports/mis_ejemplos.json
python examples/verify_launch.py --output reports/mis_mapas.json
python examples/train_decoder.py --steps 20 --gemm blocked --output reports/mi_decoder
```

Guarda tus experimentos en carpetas nuevas para conservar los informes y checkpoints de referencia. Consulta [la guía de Lumbre](code/README.md) para usar un corpus propio, entrenar con ByteBPE o reanudar un checkpoint.

## Ejecutar en CUDA

Necesitas una GPU NVIDIA compatible, un controlador operativo y `nvcc` en `PATH`. Selecciona una arquitectura que corresponda a tu dispositivo y que admita tu toolkit. En la RTX 5070 Ti Laptop usada para validar este repositorio, el destino es `sm_120` y el toolkit es CUDA 12.9.

Desde `code/`, con el entorno Python activado:

```bash
export LUMBRE_CUDA_ARCH=sm_120
export LUMBRE_CACHE="$HOME/.cache/lumbre-mi-validacion"

python -m pytest -q --backend=cuda
python examples/validate_book.py --backend cuda --output reports/mis_ejemplos_cuda.json
python examples/bench_kernels.py --backend cuda --repeat 20 --output reports/mi_benchmark_cuda.json

nvcc -O2 -std=c++17 -arch=sm_120 kernels/wmma_demo.cu -o /tmp/wmma_demo
/tmp/wmma_demo
```

Para reproducir el entrenamiento principal de **75.584 parámetros y 600 actualizaciones**:

```bash
python examples/train_decoder.py --backend cuda --steps 600 \
  --dim 64 --layers 2 --context 32 --batch 2 \
  --heads 2 --kv-heads 1 --hidden 128 \
  --online --gemm blocked --output reports/mi_decoder_cuda_600
```

Puedes ejecutar la misma configuración en CPU cambiando `--backend cuda` por `--backend cpu`.

## Resultados comprobados

La revisión local utiliza Ubuntu 24.04 bajo WSL2, Python 3.12.3, GCC 13.3.0, NumPy 2.3.5 y pytest 9.0.2. Para CUDA se usó una NVIDIA GeForce RTX 5070 Ti Laptop de 12 GB, controlador 591.86, `nvcc` 12.9.86 y destino `sm_120`.

| Comprobación | Resultado | Evidencia |
| --- | --- | --- |
| Batería CPU | 61 pruebas superadas, sin omisiones. | [Registro CPU](code/reports/validation_20261001/tests_cpu.log) |
| Batería con backend CUDA | 61 pruebas superadas, sin omisiones. | [Registro CUDA](code/reports/validation_20261001/tests_cuda.log) |
| Ejemplos numéricos del libro | Tres ejemplos con pérdidas y gradientes correctos en ambos backends. | [CPU](code/reports/validation_20261001/book_examples_cpu.json), [CUDA](code/reports/validation_20261001/book_examples_cuda.json) |
| Bordes e índices | Longitudes 0, 1, 127, 128, 129 y 257; 15 mapas lógicos y rechazo de un error intencional. | [Bordes CUDA](code/reports/validation_20261001/boundaries_cuda.json), [mapas](code/reports/validation_20261001/passport.json) |
| Kernels | 12 mediciones residentes por backend, con comprobación numérica. | [CPU](code/reports/validation_20261001/kernels_cpu.json), [CUDA](code/reports/validation_20261001/kernels_cuda.json) |
| WMMA | Compilación y ejecución; error absoluto máximo cero. | [Registro WMMA](code/reports/validation_20261001/wmma_run.log) |
| Reanudación | Checkpoints idénticos para 10 pasos frente a 6 + 4, dentro de cada backend. | [CPU](code/reports/validation_20261001/resume_cpu.json), [CUDA](code/reports/validation_20261001/resume_cuda.json) |
| Decoder principal en CUDA | 600 pasos; pérdida final 0,2508018 y evaluación 0,3293002. | [Informe y checkpoint](code/reports/validation_20261001/decoder_cuda_75584/) |
| Libro | Tres pasadas LaTeX, referencias resueltas y 305 páginas renderizadas; 21 archivos incrustados coinciden con el código. | [Compilación](code/reports/validation_20261001/book_build.json), [revisión PDF](pdf_qa.json) |

Al seleccionar CUDA, los programas tensoriales de las pruebas se ejecutan en la GPU. Las pruebas de tokenización, estructuras del host y convolución C independiente mantienen su ejecución en CPU. El verificador de mapas enumera índices en CPU y comprueba un contrato lógico.

Conservamos las evidencias originales de CPU junto con la [revisión local completa](code/reports/validation_20261001/validation.json). Los tiempos de compilación, ejecución y evaluación tienen fronteras distintas; consulta cada informe antes de comparar resultados.

## Reconstruir el libro

Desde la raíz del repositorio, edita los Markdown de `chapters/` o el código y regenera el fuente:

```bash
python build_book.py
```

Compila con una distribución TeX que incluya los paquetes del preámbulo: babel con español, Latin Modern, TikZ, tcolorbox, listings y titlesec, entre otros.

```bash
pdflatex -interaction=nonstopmode -halt-on-error Compiladores_ML_Desde_Cero_2026.tex
pdflatex -interaction=nonstopmode -halt-on-error Compiladores_ML_Desde_Cero_2026.tex
pdflatex -interaction=nonstopmode -halt-on-error Compiladores_ML_Desde_Cero_2026.tex
```

También puedes usar `latexmk -pdf Compiladores_ML_Desde_Cero_2026.tex`. El constructor usa la biblioteca estándar de Python y produce el mismo fuente UTF-8 en Windows y Linux. El `.tex` principal es autocontenido: incorpora texto, diagramas, bibliografía y listados. No necesita BibTeX, imágenes externas ni `--shell-escape`.

## Estructura del repositorio

```text
.
├── Compiladores_ML_Desde_Cero_2026.pdf   # Libro listo para leer
├── Compiladores_ML_Desde_Cero_2026.tex   # Fuente autocontenido
├── chapters/                            # Capítulos editables en Markdown
├── preamble.tex                         # Tipografía, estilos y macros
├── appendices.tex                       # Validación y reproducción
├── bibliography.tex                     # Referencias
├── build_book.py                        # Constructor del libro
├── code/
│   ├── lumbre/                          # IR, compilador, autodiff y modelo
│   ├── kernels/                         # Convolución C y demostración WMMA
│   ├── examples/                        # Entrenamiento, benchmarks y validadores
│   ├── tests/                           # Pruebas CPU/CUDA/HIP
│   └── reports/                         # Evidencias y checkpoints
├── pdf_qa.json                          # Revisión del PDF
└── MANIFEST_SHA256.txt                   # Integridad de los archivos publicados
```

## Alcance

El corpus de demostración repite seis frases para depurar el compilador y el entrenamiento. La pérdida obtenida acredita aprendizaje sobre ese material; la evaluación de calidad lingüística general requiere otros datos y otro diseño experimental.

La ruta HIP está implementada y queda pendiente de validación en hardware compatible. La atención online reduce intermediarios de forward; su backward recompone operaciones densas. Los proyectos de FP8/FP4, entrenamiento distribuido, MoE distribuido y KV cache incremental describen extensiones por desarrollar. Los resultados locales no establecen paridad con PyTorch, BLAS industrial ni rendimiento SOTA.

## Fuentes y licencia

La exposición, los ejemplos, los diagramas y Lumbre son originales. El curso es material didáctico independiente de Tiny Corp, NVIDIA, AMD, Huawei y las universidades. La investigación se cierra el **1 de octubre de 2026**; la bibliografía distingue artículos, documentación y repositorios. La referencia a tinygrad se ancla al commit `c3aec477b99d9bb87c54d91897cf60acd3f17441`, fechado el 30 de septiembre de 2026.

El código original de Lumbre utiliza la [licencia MIT de `code/LICENSE`](code/LICENSE), cuyo alcance se especifica en ese archivo. Esa licencia no modifica los derechos de las publicaciones, repositorios o marcas externos que citamos. Consulta [la bibliografía](bibliography.tex) para localizar las fuentes.
