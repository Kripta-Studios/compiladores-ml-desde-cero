# Lumbre — compilador de ML desde cero

Código original que acompaña al libro **Compiladores para aprendizaje automático: desde cero**, edición 1 de octubre de 2026. Python construye y diferencia grafos; los cálculos tensoriales del entrenamiento se ejecutan en C generado. NumPy se usa para almacenamiento, datos, referencias y operaciones de muestreo del host, no como motor de autodiferenciación. No depende de PyTorch ni de tinygrad.

## Inicio (Linux / WSL)

Entorno comprobado: Python 3.13.5, NumPy 2.3.5, pytest 9.0.2, GCC 14.2.0, x86_64 Linux. Necesitas un compilador `cc` en PATH. La declaración Python >=3.11 expresa compatibilidad de sintaxis prevista; las pruebas de esta entrega se ejecutaron en 3.13.5.

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python examples/validate_book.py
python examples/verify_launch.py
python examples/train_decoder.py --steps 20 --gemm blocked --output reports/mi_prueba
```

Al ejecutar desde la raíz de `code` sin instalación editable, usa `PYTHONPATH=.` antes de cada orden. El runtime usa bibliotecas `.so`: no es una implementación nativa de DLL para Windows. No cambies ni instales controladores del sistema por ejecutar esta demo.

## Entrenamiento principal reproducible

```bash
python examples/train_decoder.py --backend cpu --steps 600 \
  --dim 64 --layers 2 --context 32 --batch 2 \
  --heads 2 --kv-heads 1 --hidden 128 \
  --online --gemm blocked --output reports/mi_decoder_600
```

El modelo tiene 75.584 parámetros sobre el vocabulario del corpus de demostración. El corpus repite seis frases originales: sirve para depurar, no para un benchmark lingüístico independiente. Hay 600 actualizaciones guardadas en `reports/decoder_final/report.json`. No se afirma calidad SOTA. El reporte histórico y el checkpoint de referencia se conservan; usa otra carpeta para tus ejecuciones.

Ruta bytes/BPE:

```bash
python examples/train_decoder.py --steps 20 --dim 16 --layers 1 --context 8 \
  --byte-tokens --bpe-merges 30 --gemm blocked --output reports/mi_bpe
```

Un archivo UTF-8 propio se proporciona mediante `--text archivo.txt`. Usa texto autorizado; diseña particiones por documentos para un experimento riguroso. La ruta BPE aprende fusiones después del corte; la ruta de caracteres de depuración construye el alfabeto de todo el texto. Los ejemplos no implementan pretokenización, tokens especiales ni el tokenizador exacto de un modelo comercial.

## Reanudar

`--steps` es el número de pasos adicionales, no el paso final absoluto. Repite las mismas opciones de arquitectura, tokenizador y corpus:

```bash
python examples/train_decoder.py --steps 4 --gemm blocked \
  --resume reports/parte/checkpoint.npz --output reports/reanudado
```

El checkpoint valida configuración y hash de corpus y conserva pesos, momentos, paso y RNG. El informe `reports/resume_check.json` documenta la equivalencia exacta observada entre diez pasos y seis más cuatro. Cambiar precisión, toolchain o hardware puede cambiar la reproducibilidad numérica.

## Medir

```bash
python examples/bench_kernels.py --repeat 20 --output reports/mi_benchmark.json
```

Cronometra `run` residente, sin compilación, carga de entradas ni copia de resultados. Incluye overhead Python de la llamada y, en su diseño general, sincronización. El informe original es CPU; la revisión de `reports/validation_20261001/` incluye CPU y CUDA. En GEMM el campo `shape` del JSON se ordena **[M,N,K]**: `[31,47,65]` significa A[31,65] y B[65,47]. En atención se usa [B,H,T,D]. NumPy es oráculo numérico, no un baseline de velocidad medido.

## GPU: CUDA validada en la revisión local

La revisión del 1 de octubre de 2026 utiliza Ubuntu 24.04 en WSL2, una NVIDIA GeForce RTX 5070 Ti Laptop de 12 GB, controlador 591.86, CUDA Toolkit 12.9.86 y destino `sm_120`. Los nuevos resultados están en `reports/validation_20261001/`; los informes originales se conservan. WMMA se compiló y ejecutó con error absoluto máximo cero; los tres ejemplos numéricos también pasaron en CUDA. HIP sigue pendiente de hardware compatible. No se ha medido competitividad frente a PyTorch.

La batería pasó 61 pruebas en CPU y 61 con `--backend=cuda`, sin omisiones. Las pruebas de estructuras del host, tokenización y convolución C independiente permanecen en CPU; los `Program` tensoriales seleccionan el dispositivo solicitado.

El entrenamiento principal de 75.584 parámetros completó 600 actualizaciones en CUDA: pérdida final 0,2508018, evaluación 0,3293002 y 32,493 segundos de bucle sin compilación. El reporte y checkpoint están en `reports/validation_20261001/decoder_cuda_75584/`. El decoder pequeño pasó además la comprobación exacta de reanudación de diez pasos frente a seis más cuatro.

El toolkit de esta comprobación está aislado en `~/.local/opt/cuda-12.9.1`, y el entorno Python en `~/.venvs/lumbre-validation-20261001`. Desde WSL y la raíz del proyecto:

```bash
source ~/.venvs/lumbre-validation-20261001/bin/activate
export PATH="$HOME/.local/opt/cuda-12.9.1/bin:$PATH"
export LUMBRE_CUDA_ARCH=sm_120
export LUMBRE_CACHE="$HOME/.cache/lumbre-mi-validacion"
cd code
python -m pytest -q --backend=cuda
python examples/validate_book.py --backend cuda --output reports/mis_ejemplos_cuda.json
python examples/bench_kernels.py --backend cuda --repeat 20 --output reports/mi_benchmark_cuda.json
nvcc -O2 -std=c++17 -arch=sm_120 kernels/wmma_demo.cu -o /tmp/wmma_demo
/tmp/wmma_demo
```

El toolkit procede de los [componentes oficiales de CUDA 12.9.1](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json), verificados por SHA-256. La [guía de NVIDIA](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-installation-guide-linux/index.html) describe la distribución por componentes. La revisión utiliza el controlador Windows existente.

Para CUDA, `LUMBRE_CUDA_ARCH` debe corresponder al dispositivo real y ser admitido por el toolkit (p.ej. `sm_120` solo donde corresponda). HIP requiere `hipcc` y una selección compatible del objetivo. Empieza con 257 elementos, después matrices irregulares y solo luego el decoder. Registra arquitectura, driver y herramientas. Consulta las fuentes oficiales enlazadas en el libro.

## Módulos y límites

- `ir.py`: FP32/int32 estático, DAG, operaciones y VJP simbólico.
- `compiler.py`: planificación, arena, C/CUDA/HIP, JIT y runtime. C y CUDA son los backends validados.
- `nn.py`: decoder y paso AdamW. No es un sistema de entrenamiento distribuido.
- `nnops.py`: convolución por composición para casos pequeños (máximo 1024 posiciones espaciales).
- `layout.py`, `symbolic.py`: laboratorios separados; no implican soporte de toda vista externa o forma dinámica en `Program`.
- `tokenizer.py`: ByteBPE educativo determinista.
- `passport.py`: verificación por enumeración de un mapa lógico, **no** simulador GPU ni prueba de carreras.

La atención online reduce intermediarios de forward; su VJP recompone operaciones densas, por lo que **no garantiza memoria lineal del backward**. El microkernel GEMM no incluye packing avanzado ni BLAS multihilo. No se implementan kernels FP8/FP4, ZeRO, MoE distribuido, KV cache incremental ni backward FlashAttention optimizado: esas extensiones están desarrolladas como temario/proyectos, no como capacidades ejecutadas de este código.

El JIT compila código nativo propio. No es un sandbox para ejecutar código de terceros no confiable. No cargues bibliotecas de una caché ajena. La caché por defecto está en `~/.cache/lumbre`; puedes usar `LUMBRE_CACHE=/ruta/de/prueba` para separar ensayos.

## Evidencias

`reports/environment.json`, `tests_final.log`, `book_examples.json`, `passport.json`, `kernels.json`, `resume_check.json`, `decoder_final/` y `decoder_bpe/`. Los tiempos son observaciones locales, no predicciones para otras máquinas. Las rutas de caché que puedan figurar en reportes históricos identifican el origen local; el programa crea las suyas al ejecutarse.
