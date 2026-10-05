# Tutoriales prácticos de CUDA y tinygrad

Continuación de *Compiladores de ML desde cero*. Los dos cuadernos explican las APIs que introducen, incluyen ejercicios con solución y reproducen el código completo de sus prácticas.

- [Tutorial CUDA](../Tutorial_CUDA_Desde_Cero_2026.pdf): índices, memoria, reducción, GEMM, medidas y streams.
- [Tutorial tinygrad](../Tutorial_Tinygrad_Desde_Cero_2026.pdf): tensores, gradientes, regresión, atención causal y TinyJit.
- [Fuentes CUDA](cuda/) y [fuentes tinygrad](tinygrad/).
- [Evidencias locales](reports/).

La edición ampliada del **5 de octubre de 2026** incorpora el syllabus completo de *Compilers for Machine Learning*. CUDA añade dos capítulos sobre rangos, dependencias, kernels y medición dentro del modelo. tinygrad añade dos capítulos de contraste con el compilador propio, desde UOps hasta MNIST, formas simbólicas y reanudación. Los nuevos ejercicios distinguen trabajo propuesto de las prácticas ya ejecutadas.

## CUDA

Desde la raíz del repositorio, con `nvcc` en PATH y un controlador NVIDIA operativo:

```bash
python3 tutorials/validate_cuda.py --arch sm_120 --output tutorials/reports/mi_cuda
```

Selecciona el destino que corresponda a tu GPU. La campaña publicada utiliza CUDA 12.9.86, RTX 5070 Ti Laptop y WSL2. El ejecutor elimina sus binarios temporales.

## tinygrad

Usa un entorno separado de Lumbre. Instala las dependencias fijadas y Clang para CPU. Para CUDA, incluye el toolkit en PATH. En esta campaña utilizamos `clang-18` y el renderer NVCC; no hace falta descargar pesos.

```bash
python -m pip install -r tutorials/requirements-tinygrad.txt
export DEV='CPU:CLANG;CUDA:NVCC'
export CC=clang-18
python tutorials/tinygrad/labs.py --device CPU --output tutorials/reports/mi_tiny_cpu
python tutorials/tinygrad/labs.py --device CUDA --output tutorials/reports/mi_tiny_cuda
```

Con ese entorno y CUDA preparados, `python tutorials/validate_all.py --output tutorials/reports/mi_campana` ejecuta CUDA, tinygrad CPU y tinygrad CUDA en secuencia y guarda el inventario del entorno.

Los cinco grupos de prácticas comprueban resultados contra referencias. El archivo NPZ guarda pesos de regresión, no un checkpoint completo de entrenamiento. Las APIs corresponden al commit fijado de tinygrad, cuya versión declarada es 0.14.0.

## Lectura de DeepSeek y experimento independiente

```bash
python tutorials/research/cache_lab.py
```

Este programa comprueba cuentas de almacenamiento KV y reconstrucción de una cadena de medias causales. No implementa DeepSeek-V4.1-Flash ni ejecuta DeepGEMM-Ascend. El capítulo 70 del libro principal distingue lectura de fuentes, derivaciones y pruebas locales.

## Reconstrucción

Edita `cuda/tutorial.md`, `tinygrad/tutorial.md` o los respectivos `syllabus.md`; los capítulos de resultados proceden de los informes. El constructor comparte estilos y parser con el libro y genera fuentes LaTeX autocontenidos:

```bash
python build_tutorials.py --pdf
python build_book.py --pdf --report tutorials/reports/main_build.json
```

Para actualizar las tablas desde la campaña de referencia, ejecuta antes `python tutorials/update_results.py`. El script comprueba los hashes de las prácticas antes de generar los capítulos de resultados.

Ambos constructores compilan en carpetas temporales y eliminan auxiliares al terminar. Las prácticas originales de esta carpeta se distribuyen bajo [MIT](LICENSE). Las referencias externas conservan sus licencias; el texto de los libros no se relicencia mediante ese archivo.

Para auditar los tres PDF y comprobar que sus listados coinciden con el código, instala PyMuPDF en el entorno de revisión y ejecuta `python tutorials/audit_documents.py --render`. Guarda los renders en `tmp/pdfs/` y actualiza los informes estructurados; la inspección visual se realiza después y se registra por separado.
