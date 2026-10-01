# Resultados de esta edición

## Diez grupos de prácticas superados

La campaña ejecutó cinco grupos en CPU y cinco en CUDA: arrays, gradientes, regresión, atención causal y TinyJit. Ambos dispositivos completaron todas las comprobaciones. El entorno utiliza Python 3.12.3, NumPy 2.3.5, Clang 18 y el commit de tinygrad indicado al comienzo. CUDA usa NVCC 12.9.86 y la RTX 5070 Ti Laptop bajo WSL2.

| Dispositivo | Pérdida inicial | Pérdida tras 99 actualizaciones | Peso final | Bias final |
|---|---|---|---|---|
| CPU | 3.4375 | 1.87062e-13 | 2.999999 | -0.5 |
| CUDA | 3.4375 | 1.87062e-13 | 2.999999 | -0.5 |

Se realizaron cien actualizaciones. La última pérdida de la historia se leyó antes de la actualización número cien; los pesos de la tabla son posteriores a ella. La recarga de pesos reprodujo exactamente las predicciones de cada backend dentro de su propia ejecución.

## Comprobaciones numéricas y JIT

| Dispositivo | Error del gradiente analítico | Error de atención frente a NumPy | Mediana JIT (us) |
|---|---|---|---|
| CPU | 0 | 1.79e-07 | 134.792 |
| CUDA | 0 | 1.19e-07 | 121.169 |

El JIT se comprobó con cinco entradas diferentes antes de medir treinta llamadas calientes. Cada intervalo incluye llamada Python y sincronización; excluye transferencias de entrada y salida. La tabla acredita una medición local, no una comparación aislada de hardware. Los JSON conservan las muestras individuales.

Los informes de `tutorials/reports/tinygrad/CPU/` y `tutorials/reports/tinygrad/CUDA/` incluyen historias, errores y pesos. `tutorials/reports/campaign.json` registra el orden de ejecución y hashes del código. La atención comprueba un forward y causalidad respecto a V; no valida un decoder completo. La regresión comprueba aprendizaje de una función conocida, no calidad de un modelo de lenguaje.
