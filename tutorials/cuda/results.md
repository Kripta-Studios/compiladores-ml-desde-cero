# Resultados de esta edición

## Corrección observada

Los cuatro programas compilaron y terminaron con código cero: siete casos de vectores, siete de reducción, diez de GEMM y una cadena de stream. Los 25 casos pasaron sus comparaciones. La campaña secuencial y las versiones se conservan en `tutorials/reports/campaign.json`; los registros individuales están en `tutorials/reports/cuda/`.

## Intervalos de GEMM

Cada fila resume treinta muestras de cien lanzamientos residentes, con eventos CUDA. Los tiempos excluyen reservas, copias iniciales y referencia CPU. Son observaciones de este portátil bajo WSL; no se fijaron frecuencias ni se aisló la carga del escritorio.

| M,N,K | Variante | Mediana (us) | Mínimo (us) | Máximo (us) |
|---|---|---|---|---|
| 1,1,1 | directa | 14.980 | 11.519 | 27.921 |
| 1,1,1 | tile | 13.452 | 10.796 | 22.863 |
| 17,19,23 | directa | 12.557 | 9.618 | 21.535 |
| 17,19,23 | tile | 11.317 | 9.132 | 22.561 |
| 31,47,65 | directa | 11.487 | 8.646 | 18.793 |
| 31,47,65 | tile | 10.077 | 7.796 | 17.468 |
| 128,128,128 | directa | 10.408 | 9.161 | 11.997 |
| 128,128,128 | tile | 11.221 | 9.634 | 25.703 |
| 256,256,256 | directa | 29.201 | 25.146 | 40.934 |
| 256,256,256 | tile | 26.356 | 21.336 | 33.560 |

La dispersión forma parte del resultado. Una diferencia pequeña entre medianas frente a ese intervalo no sostiene una afirmación general sobre qué kernel gana. Para investigar rendimiento, repite la campaña con carga controlada y conserva el mismo contrato de medición.

## Límites y repetición

Las entradas elegidas facilitan verificar índices y sumas; no cubren todos los valores FP32. No se ejecutó Compute Sanitizer ni se midió solapamiento entre streams. Los tests acreditan estos programas y estos casos. Usa la orden del capítulo de instalación para crear tus propios registros.
