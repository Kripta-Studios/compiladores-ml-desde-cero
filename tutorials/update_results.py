"""Regenerate tutorial tables from the checked-in, successful campaign."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tutorials'
campaign=json.loads((BASE/'reports/campaign.json').read_text())
assert all(item['returncode']==0 for item in campaign['sequence'])
for relative,digest in campaign['sources'].items():
    assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative

cuda=json.loads((BASE/'reports/cuda/results.json').read_text())
assert sum(p['passing_cases'] for p in cuda['programs'])==25
text='''# Resultados de esta edición

## Corrección observada

Los cuatro programas compilaron y terminaron con código cero: siete casos de vectores, siete de reducción, diez de GEMM y una cadena de stream. Los 25 casos pasaron sus comparaciones. La campaña secuencial y las versiones se conservan en `tutorials/reports/campaign.json`; los registros individuales están en `tutorials/reports/cuda/`.

## Intervalos de GEMM

Cada fila resume treinta muestras de cien lanzamientos residentes, con eventos CUDA. Los tiempos excluyen reservas, copias iniciales y referencia CPU. Son observaciones de este portátil bajo WSL; no se fijaron frecuencias ni se aisló la carga del escritorio.

| M,N,K | Variante | Mediana (us) | Mínimo (us) | Máximo (us) |
|---|---|---|---|---|
'''
rows=[]
for line in (BASE/'reports/cuda/03_gemm.log').read_text().splitlines():
    fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
    if not fields: continue
    assert 'PASS' in line
    rows.append(fields)
    text+=f"| {fields['m']},{fields['n']},{fields['k']} | {'tile' if fields['tiled']=='1' else 'directa'} | {float(fields['median_us']):.3f} | {float(fields['min_us']):.3f} | {float(fields['max_us']):.3f} |\n"
assert len(rows)==10
text+='''
La dispersión forma parte del resultado. Una diferencia pequeña entre medianas frente a ese intervalo no sostiene una afirmación general sobre qué kernel gana. Para investigar rendimiento, repite la campaña con carga controlada y conserva el mismo contrato de medición.

## Límites y repetición

Las entradas elegidas facilitan verificar índices y sumas; no cubren todos los valores FP32. No se ejecutó Compute Sanitizer ni se midió solapamiento entre streams. Los tests acreditan estos programas y estos casos. Usa la orden del capítulo de instalación para crear tus propios registros.
'''
(BASE/'cuda/results.md').write_text(text,encoding='utf-8',newline='\n')

reports={d:json.loads((BASE/'reports/tinygrad'/d/'results.json').read_text()) for d in ('CPU','CUDA')}
text='''# Resultados de esta edición

## Diez grupos de prácticas superados

La campaña ejecutó cinco grupos en CPU y cinco en CUDA: arrays, gradientes, regresión, atención causal y TinyJit. Ambos dispositivos completaron todas las comprobaciones. El entorno utiliza Python 3.12.3, NumPy 2.3.5, Clang 18 y el commit de tinygrad indicado al comienzo. CUDA usa NVCC 12.9.86 y la RTX 5070 Ti Laptop bajo WSL2.

| Dispositivo | Pérdida inicial | Pérdida tras 99 actualizaciones | Peso final | Bias final |
|---|---|---|---|---|
'''
for dev,report in reports.items():
    row=report['lessons']['regression']
    text+=f"| {dev} | {row['initial_loss']:.6g} | {row['final_loss']:.6g} | {row['w'][0][0]:.7g} | {row['b'][0]:.7g} |\n"
text+='''
Se realizaron cien actualizaciones. La última pérdida de la historia se leyó antes de la actualización número cien; los pesos de la tabla son posteriores a ella. La recarga de pesos reprodujo exactamente las predicciones de cada backend dentro de su propia ejecución.

## Comprobaciones numéricas y JIT

| Dispositivo | Error del gradiente analítico | Error de atención frente a NumPy | Mediana JIT (us) |
|---|---|---|---|
'''
for dev,report in reports.items():
    r=report['lessons']
    text+=f"| {dev} | {r['gradients']['analytical_error']:.3g} | {r['attention']['numpy_error']:.3g} | {r['jit']['median_seconds']*1e6:.3f} |\n"
text+='''
El JIT se comprobó con cinco entradas diferentes antes de medir treinta llamadas calientes. Cada intervalo incluye llamada Python y sincronización; excluye transferencias de entrada y salida. La tabla acredita una medición local, no una comparación aislada de hardware. Los JSON conservan las muestras individuales.

Los informes de `tutorials/reports/tinygrad/CPU/` y `tutorials/reports/tinygrad/CUDA/` incluyen historias, errores y pesos. `tutorials/reports/campaign.json` registra el orden de ejecución y hashes del código. La atención comprueba un forward y causalidad respecto a V; no valida un decoder completo. La regresión comprueba aprendizaje de una función conocida, no calidad de un modelo de lenguaje.
'''
(BASE/'tinygrad/results.md').write_text(text,encoding='utf-8',newline='\n')
print('Tutorial tables regenerated from successful reports.')
