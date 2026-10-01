# Kernels portables y estado persistente: dos lecturas de DeepSeek {#ch:deepseek-estado}

Los capítulos anteriores permiten formular dos preguntas separadas: cómo ejecutar una multiplicación con eficiencia en otra arquitectura y cuánto estado conservar al generar texto. Las dos afectan al coste de un sistema, pero requieren contratos y pruebas diferentes. Este capítulo propone una lectura guiada y un experimento pequeño que puedes reproducir en CPU.

## DeepGEMM-Ascend: el contrato público y la representación física

DeepGEMM-Ascend se publicó para Ascend 950. Su documentación declara compatibilidad de API con DeepGEMM y kernels BF16, FP8 y FP4, además de operaciones de atención y MoE. Requiere CANN 9.20, `torch_npu`, Python y herramientas C++20. La revisión consultada es `8491bbb4b8c02a094a2318965f50c70438a3e73c`. [@deepgemm-ascend]

El repositorio advierte que los factores de escala UE8M0 se empaquetan por pares a lo largo de K dentro de enteros de 16 bits, con almacenamiento MN-major. Compartir una función Python no garantiza que el layout de sus buffers sea idéntico entre arquitecturas. [@deepgemm-ascend]

Aquí solo disponemos de una GPU NVIDIA. La lectura del código Ascend no constituye una ejecución de esos kernels. Las medidas publicadas por sus autores pertenecen a su plataforma; no se incorporan como resultados locales.

::: idea Separar tres contratos
El contrato matemático define formas, tipos y operación. El de almacenamiento añade strides, alineaciones y metadatos de cuantización. El de ejecución determina quién produce un buffer, quién lo consume y cuándo puede reutilizarse. Una API común necesita preservar los tres, aunque oculte decisiones del dispositivo.
:::

## Leer un kernel industrial sin empezar por todos sus detalles

Abre `deep_gemm/include/deep_gemm/bf16_gemm.hpp` en el commit citado. Sus parámetros separan dimensiones de bloque, dimensiones de las operaciones MAD y cantidades de etapas. El fuente declara almacenamiento L1 y L0, comprueba capacidades mediante `static_assert` y utiliza un scheduler para repartir bloques. El bucle de reducción coordina copias y multiplicación mediante esperas y notificaciones. [@deepgemm-bf16]

Relaciona esa estructura con la GEMM compartida del tutorial CUDA: primero localiza el dominio lógico [M,N,K]; después identifica dónde se coloca una pieza de A y B, quién la consume y qué autoriza a sobrescribirla. El código consultado distingue copias hacia L1, movimientos hacia L0 y acumulación matricial. El epílogo gestiona la salida y admite configuraciones diferentes. [@deepgemm-bf16]

No necesitas memorizar cada plantilla para dibujar una dependencia. Para un tile, marca tres estados: libre, cargado y consumido. Con dos buffers, el productor puede preparar el siguiente tile mientras el consumidor trabaja sobre el anterior, siempre que exista un protocolo de propiedad que impida adelantarse al uso del buffer.

```diagram
Cargar y publicar un tile | Esperar y multiplicar | Liberar para reutilizar
```

El archivo `ascend/sync.hpp` ofrece helpers de señales entre pipelines y barreras entre partes del kernel. Sus identificadores describen participantes del hardware Ascend. No deben sustituirse mecánicamente por una barrera CUDA de bloque: primero hay que reconstruir qué dependencia expresa cada señal. [@deepgemm-sync]

::: practica Diseñar el descriptor de un kernel externo
Imagina que Lumbre delega una GEMM a una biblioteca. Escribe los campos del descriptor: formas, dtype de entradas y acumulador, layout, stride, factores de escala, scratch necesario y stream o cola. Añade qué comprobación rechaza una entrada incompatible antes de llamar al kernel. La solución debe incluir también vida de los buffers y estado de error.
:::

## Presupuestar KV antes de hablar de compresión

En atención autoregresiva, guardar claves y valores anteriores evita recalcularlos en cada paso. Para un modelo hipotético con L capas, T tokens, H cabezas KV, dimensión D y b bits por elemento, el almacenamiento denso ideal de K y V es:

```math
B_{\mathrm{KV}}=2LTHD\frac{b}{8}.
```

El factor dos corresponde a K y V. Esta cuenta no incluye escalas, alineación, tablas de páginas ni buffers de trabajo. Tampoco representa automáticamente una arquitectura que comprime secuencias o comparte estado entre capas.

Para L=24, T=32768 y D=64, pasar de 16 a cuatro cabezas KV reduce por cuatro la cuenta ideal a igual precisión. Pasar después de 16 a cuatro bits reduce otra vez por cuatro el contenido numérico. El programa `tutorials/research/cache_lab.py` comprueba:

| Cabezas KV | Bits | Bytes ideales | MiB |
|---|---|---|---|
| 16 | 16 | 3221225472 | 3072 |
| 4 | 16 | 805306368 | 768 |
| 4 | 4 | 201326592 | 192 |

MiB significa $2^{20}$ bytes. La fila de cuatro bits es aritmética de almacenamiento, no una implementación FP4. Una cuantización real necesita especificar cómo codifica valores y escalas, cuánto error introduce y qué instrucciones realizan la conversión.

## Qué añade la lectura de DeepSeek-V4.1-Flash

El informe DeepSeek-V4.1-Flash, arXiv:2609.19969v1, combina una arquitectura causal encoder-decoder, reutilización de KV entre capas en CSA2 y caché FP4. Sus autores reportan 890 bytes por token para la caché global residente en HBM. Esa cifra no representa toda la memoria de inferencia. [@deepseek-v41]

El trabajo también propone SWA Bounded Replay: reconstruir estados de atención de ventana mediante un sufijo acotado y aceptar una aproximación. Los autores distinguen esa reconstrucción del estado exacto y señalan posibles límites en casos no evaluados. Nos interesa esa distinción entre ahorrar almacenamiento y preservar semántica exacta; aquí no reproducimos el modelo ni sus resultados. [@deepseek-v41]

Para tu compilador, separa la memoria de pesos, KV global, ventanas locales, activaciones temporales y metadatos. Anota además si un dato vive en GPU, host o almacenamiento persistente. Reducir una partida no elimina las otras. El coste de recuperar una conversación puede incluir transferencia y recomputación, aunque el siguiente token utilice pocos kernels.

## Un experimento de reconstrucción exacta y aproximada

Construiremos un sistema propio, mucho más pequeño que un Transformer. Cada capa sustituye un valor por la media causal de hasta W valores de la capa anterior. Con L capas, la última salida depende de un máximo de $1+L(W-1)$ entradas originales. Cada capa amplía el alcance hacia atrás en W-1 posiciones.

Para L=3 y W=4, el sufijo necesario para la última salida contiene diez entradas. El programa compara el resultado de una secuencia completa de 24 cuadrados con reconstruir solo sus diez últimas entradas o sus cuatro últimas. El operador y sus condiciones de borde se definen en el fuente; no simulan CSA2.

```bash
python tutorials/research/cache_lab.py
```

La secuencia completa y el sufijo de diez producen 384. El sufijo de cuatro produce aproximadamente 457,347222; su error absoluto es 73,347222. Las aserciones exigen identidad dentro de tolerancia para el sufijo suficiente y una diferencia observable para el sufijo corto.

El ejemplo muestra por qué el contexto necesario puede acumularse a través de capas. Una ventana local de cuatro no implica que un sufijo original de cuatro reconstruya el estado profundo. Si aceptas una aproximación, necesitas medir su impacto sobre las salidas que importan, no describirla como una optimización que conserva el programa exacto.

::: practica Una política de caché con pruebas
Supón que un runtime expulsa estados de una conversación y luego la reanuda. Define una prueba de salida exacta si promete reconstrucción exacta. Si utiliza una aproximación, registra el punto de reanudación, el contexto rehecho, el error frente a la referencia y una métrica de la tarea. Incluye casos donde cambie el punto de corte: una política puede comportarse de forma distinta en dos prefijos de la misma conversación.
:::

## Un proyecto que sí cabe en esta entrega

Puedes extender el decoder educativo con KV incremental exacta antes de explorar compresión. Empieza con una capa y pesos fijos. Compara los logits de cada posición calculados con el prefijo completo frente a los de la ruta incremental. Después añade capas, GQA y reanudación de la caché. Mantén el tamaño pequeño hasta entender las discrepancias.

La integración de DeepGEMM-Ascend requeriría hardware Ascend y una validación propia. El experimento de medias causales solo necesita NumPy. Los tutoriales CUDA y tinygrad aportan prácticas ejecutadas en los dispositivos disponibles; estos tres alcances quedan separados en los informes del repositorio.
