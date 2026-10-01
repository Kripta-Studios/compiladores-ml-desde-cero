# Laboratorio 5. GEMM: mejorar sin esconder el coste {#lab:cinco}

## Una pregunta experimental concreta

Queremos saber si reutilizar pequeños bloques mejora la multiplicación de matrices de Lumbre en nuestra CPU. La hipótesis no es «hemos inventado la multiplicación más rápida». Es: **para las formas elegidas, el microkernel bloqueado reduce el tiempo medido frente al recorrido de referencia, con el mismo contrato de entrada y una comprobación numérica explícita**.

El paquete incluye `examples/bench_kernels.py`. Su informe conserva muestras, no únicamente el mejor número. Empieza por leer qué se cronometra. El camino medido utiliza datos residentes; no incluye necesariamente construir el grafo, copiar entradas, compilar o guardar resultados. Esas fases deben medirse aparte cuando se evalúe la experiencia completa de una aplicación.

```bash
python examples/bench_kernels.py
```

El script escribe su informe según la ruta establecida en el propio archivo. Conserva una copia con fecha y descripción de la máquina antes de repetir pruebas que puedan sobrescribirlo.

## Cuenta de operaciones y bytes

Para matrices $M\times K$ y $K\times N$, la convención habitual cuenta aproximadamente $2MKN$ operaciones de coma flotante. Una suma y un producto se cuentan como dos, incluso si una instrucción FMA los combina. La cifra es una convención de rendimiento; no significa que el hardware ejecute exactamente ese número de instrucciones.

En una multiplicación $128^3$, son 4.194.304 operaciones. Los datos de entrada y salida ocupan $4(128^2+128^2+128^2)=196.608$ bytes en FP32, sin contar tráfico repetido, cachés ni almacenamiento auxiliar. Dividir las operaciones entre ese mínimo de bytes da una intensidad idealizada. No demuestra el tráfico real de una implementación que relee elementos muchas veces.

El microkernel del paquete conserva un bloque de acumuladores de $4\times8$ y recorre $K$ en tramos de 64. Es una decisión didáctica, no un parámetro universal. El compilador C puede vectorizar o no según el código y la plataforma. No se han añadido instrucciones intrínsecas específicas ni un sistema de packing comparable al de una BLAS madura.

## Comprobar bordes antes de medir cuadrados bonitos

Una matriz $31\times65$ multiplicada por otra $65\times47$ obliga a tratar bordes en filas, columnas y reducción. Los tamaños potencia de dos son útiles, pero pueden ocultar un `min` ausente o un acumulador mal inicializado. La batería del paquete incluye formas no alineadas y comparación de estrategias.

```python
import numpy as np
from lumbre import param, Program

rng = np.random.default_rng(5)
a = param("A", (31, 65))
b = param("B", (65, 47))
c = a @ b
A = rng.normal(size=a.shape).astype(np.float32)
B = rng.normal(size=b.shape).astype(np.float32)

resultados = []
for estrategia in ("reference", "blocked"):
    with Program([c], gemm=estrategia) as p:
        resultados.append(p.run({"A": A, "B": B})[0])
        print(p.report())
for resultado in resultados:
    np.testing.assert_allclose(resultado, A @ B,
                               rtol=3e-5, atol=1e-5)
```

NumPy sirve aquí como referencia de comprobación. No se está cronometrando NumPy como competidor, ni atribuyendo su backend a Lumbre. Si se quisiera comparar con una BLAS, habría que registrar también su implementación, número de hilos y configuración.

## Entender una medición local real

Las medianas observadas en esta edición para referencia y bloqueado fueron aproximadamente 51,4 y 11,4 microsegundos en la forma irregular; 1.566 y 157 microsegundos en $128^3$; 15.247 y 1.371 microsegundos en $256^3$. Son resultados de la CPU y el entorno del informe entregado. No son una predicción de tu portátil ni una comparación contra PyTorch.

El cociente referencia/bloqueado responde cuánto mejora esa sustitución dentro del ensayo. No se puede trasladar sin más al entrenamiento completo. Si GEMM ocupase solo la mitad del tiempo, incluso una aceleración enorme dejaría intacta la otra mitad. La ley de Amdahl del capítulo de medidas permite calcular esa limitación antes de hacer promesas.

## Experimento de una sola variable

Cambia un tamaño de tile en una copia del microkernel, no en cinco lugares a la vez. Para cada candidato ejecuta primero corrección; solo los correctos pasan al cronómetro. Conserva la configuración original como baseline. Si un candidato falla, guarda su fuente y el caso mínimo que lo contradice.

Una exploración responsable puede usar tres valores para filas, tres para columnas y tres para tramo de reducción: 27 candidatos. Si cada compilación tarda un segundo y cada validación medio segundo, el coste ya no es despreciable. El autotuning debe contar esos intentos. Una mejora de un microsegundo que requiere un minuto de búsqueda necesita muchas ejecuciones para amortizarse.

::: practica Explica un resultado contrario a tu intuición
Aumentar el tile de salida reduce el número de iteraciones exteriores, pero empeora el tiempo. Propón dos mecanismos diferentes.

**Solución:** puede aumentar la presión de registros y provocar spills; también puede dificultar la vectorización o aumentar trabajo inútil en bordes. Otra posibilidad es que la medida incluya más compilación. Ninguna se acepta solo por parecer plausible: inspecciona ensamblador, recursos o fases de tiempo para distinguirlas.
:::

## Condición de entrega

Incluye una tabla con forma, estrategia, error máximo, mediana y número de muestras. Añade condiciones de compilación, hilos y datos residentes. La conclusión debe nombrar la referencia exacta. «Más rápido que nuestro bucle de referencia» es una afirmación útil y demostrable; «SOTA CPU» requiere una comparación mucho más amplia que este laboratorio no ha realizado.

# Laboratorio 6. Convolución desde índices hasta gradientes {#lab:seis}

## Una imagen diminuta y un filtro que puedes calcular

Trabajaremos con un solo canal, una imagen $3\times3$ y un filtro $2\times2$, sin padding y con stride uno. La imagen contiene los números del 1 al 9 por filas. El filtro es `[[1,0],[0,-1]]`. Cada salida resta la esquina inferior derecha de la superior izquierda de una ventana.

Las cuatro ventanas producen -4. La forma de salida es $2\times2$, porque el filtro puede comenzar en dos filas y dos columnas. No se invierten los coeficientes del filtro: la operación de las redes se formula aquí como correlación cruzada, siguiendo el contrato declarado en el capítulo de convoluciones.

```python
import numpy as np
from lumbre import param, Program, gradients
from lumbre.nnops import conv2d

x = param("imagen", (1, 1, 3, 3))
w = param("filtro", (1, 1, 2, 2))
y = conv2d(x, w)
perdida = y.sum()
gx, gw = gradients(perdida, [x, w])
X = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
W = np.array([1, 0, 0, -1], dtype=np.float32).reshape(1, 1, 2, 2)
with Program([y, gx, gw], gemm="blocked") as p:
    Y, dX, dW = p.run({"imagen": X, "filtro": W})
np.testing.assert_array_equal(Y, -4 * np.ones((1, 1, 2, 2)))
print(dX)
print(dW)
```

La implementación compuesta `nnops.conv2d` construye parches con operaciones ya presentes en la IR y usa multiplicaciones de matrices. Por tanto, los gradientes proceden de las reglas de esas operaciones, no de una función de convolución backward tomada de otro framework.

## Resolver el gradiente del filtro a mano

Como la pérdida suma todas las salidas, cada salida aporta uno. El gradiente de un coeficiente del filtro es la suma de los píxeles que multiplicó. Para la esquina superior izquierda: $1+2+4+5=12$. Para la superior derecha: $2+3+5+6=16$. Para la inferior izquierda: $4+5+7+8=24$. Para la inferior derecha: $5+6+8+9=28$.

El gradiente esperado es `[[12,16],[24,28]]`. No depende aquí del valor actual del filtro, porque la salida es lineal en sus coeficientes y la pérdida es una suma. Cambiar la pérdida a cuadrados introduciría otra dependencia y cambiaría el resultado.

El gradiente de la imagen reúne filtros solapados. El píxel central participa como esquina inferior derecha de una ventana y superior izquierda de otra, con contribuciones -1 y +1 que se cancelan. La matriz resultante es `[[1,1,0],[1,0,-1],[0,-1,-1]]`. Esta cancelación es una buena prueba: un scatter que sobrescriba en lugar de sumar puede devolver un valor distinto de cero en el centro.

## Dos implementaciones con propósitos distintos

El archivo `kernels/conv.c` contiene una convolución directa con grupos, padding simétrico, stride y dilatación. Su prueba compila C y compara con un recorrido de referencia. `nnops.conv2d` expresa la operación mediante la IR para obtener autodiferenciación por composición. No hay que confundirlas: la primera es un kernel C independiente; la segunda conecta con el compilador y su grafo de gradientes.

La versión compuesta tiene una cota explícita de 1.024 posiciones espaciales de salida para impedir la construcción accidental de grafos enormes. Su estructura es apropiada para estudiar corrección y derivación, no para afirmar rendimiento de una implementación im2col optimizada. Una mejora futura consiste en representar la extracción de parches con un operador de índices compacto o generar directamente un kernel de convolución.

## Padding y dilatación en coordenadas

Para una salida $(o_y,o_x)$ y una posición del filtro $(k_y,k_x)$, la entrada correspondiente es $(o_y s_y-p_y+k_y d_y,\;o_x s_x-p_x+k_x d_x)$. Cada término tiene una función: stride desplaza el comienzo de ventana; padding cambia el origen lógico; dilatación separa muestras dentro del filtro.

No memorices la fórmula sin probarla. Con stride dos, padding uno y dilatación uno, la primera ventana comienza en coordenada -1. Algunas posiciones quedan fuera y aportan cero. La segunda ventana comienza dos lugares después. Con dilatación dos, un filtro de tres elementos cubre un intervalo de cinco posiciones: no toma las cinco, sino tres separadas.

::: practica Grupos y forma de los pesos
Una entrada tiene cuatro canales y una salida seis. Se usan dos grupos. ¿Cuántos canales de entrada ve cada canal de salida? ¿Qué forma de pesos corresponde a filtros $3\times3$?

**Solución:** cada grupo recibe dos canales de entrada y produce tres de salida. Los pesos tienen forma `(6,2,3,3)`, no `(6,4,3,3)`. Los tres primeros canales de salida se conectan con el primer grupo de entrada y los otros tres con el segundo. La divisibilidad de canales de entrada y salida por grupos se valida antes de compilar.
:::

## Prueba numérica de un gradiente

Escoge un coeficiente, súmale y réstale una pequeña cantidad y calcula la diferencia central de pérdidas. Compara con el gradiente simbólico. Evita perturbaciones tan pequeñas que FP32 las redondee al mismo valor, y no confundas una coincidencia numérica en un coeficiente con la validación de todo el operador.

Las pruebas del paquete usan varias configuraciones de grupos, stride, padding y dilatación, además de perturbaciones de entradas, pesos y sesgo. El resultado local es evidencia de esas pruebas. El argumento general de composición explica por qué el mismo mecanismo produce contribuciones de todos los caminos, siempre que cada primitiva y su regla de gradiente sean correctas.

## Entrega

Incluye las cuatro salidas, los dos gradientes calculados a mano y la comparación ejecutada. Después añade un caso con padding y otro con grupos. Señala qué implementación utilizaste y qué cota impide escalarla directamente. El laboratorio se completa comprendiendo el solapamiento; no basta con obtener la misma forma que una biblioteca externa.

# Laboratorio 7. Llevar el mismo grafo a CUDA o HIP {#lab:siete}

::: cuidado Estado de validación de esta edición
Este laboratorio requiere una GPU y su toolchain. La revisión del 1 de octubre valida CUDA en una RTX 5070 Ti Laptop con WSL2, toolkit 12.9 y destino `sm_120`; HIP sigue pendiente. Los informes están en `code/reports/validation_20261001/`. Las salidas siguientes definen los requisitos de aceptación que debes volver a comprobar en tu dispositivo.
:::

## Preparar un inventario, no una instalación a ciegas

Antes de ejecutar, registra sistema operativo, dispositivo exacto, controlador, compilador y arquitectura destino. En CUDA, consulta la compute capability del modelo concreto. La tabla oficial distingue, por ejemplo, RTX 50 y las distintas GPUs Blackwell de centro de datos; compartir un nombre de generación no garantiza el mismo conjunto de instrucciones. En HIP, consulta la compatibilidad del dispositivo y del entorno en la documentación vigente. [@cuda-gpus;@hip]

En Windows, la guía CUDA para WSL explica un entorno específico; no debe mezclarse sin más con instrucciones de instalación de un controlador Linux físico. El runtime de Lumbre usa una biblioteca compartida POSIX. Comprueba primero un ejemplo del toolchain, después el programa elemento a elemento y solo entonces una matriz o un modelo. [@cuda-wsl]

## Primera prueba: 257 elementos

Usa un tamaño que no sea múltiplo del bloque de lanzamiento. El siguiente programa selecciona el backend por una variable de entorno y conserva exactamente el grafo de la CPU.

```python
import os
import numpy as np
from lumbre import param, Program

backend = os.environ.get("LUMBRE_BACKEND", "cpu")
x = param("x", (257,))
y = x * 3.0 + 1.0
X = np.arange(257, dtype=np.float32)
with Program([y], backend=backend) as p:
    Y = p.run({"x": X})[0]
    np.testing.assert_array_equal(Y, X * 3 + 1)
    print(p.report())
```

Para una GPU NVIDIA con destino real `sm_120`, el entorno podría configurarse así; ese valor es un ejemplo condicionado al dispositivo y a soporte del toolkit, no un valor universal.

```bash
export LUMBRE_BACKEND=cuda
export LUMBRE_CUDA_ARCH=sm_120
python laboratorio_gpu.py
```

Para HIP, selecciona `LUMBRE_BACKEND=hip` y un entorno donde `hipcc` conozca la arquitectura apropiada. El ejemplo entregado depende de la selección del toolchain; un port de producción debe hacer explícita esa arquitectura también en el manifiesto y en la clave de caché.

## Leer el kernel generado

Para contrastar la prueba aislada con la batería completa, ejecuta `python -m pytest -q --backend=cuda` desde `code/`. El capítulo «CPU y CUDA en la misma máquina» desarrolla el protocolo de comparación, muestra las medidas y conserva la distinción entre las pruebas tensoriales de GPU y las pruebas que siguen ejecutándose en el host.

Busca el índice global construido a partir de bloque e hilo. Después localiza la condición que protege `q<257`. Si el grid contiene más posiciones, no es necesariamente un error: el redondeo del número de bloques necesita la máscara. El error sería ejecutar cargas o escrituras de los hilos sobrantes sin protección.

Localiza también las llamadas del host. Una función `__global__` describe trabajo del dispositivo; la sintaxis de lanzamiento y el runtime son otra parte. El programa debe comprobar errores del lanzamiento y esperar antes de leer la salida. El runtime educativo sincroniza para simplificar la observación, lo que es correcto pero limita concurrencia.

## Segunda prueba: GEMM con memoria compartida

La ruta GPU de Lumbre incluye un GEMM por tiles de $16\times16$. Cada etapa carga regiones compartidas, sincroniza, acumula y vuelve a sincronizar antes de reutilizarlas. La segunda barrera es tan importante como la primera: impide sobrescribir una etapa que otros hilos todavía leen.

Prueba formas `(17,19) @ (19,23)` y luego una multiplicación por lotes. Cambia datos y repite muchas veces. Una sola ejecución puede no revelar una carrera. Utiliza las herramientas de diagnóstico del proveedor disponibles para tu versión y conserva sus opciones y resultados; no inventes un informe de ausencia de carreras a partir de una comparación numérica aislada.

Un hilo cuya salida está fuera de la matriz puede seguir necesitando participar en cargas y barreras del bloque. Por eso una salida temprana antes de una barrera colectiva puede ser incorrecta. El diseño debe permitir que todos los participantes requeridos sigan el protocolo, aunque algunos valores se enmascaren.

## Tercera prueba: WMMA separado del compilador

`kernels/wmma_demo.cu` es un ejemplo completo para estudiar una operación matricial colectiva con entradas FP16 y acumulación FP32. Compílalo para una arquitectura compatible y revisa todas las comprobaciones del host. Que este ejemplo funcione no demuestra que el renderer tensorial emita automáticamente tensor cores para todo GEMM: son hitos diferentes.

La entrega del laboratorio debe separar «he ejecutado el ejemplo WMMA» de «he integrado una transformación que reconoce GEMM y produce el layout WMMA correcto». La segunda requiere inspeccionar los operands y probar bordes, transpuestas y lotes. La mera presencia de la cadena `wmma` en un archivo no demuestra uso correcto en el entrenamiento.

## Depuración en orden

Si no compila, conserva el archivo generado y el error exacto. Si no se puede cargar, revisa ABI y dependencias. Si el dispositivo rechaza el lanzamiento, revisa arquitectura, dimensiones y recursos. Si los valores son erróneos, empieza con tamaños pequeños y patrones distinguibles. Si solo falla al repetir, investiga vida de memoria y sincronización.

::: practica Un resultado correcto que no valida rendimiento
El kernel devuelve lo esperado, pero cada operación copia datos host-dispositivo y sincroniza globalmente. ¿Puede declararse competitivo frente a un entrenamiento con datos residentes?

**Solución:** no con esa comparación. Primero hay que igualar la frontera medida. Puede publicarse el tiempo extremo a extremo del programa educativo y el tiempo del kernel residente como métricas distintas. Ocultar las copias en un lado y contarlas en el otro sesga la conclusión.
:::

# Laboratorio 8. Autodiferenciación con una cuenta que puedes verificar {#lab:ocho}

## Una regresión de dos parámetros

Queremos ajustar una recta $\hat y=wx+b$ a tres parejas: $(0,1)$, $(1,3)$ y $(2,5)$. La solución exacta es $w=2,b=1$. Empezaremos con ambos a cero y usaremos la media del error cuadrático. El objetivo inmediato no es converger: es comprobar el primer gradiente.

Al principio, los errores son $(-1,-3,-5)$. La pérdida vale $(1+9+25)/3=35/3$. El gradiente de $w$ es $\frac23\sum_i e_ix_i=-26/3$; el de $b$ es $\frac23\sum_i e_i=-6$. Con tasa 0,1, un paso debe producir aproximadamente $w=0,8666667$ y $b=0,6$.

## Construir el paso como otro grafo

```python
import numpy as np
from lumbre import param, gradients, Program

x = param("x", (3,))
t = param("objetivo", (3,))
w = param("w", ())
b = param("b", ())
error = x * w + b - t
perdida = (error * error).mean()
gw, gb = gradients(perdida, [w, b])
nuevo_w, nuevo_b = w - 0.1 * gw, b - 0.1 * gb

with Program([perdida, gw, gb, nuevo_w, nuevo_b]) as p:
    valores = p.run({"x": [0, 1, 2], "objetivo": [1, 3, 5],
                     "w": np.float32(0), "b": np.float32(0)})
    np.testing.assert_allclose(valores[1], -26/3, rtol=1e-6)
    np.testing.assert_allclose(valores[2], -6, rtol=1e-6)
    p.assign({"w": nuevo_w, "b": nuevo_b})
    siguiente = p.run(read=[perdida])[0]
    print(float(valores[0]), float(siguiente))
```

El gradiente se genera como operaciones de la misma IR. El compilador no llama a `torch.autograd`. Al aplicar `assign`, los nuevos parámetros se copian después de calcular todas las salidas; el segundo `run` usa el estado actualizado y conserva los datos que ya estaban inicializados.

## Por qué una contribución no puede sobrescribir otra

En `error*error`, el mismo nodo aparece en dos aristas del producto. Cada arista aporta una derivada. La suma de ambas produce el factor dos. Un algoritmo que marque un nodo como visitado y descarte automáticamente la segunda contribución obtendría la mitad del gradiente.

El orden topológico inverso sirve para esperar a que estén reunidas las contribuciones antes de propagar a los padres. No significa que cada arista tenga una única ruta hasta la pérdida. Los parámetros compartidos en un decoder producen el mismo problema a mayor escala: una tabla usada como embedding y como proyección de salida recibe contribuciones de ambos usos.

## Broadcasting en el backward

`b` es escalar, pero participa en tres predicciones. La derivada respecto a su valor es la suma de las tres contribuciones, dividida por tres por la media. La forma final debe ser `()`, no `(3,)`. `w` también es escalar, aunque cada contribución lleve multiplicado un dato distinto.

Una prueba de formas es tan importante como una de valores. Un gradiente con números plausibles y forma equivocada puede difundirse por broadcasting en la actualización y producir un programa que no representa los parámetros originales. El optimizador debe validar las formas antes de comprometer el estado.

## Diferencias finitas: escoger una perturbación razonable

Calcula la pérdida en $w+\delta$ y $w-\delta$, dejando lo demás fijo. La diferencia dividida entre $2\delta$ aproxima la derivada. Prueba varios valores de $\delta$, como $10^{-2}$ y $10^{-3}$, y observa cuándo el error deja de mejorar. En FP32, hacerlo arbitrariamente pequeño puede aumentar el error por cancelación y redondeo.

No utilices diferencias finitas en un punto no diferenciable esperando una derivada única. Para máximo con empate, el libro declara una convención de reparto. Una perturbación que rompe el empate puede observar una pendiente unilateral diferente sin demostrar que el código incumpla esa convención.

::: practica Diagnóstico de un gradiente reducido a la mitad
El gradiente calculado de `w` es aproximadamente -4,3333 en lugar de -8,6667, mientras que la pérdida es correcta. ¿Qué parte investigarías primero?

**Solución:** la acumulación de contribuciones del producto `error*error`. Si se propaga solo por una arista, se pierde el factor dos. También hay que descartar otra media introducida por error, pero la relación exacta de la discrepancia y el nodo compartido proporcionan una hipótesis concreta para empezar.
:::

## Entrega y ampliación

Entrega la cuenta del primer paso, la ejecución y una curva de pérdidas registrada durante varios pasos. La curva puede no ser monótona con cualquier tasa; no la alteres para que parezca ideal. Añade una prueba donde `w` se use en dos expresiones distintas y demuestra que recibe ambas contribuciones. Después cambia a AdamW utilizando el módulo del paquete y comprueba que guardas también sus momentos, no solo los pesos.
