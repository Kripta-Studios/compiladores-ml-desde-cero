# Laboratorio 9. Un decoder completo y un checkpoint que realmente reanuda {#lab:nueve}

## Qué vamos a demostrar

En este laboratorio se ejecuta entrenamiento real: se evalúa un decoder, se calcula una pérdida de siguiente token, se construyen gradientes y se actualizan parámetros. El tamaño es pequeño para que sea inspeccionable y ejecutable en CPU. No vamos a llamar «modelo de frontera» a un decoder de decenas de miles de parámetros, ni a confundir una pérdida baja sobre frases repetidas con comprensión general del lenguaje.

La arquitectura del paquete reúne componentes estudiados antes: embedding, RMSNorm, posiciones RoPE, atención causal con cabezas de consulta y de clave/valor, bloque SwiGLU, conexiones residuales y proyección de salida con pesos compartidos. La implementación exacta y sus formas aparecen en el módulo `nn.py` y en los capítulos del modelo. Los artículos de esos componentes son referencias del diseño, no una certificación de calidad del modelo entrenado aquí. [@gqa;@rope;@rmsnorm;@swiglu]

## Primera ejecución pequeña

Empieza por una configuración breve para verificar instalación, compilación y guardado. No aumentes a la vez dimensiones, capas y contexto: si falla, perderás la pista de qué recurso o contrato cambió.

```bash
python examples/train_decoder.py --backend cpu --steps 20 \
  --dim 16 --layers 1 --context 8 --batch 2 \
  --heads 2 --kv-heads 1 --gemm blocked \
  --output reports/mi_decoder_inicial
```

Abre `config.json` y comprueba que los valores corresponden a la orden. Después abre `report.json`: debe haber pérdidas finitas, un informe de compilación y una muestra. Que la muestra sea mala no implica automáticamente que el compilador falle; que parezca una frase tampoco demuestra corrección. Primero se validan operadores, causalidad, gradientes y actualización.

## Reproducir la configuración principal

La ejecución principal de esta edición utilizó la siguiente configuración. Los tiempos pueden variar entre máquinas y según exista una entrada válida en caché.

```bash
python examples/train_decoder.py --backend cpu --steps 600 \
  --dim 64 --layers 2 --context 32 --batch 2 \
  --heads 2 --kv-heads 1 --hidden 128 \
  --online --gemm blocked --output reports/mi_decoder_600
```

En el registro entregado hay 75.584 parámetros y 600 actualizaciones. La primera pérdida fue aproximadamente 3,2164 y la última 0,2747. La evaluación de cuatro lotes del fragmento reservado produjo aproximadamente 0,3322. El corpus de demostración repite seis frases: por tanto, esa evaluación no es un benchmark lingüístico independiente y puede compartir patrones casi idénticos con entrenamiento.

El tiempo registrado del bucle de entrenamiento fue aproximadamente 3,44 segundos en el entorno local, excluyendo compilación, evaluación posterior y generación. La compilación registrada en ese ensayo fue un acierto de caché. No se deben sumar o comparar esas cifras como si midieran una ejecución limpia completa en cualquier equipo.

## Revisar un token de supervisión

Imprime una ventana de entrada y su objetivo. El objetivo debe estar desplazado una posición. Si la entrada contiene los tokens de «el sol», la posición del primer token intenta predecir el segundo, no reconstruirse a sí misma. Un fallo de desplazamiento puede producir una pérdida excelente para una tarea equivocada.

La máscara causal impide usar el futuro de la ventana para predecir el siguiente token. Para probarla, cambia tokens posteriores a una posición y compara sus logits anteriores. No deberían cambiar, salvo la tolerancia numérica pertinente. Esta prueba comprueba causalidad del programa; no mide habilidad de lenguaje.

## La prueba fuerte de reanudación

Ejecuta diez pasos continuos en una carpeta. En otra, ejecuta seis pasos y reanuda cuatro más desde su checkpoint. Usa exactamente la misma configuración y corpus. El paquete guarda pesos, momentos del optimizador, número de paso, vocabulario/configuración y estado del generador aleatorio de lotes.

```bash
python examples/train_decoder.py --steps 10 --gemm blocked \
  --output reports/continuo
python examples/train_decoder.py --steps 6 --gemm blocked \
  --output reports/parte
python examples/train_decoder.py --steps 4 --gemm blocked \
  --resume reports/parte/checkpoint.npz --output reports/reanudado
```

Compara los archivos con `np.load(..., allow_pickle=False)`. Los campos numéricos deben coincidir para esta ejecución determinista en el mismo entorno. En la validación local, la comparación de diez pasos frente a seis más cuatro fue exacta en los campos guardados. En otros dispositivos o algoritmos con reducciones no deterministas, la especificación debe declarar si espera identidad bit a bit o una equivalencia tolerada.

Guardar únicamente los pesos no basta para esa prueba. AdamW necesita momentos y correcciones dependientes del paso. El muestreo de lotes necesita su estado aleatorio. Si alguno falta, se puede continuar entrenando, pero no se está reproduciendo la misma trayectoria.

## Interpretar fallos con orden

Si la pérdida no baja, primero intenta sobreajustar un único lote pequeño. Si tampoco funciona, revisa gradientes y actualización. Si solo falla con varias ventanas, revisa datos y objetivos. Si aparece NaN, guarda el primer paso afectado y localiza la primera operación no finita. No cambies simultáneamente tasa, precisión, arquitectura y datos esperando que desaparezca el síntoma.

::: practica Un checkpoint que parece completo
El archivo contiene todos los parámetros y el número de paso, pero omite los momentos de AdamW. ¿Por qué reanudar con el mismo número de paso no arregla el problema?

**Solución:** las correcciones del paso y las medias móviles describen conjuntamente el estado del optimizador. Usar el paso antiguo con momentos reiniciados combina dos historias incompatibles. Puede ser una decisión intencional de reinicio del optimizador, pero debe declararse y no presentarse como una reanudación exacta.
:::

# Laboratorio 10. Atención online: misma función, otra memoria {#lab:diez}

## La recurrencia que queremos probar

Una fila de atención combina valores con pesos proporcionales a exponenciales de puntuaciones. La implementación densa guarda puntuaciones o probabilidades de toda la fila. La online conserva un máximo, un denominador y una suma ponderada que se actualizan al recibir cada bloque. Su interés es evitar una matriz intermedia cuadrática; no cambia la definición de la atención. [@flash1]

Empieza con un ejemplo donde puedas usar fracciones exactas. Puntuaciones $(0,\log2,\log4)$ y valores $(1,3,5)$ producen pesos proporcionales a $(1,2,4)$. El resultado es $(1+6+20)/7=27/7$. Cuando cambia el máximo, las contribuciones anteriores se reescalan para seguir expresadas respecto al mismo origen.

## Un programa de comparación

```python
import numpy as np
from lumbre import param, Program, softmax, online_attention, gradients
from lumbre.ir import node

forma = (1, 2, 7, 4)
q, k, v = [param(nombre, forma) for nombre in ("q", "k", "v")]
puntuaciones = (q @ k.transpose()) / 2.0
mascara = node("causal", (7, 7))
densa = softmax(puntuaciones + mascara) @ v
online = online_attention(q, k, v)
rng = np.random.default_rng(8)
datos = {u.arg: rng.normal(size=forma).astype(np.float32)
         for u in (q, k, v)}
with Program([densa, online], gemm="blocked") as p:
    a, b = p.run(datos)
np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)
```

La raíz cuadrada de la dimensión de cabeza es dos porque $D=4$. La máscara solo permite claves hasta la posición de la consulta. Las dos variantes deben usar el mismo convenio causal. Comparar una atención causal con otra bidireccional no es una prueba de aproximación: son operaciones distintas.

## Comprobar backward sin prometer una memoria que no existe

Construye la suma de la salida de cada variante y pide gradientes respecto a `q,k,v`. Compáralos en un programa aparte. También conviene usar un gradiente de salida no uniforme, por ejemplo ponderar cada posición por una constante diferente; la suma uniforme puede ocultar ciertos errores.

En Lumbre, la regla de gradiente de la operación online recompone operaciones densas para obtener un VJP correcto dentro de su dominio. Por eso la reducción de memoria observada en forward **no se extiende automáticamente a todo el entrenamiento**. Implementar backward por tiles con recomputación de probabilidades y acumulación adecuada es una extensión real, no un detalle que ya esté resuelto por usar la palabra FlashAttention.

## Separar efecto algorítmico y efecto de implementación

Mide las variantes en programas separados. Si declaras ambas como salidas de un único programa, su arena y tiempo incluyen las dos; no puedes atribuir ese total a una de ellas. Usa la misma entrada residente, tres calentamientos y varias repeticiones. Conserva error máximo, mediana y bytes de arena.

En la medición local para forma `(1,2,128,32)`, la arena de la variante densa fue 393.216 bytes y la online 131.072. Sin embargo, sus medianas fueron aproximadamente 437,5 y 624,6 microsegundos: la versión online fue más lenta. La reducción de almacenamiento es clara en ese ensayo; la aceleración no.

Las causas posibles incluyen trabajo escalar por fila, exponenciales frecuentes, falta de vectorización y tamaños donde la versión densa bloqueada reutiliza eficazmente la caché. Estas son hipótesis de análisis, no causas medidas de forma aislada. Para atribuir causalidad hace falta una ablación que cambie un mecanismo y mantenga lo demás.

::: practica Diseña la siguiente ablación
Quieres comprobar si el overhead de procesar elementos individualmente perjudica a la versión online. ¿Qué variante construirías?

**Solución:** una versión que procese bloques de claves y reúna máximos y sumas parciales antes de combinar estados, conservando la misma función y tipos. Se compararía contra la online escalar y la densa, con las mismas formas y verificación. Cambiar además precisión y hardware impediría atribuir la diferencia al tamaño del bloque.
:::

## Pruebas que no debes omitir

Incluye contexto uno, tamaños irregulares, valores repetidos, puntuaciones muy negativas y cambios del máximo. Cada fila causal conserva aquí la clave propia; una máscara que oculte todo exigiría definir otro comportamiento antes de normalizar.

# Laboratorio 11. Tokenización, particiones y pérdidas comparables {#lab:once}

## Por qué los datos también forman parte del experimento

Un compilador correcto puede entrenar una tarea equivocada. Si el conjunto de evaluación aparece repetido en entrenamiento, la pérdida no mide generalización independiente. Si cambia la tokenización, cambia la unidad sobre la que se calcula la pérdida. Si se entrena el tokenizador con todo el corpus, el procedimiento ya ha observado información del conjunto reservado.

El objetivo del laboratorio es construir una tubería pequeña cuyo estado pueda auditarse. Usaremos el ByteBPE original del paquete. No pretende reproducir exactamente el tokenizador de un modelo comercial ni incluir normalización, pretokenización o todos sus tokens especiales. Su contrato es representar bytes UTF-8 y aprender fusiones ordenadas sobre texto de entrenamiento.

## Una prueba de ida y vuelta

```python
from lumbre.tokenizer import ByteBPE

texto_entrenamiento = "la casa y la carta. la casa y la calma."
codec = ByteBPE.fit(texto_entrenamiento, 12)
texto_prueba = "Otra casa: 123."
ids = codec.encode(texto_prueba)
assert codec.decode(ids) == texto_prueba
restaurado = ByteBPE.from_state(codec.state())
assert restaurado.encode(texto_prueba) == ids
```

La base de 256 bytes permite representar bytes no vistos durante el aprendizaje de fusiones. Eso es distinto de un vocabulario de caracteres creado solo con los caracteres observados y sin una política de desconocidos. En generación, una secuencia arbitraria de tokens puede formar bytes que no sean UTF-8 válido; el código de muestra utiliza reemplazo al decodificar para mostrarla sin bloquear el programa.

## Fusiones solapadas: un caso pequeño pero decisivo

En una secuencia de cuatro símbolos iguales hay tres parejas adyacentes si contamos solapamientos, pero una pasada de reemplazo no puede usar el mismo símbolo en dos fusiones. El procedimiento debe especificar cómo recorre y reemplaza parejas. De lo contrario, aprender una lista de fusiones y aplicarla puede producir segmentaciones diferentes.

La implementación del paquete utiliza una regla determinista y reemplazos no solapados. Sus pruebas verifican ida y vuelta y serialización. Ese determinismo importa para checkpoints: los identificadores de tokens no tienen significado fuera del vocabulario y orden de fusiones que los definió.

## Dividir antes de aprender el vocabulario

Prepara dos archivos de documentos distintos, o separa grupos completos antes de entrenar el tokenizador. No cortes ventanas casi idénticas a ambos lados de una frontera y las llames datos independientes. En el script educativo, la ruta BPE hace el corte antes de ajustar fusiones; para un experimento científico, conviene sustituir el corte por documentos por una partición explícita y persistida.

La ruta de caracteres del corpus de demostración usa el alfabeto del texto completo, una simplificación declarada del ejemplo. No se debe presentar esa ruta como una tubería de evaluación rigurosa sin modificarla. El ejercicio consiste precisamente en separar el ejemplo de depuración de un experimento de lenguaje defendible.

## Entrenar con tokens de bytes

```bash
python examples/train_decoder.py --steps 20 --dim 16 \
  --layers 1 --context 8 --byte-tokens --bpe-merges 30 \
  --gemm blocked --output reports/mi_decoder_bpe
```

Esta ruta fue comprobada localmente: construye un vocabulario, entrena y guarda su estado. Veinte pasos y una arquitectura mínima solo validan la tubería; no producen por ese hecho texto útil. La pérdida de ese ensayo no debe compararse directamente con la pérdida del modelo de caracteres como si ambos predijeran la misma unidad.

## De nats por token a una unidad interpretable

Si la pérdida media usa logaritmo natural, está en nats por token. La perplejidad es su exponencial, pero depende de la tokenización y de cómo se ponderan secuencias. Para comparar segmentaciones, puede ser útil acumular la log-verosimilitud total y dividir por bytes del texto original, explicando los detalles de codificación y contexto. No basta con aplicar una conversión constante cuando cada token representa una cantidad variable de bytes.

Una media de medias puede sesgar el resultado. Dos lotes con diferente número de tokens válidos no deben pesar lo mismo si la métrica deseada es media por token. Conserva suma de pérdidas y número de tokens, y divide al final. El padding debe excluirse según un contrato explícito de máscara de pérdida.

::: practica Detectar una comparación inválida
El modelo A tiene pérdida 1 por carácter; B tiene pérdida 2 por token BPE, cuyos tokens cubren varias letras. ¿Demuestra la tabla que A predice mejor el texto?

**Solución:** no. Las unidades difieren. Hace falta evaluar sobre el mismo texto y una normalización comparable, además de controlar contexto y datos. La perplejidad tampoco arregla por sí sola esa diferencia: exponentiar conserva la dependencia de la unidad de tokenización.
:::

## Entrega

Incluye estado del tokenizador, hash del corpus, manifiesto de particiones, prueba de ida y vuelta y definición exacta de la métrica. Añade un texto con caracteres no presentes en el conjunto de entrenamiento. Tu informe debe poder explicar qué identificador representa cada token y cómo reconstruir la misma secuencia después de reanudar.

# Laboratorio 12. Una entrega reproducible y una defensa técnica {#lab:doce}

## El objetivo final no es una carpeta que solo funciona en tu sesión

Al terminar el semestre, otra persona debe poder reconstruir qué hiciste. Eso exige un entorno, órdenes, datos identificados, resultados y límites. No exige empaquetar controladores, gigabytes de caché o binarios incompatibles con su máquina. Una buena entrega separa fuente portable, datos permitidos y resultados específicos de una ejecución.

Crea una carpeta nueva y copia únicamente el paquete fuente. Instala las dependencias declaradas. Ejecuta pruebas, un entrenamiento pequeño y un benchmark. Esta reproducción limpia detecta importaciones accidentales desde tu directorio de trabajo, archivos omitidos y dependencias no registradas.

## Qué pertenece al manifiesto

El manifiesto debe identificar código y datos, configuración, semilla, sistema, compilador y ámbito de la medida. Para GPU añade dispositivo, arquitectura destino, controlador y bibliotecas. Un nombre como «mi ordenador» no permite interpretar diferencias. Un archivo `requirements.txt` no sustituye al resto: dos máquinas con las mismas bibliotecas pueden generar instrucciones distintas.

```json
{
  "experiment": "attention_cpu_comparison",
  "backend": "cpu",
  "dtype": "float32",
  "shape": [1, 2, 128, 32],
  "seed": 17,
  "warmup": 3,
  "repetitions": 20,
  "timed_region": "resident run; no compilation or result copy",
  "correctness": {"rtol": 0.00002, "atol": 0.00002},
  "status": "replace_with_observed_status"
}
```

El ejemplo es una **plantilla**, no un resultado. El campo de estado debe sustituirse por una observación real. Es preferible un estado «no ejecutado: falta dispositivo» a inventar ceros para rellenar una tabla. Los valores ausentes no son tiempos nulos ni errores perfectos.

## Separar tres rutas de ejecución

La ruta de **corrección** utiliza tamaños pequeños, entradas diseñadas y comprobaciones estrictas. La ruta de **rendimiento** utiliza tamaños representativos, calentamiento y repeticiones, pero solo después de superar corrección. La ruta de **entrenamiento** comprueba estabilidad del sistema completo y conserva estado. Mezclarlas en un script gigantesco hace difícil interpretar un fallo.

También debe separarse tiempo frío y caliente. El tiempo frío incluye trabajo inicial que puede amortizarse, como compilación y carga. El caliente describe repeticiones con recursos preparados. Ninguno es más verdadero por definición; responden a preguntas de uso distintas. Una aplicación interactiva puede sufrir el frío, mientras un entrenamiento largo amortiza gran parte de él.

## Preparar una defensa con un caso que no conozcas

Pide a otra persona que cambie una forma no fundamental: añade un elemento al contexto o usa una matriz irregular. Debes predecir qué máscaras y qué buffers cambian. Después ejecuta. Si solo sabes repetir la orden exacta del ejemplo, todavía no has construido una comprensión transferible.

La defensa puede comenzar con cuatro minutos de explicación sin pantalla: problema, representación, transformación y evidencia. Después muestra un resultado correcto y un fallo intencional detectado. Termina con una limitación concreta y una prueba que necesitarías para eliminarla. No hace falta afirmar que todo funciona; sí hace falta saber dónde termina la garantía.

::: practica Pregunta de defensa resuelta
«Tu compilador tarda menos con fusión. ¿Cómo sabes que no has eliminado trabajo necesario?»

**Respuesta:** comparo las salidas y gradientes dentro del dominio declarado, pruebo casos frontera y explico qué dependencia permite fusionar sin alterar efectos. Además inspecciono el plan para confirmar que todas las salidas se calculan. La mejora de tiempo por sí sola no sirve como prueba: un programa que no hace nada es rápido.
:::

## Rúbrica orientativa para el cierre del curso

| Dimensión | Evidencia sólida | Señal de insuficiencia |
|---|---|---|
| Semántica | Contrato y casos que lo distinguen | «Se parece a NumPy» sin formas ni tipos |
| Compilación | IR antes/después y código generado | Solo llamada opaca a una biblioteca |
| Correctitud | Pruebas, invariantes, errores detectados | Una captura con una salida favorable |
| Rendimiento | Frontera medida, muestras y baseline | Único mejor tiempo sin configuración |
| Entrenamiento | Gradientes, pérdida, checkpoint reproducible | Texto de muestra como única evidencia |
| Proyecto | Hipótesis, ablación y límites | Promesas de trabajo futuro sin resultado |

La rúbrica es una propuesta docente del libro, no una norma de una universidad. Un proyecto puede obtener un buen resultado científico aunque su variante sea más lenta, siempre que el experimento esté bien diseñado y explique el mecanismo y alcance del resultado. Lo que no debe premiarse es una afirmación más amplia que la evidencia.

## Entrega final del estudiante

Conserva un README con una orden de inicio, el código modificado, pruebas, un informe de diez a veinte páginas y resultados en formato legible por máquina. El informe debe incluir al menos una tabla de corrección y una de coste. Adjunta el diff respecto a la base para que se pueda identificar la contribución propia.

No publiques datos privados ni redistribuyas material cuya autorización no hayas comprobado. Las licencias del código externo siguen siendo parte del proyecto, aunque el compilador propio tenga otra licencia. Para estudiar una idea no es necesario copiar su implementación: se puede derivar una versión original y citar la fuente conceptual.
