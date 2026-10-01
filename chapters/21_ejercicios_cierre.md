# Boletín de integración: resolver antes de ejecutar {#ch:boletin-final}

## Problema 1. Un paso completo de aprendizaje con matrices pequeñas

**Enunciado.** Sean $X=\left(\begin{smallmatrix}1&2\\3&4\end{smallmatrix}\right)$, $W=I$ y $b=(1,-1)$. Calcula $Y=XW+b$, la pérdida media de los cuatro elementos de $Y^2$, sus gradientes respecto a $X,W,b$ y una actualización de descenso de gradiente de tasa 0,1 sobre $W,b$. El sesgo se comparte entre filas.

**Resolución del forward.** Multiplicar por identidad conserva $X$. Sumar el sesgo aumenta la primera columna en uno y reduce la segunda en uno. Así $Y=\left(\begin{smallmatrix}2&1\\4&3\end{smallmatrix}\right)$. Los cuadrados son 4,1,16,9, cuya suma es 30. La media es 7,5. Dividir por cuatro, y no por dos, corresponde a la media de todos los elementos.

**Resolución del backward.** La derivada de cada cuadrado es dos veces su entrada y la media introduce un cuarto. Por tanto, $G=\partial L/\partial Y=Y/2$. El sesgo recibe suma de filas: $(1+2,\;0,5+1,5)=(3,2)$. Como $W$ es identidad, el gradiente de $X$ coincide con $G$. El gradiente de pesos es $X^TG$:

```math
\frac{\partial L}{\partial W}=\begin{pmatrix}7&5\\10&7\end{pmatrix},\qquad
\frac{\partial L}{\partial X}=\begin{pmatrix}1&0.5\\2&1.5\end{pmatrix}.
```

**Actualización.** Los pesos nuevos son $\left(\begin{smallmatrix}0,3&-0,5\\-1&0,3\end{smallmatrix}\right)$ y el sesgo nuevo $(0,7,-1,2)$. La notación decimal con coma no debe confundirse con cuatro componentes: son dos valores, 0,7 y -1,2. Al repetir forward se obtiene $\left(\begin{smallmatrix}-1&-1,1\\-2,4&-1,5\end{smallmatrix}\right)$ y la pérdida es 2,555. En este ejemplo el paso reduce la pérdida; una tasa arbitraria no tendría esa garantía.

```python
import numpy as np
from lumbre import param, Program, gradients

x, w, b = param("X", (2,2)), param("W", (2,2)), param("b", (2,))
y = x @ w + b
loss = (y * y).mean()
g = gradients(loss, [x, w, b])
with Program([y, loss, *g], gemm="blocked") as p:
    outputs = p.run({"X": [[1,2],[3,4]], "W": np.eye(2), "b": [1,-1]})
np.testing.assert_allclose(outputs[1], 7.5)
np.testing.assert_allclose(outputs[3], [[7,5],[10,7]])
np.testing.assert_allclose(outputs[4], [3,2])
```

La prueba integra matmul, broadcasting, reducción, reutilización de un nodo en dos aristas y transposición en backward. Si una implementación falla, compara los intermedios en ese orden. Un gradiente final incorrecto puede originarse en un eje de reducción equivocado, no necesariamente en la regla del producto matricial.

## Problema 2. Fusión que duplica trabajo

**Enunciado.** Una operación cara $h=f(x)$ alimenta dos consumidores. Fusionarla dentro de ambos elimina un buffer de un millón de elementos, pero ejecuta $f$ dos veces. ¿Debe fusionarse?

**Solución razonada.** Falta información sobre coste de $f$, tráfico del buffer y recursos. Si $f$ es una suma sencilla y el buffer se escribiría y leería desde memoria lenta, recomputar puede convenir. Si $f$ es una reducción costosa o una función trascendental compleja, duplicarla puede perjudicar. La decisión también cambia si uno de los consumidores apenas usa una parte de $h$.

Formula dos estimaciones: coste de calcular una vez más escribir y leer, frente a coste de calcular dos veces sin intermedio. Después mide ambas con el mismo contrato. La respuesta correcta no es una regla universal, sino un procedimiento que conserva semántica y compara costes. El número de kernels por sí solo no determina el ganador.

## Problema 3. Una vista invertida y una dirección válida

**Enunciado.** Una fila de cinco elementos tiene stride uno y offset cero. Se invierte lógicamente. Escribe nuevo offset, stride y direcciones de las posiciones 0 y 4. Explica por qué no puede usarse un puntero al comienzo con stride negativo sin cambiar offset.

**Solución.** El nuevo offset es cuatro y el stride -1. La dirección de la posición lógica cero es 4; la de la posición cuatro es $4-4=0$. Si se conservase offset cero, la segunda posición lógica tendría dirección -1, fuera de la región. Invertir no es solamente cambiar el signo de un stride: hay que situar el origen lógico en el extremo correcto.

Para una dimensión vacía no existe un último elemento que consultar. El modelo de vista debe evitar fabricar un acceso a una posición anterior al buffer. Es otro motivo para probar formas vacías por separado, incluso cuando no producen salida.

## Problema 4. Un techo de rendimiento no es una medida

**Enunciado.** Una operación realiza 200 millones de operaciones y mueve como mínimo 100 millones de bytes. La máquina ofrece un techo de 1 billón de operaciones por segundo y 100 mil millones de bytes por segundo, en unidades decimales. Calcula la cota idealizada del modelo roofline y explica sus límites.

**Solución.** La intensidad idealizada es dos operaciones por byte. El techo por memoria sería 200 mil millones de operaciones por segundo, inferior al techo de cómputo de un billón. El tiempo mínimo por datos es un milisegundo; el mínimo por cómputo, 0,2 milisegundos. El modelo ideal toma el mayor, un milisegundo.

No se ha medido un tiempo de un milisegundo. Puede haber más tráfico que el mínimo, latencia, falta de paralelismo, conversiones y overhead. Tampoco se deduce automáticamente que aumentar operaciones mejore rendimiento: cambiar el algoritmo puede modificar bytes, instrucciones y recursos. El modelo sirve para formular hipótesis y cotas, no para sustituir el cronómetro. [@roofline]

## Problema 5. Dos barreras y un hilo sin salida

**Enunciado.** Un bloque carga un tile compartido, calcula con él y vuelve a usar la misma memoria para el siguiente tile. Un hilo no tiene salida válida y retorna antes de la primera barrera. Otro elimina la segunda barrera porque «ya hemos calculado». Identifica los dos riesgos.

**Solución.** El retorno puede impedir la participación colectiva requerida y, además, omitir una carga que otros hilos necesitan. El hilo sin salida final no es necesariamente un hilo sin responsabilidad. Debe respetar el protocolo del bloque y enmascarar sus accesos apropiadamente.

Eliminar la segunda barrera permite que hilos rápidos sobrescriban memoria compartida mientras otros siguen leyendo la etapa anterior. La primera protege que la etapa esté lista antes de consumir; la segunda protege que haya terminado de consumirse antes de reutilizar. Son relaciones temporales distintas. Una ejecución favorable no demuestra ausencia de carrera.

## Problema 6. Dos lotes y una media mal calculada

**Enunciado.** Un lote tiene dos tokens válidos con pérdidas 1 y 3; otro tiene seis tokens con pérdida 2 cada uno. Calcula la media global y la media de medias. Después cambia las pérdidas del segundo lote a 4 y repite.

**Solución.** En el primer caso ambos lotes tienen media 2, de modo que las dos formas dan 2. Ese ejemplo no detecta el error. En el segundo, la media global es $(1+3+6\cdot4)/8=3,5$, mientras la media de medias es $(2+4)/2=3$.

La diferencia aparece porque se dio el mismo peso a grupos con distinta cantidad de datos. Para una métrica media por token se acumula suma de pérdidas y número de tokens válidos. En entrenamiento distribuido hay que aplicar la misma disciplina al combinar contribuciones; una normalización equivocada puede cambiar la magnitud del gradiente al variar el tamaño del lote o el número de réplicas.

## Problema 7. Un caché aparentemente válido

**Enunciado.** Se guarda una biblioteca compilada usando como clave únicamente el texto C. Después se cambia de compilador y de flags, pero el texto no cambia. ¿Es correcto reutilizarla?

**Solución.** No está justificado por esa clave. El artefacto depende también del toolchain, opciones, arquitectura y ABI relevantes. Incluso si la biblioteca pudiera cargarse, ya no demostraría el experimento que se cree estar realizando. Una clave de caché debe representar las entradas que afectan al resultado, y un manifiesto debe permitir saber cuál se utilizó.

Tampoco conviene escribir directamente sobre un archivo final que otro proceso puede estar cargando. La compilación se realiza en un temporal y se publica al terminar de forma segura. El sistema de caché forma parte de la reproducibilidad, no es únicamente una optimización de comodidad.

## Problema 8. Qué significa un máximo con empate

**Enunciado.** La pérdida es el máximo de `(2,2,1)`. El sistema reparte el gradiente a partes iguales entre máximos. ¿Cuál es el vector devuelto? ¿Demuestra un error que una perturbación positiva del primer elemento observe pendiente uno?

**Solución.** La convención devuelve `(0,5;0,5;0)`. En el empate no existe una derivada ordinaria única. Una perturbación positiva del primero rompe el empate y selecciona otra región, donde la pendiente respecto a ese elemento es uno. Esa observación no contradice la convención de subgradiente declarada en el punto de empate.

La prueba debe separar puntos suaves, donde diferencias finitas son una herramienta directa, de fronteras no diferenciables, donde hay que comprobar la política elegida. Si una biblioteca usa otra convención, ambas pueden ser defendibles; no se exige igualdad entre políticas distintas sin acordar el contrato.

## Problema 9. Compresión de pesos y memoria de entrenamiento

**Enunciado.** Un modelo guarda pesos en dos bytes por parámetro. Una persona multiplica ese tamaño por el número de parámetros y afirma que ha calculado toda la memoria necesaria para entrenar con AdamW. ¿Qué falta?

**Solución.** Como mínimo hay que considerar gradientes, momentos, posibles pesos maestros, activaciones guardadas, workspace y estado del runtime. Los tipos de cada categoría pueden ser distintos. La compresión de pesos reduce una categoría; no garantiza la misma reducción del total.

También hay que distinguir el pico de una fase del tamaño final de un checkpoint. Un checkpoint puede omitir activaciones que existen durante el paso. Un proceso puede reservar una arena mayor que los tensores lógicamente vivos. Medir ambas cosas con el nombre «memoria del modelo» crea confusión.

## Problema 10. Interpretar una tabla de un artículo

**Enunciado.** Un artículo reporta aceleraciones entre 0,8 y 3,2 en su suite. Alguien resume «es tres veces más rápido». Da una reformulación precisa y una pregunta que harías antes de portarlo.

**Solución.** La reformulación conserva variabilidad: «en las operaciones y condiciones estudiadas, el candidato incluye casos más lentos que la referencia y casos con mejoras de hasta 3,2 veces». Hace falta conocer agregado, formas, precisión y hardware antes de atribuir una mejora típica.

Una pregunta de port es si el mecanismo ganador depende de una instrucción o layout exclusivo de la arquitectura evaluada. Otra es si el coste de búsqueda y compilación está amortizado. La interpretación científica conserva el dominio; no sustituye una distribución por su máximo favorable.

# Glosario razonado y conexiones que conviene conservar {#ch:glosario}

## De la intención al programa

**Tensor.** Datos organizados por ejes y forma. En este curso no significa automáticamente un objeto físico multidimensional en memoria: su almacenamiento puede ser lineal y su vista traducir coordenadas.

**Operación elemento a elemento.** Cada salida se calcula con elementos correspondientes de entradas, después de aplicar broadcasting. No necesita por definición mezclar distintas posiciones de una reducción.

**Reducción.** Combina contribuciones de uno o más ejes. Requiere identidad o política para el caso vacío, orden numérico y una salida por combinación de los ejes conservados.

**IR.** Representación intermedia del programa. Debe tener semántica, tipos y reglas de validez; no es simplemente cualquier estructura de datos que contenga nombres de operaciones.

**UOp.** Nombre usado aquí para un nodo de operación de bajo nivel o de una IR unificada, inspirado en el recorrido del curso. El UOp educativo de Lumbre no es la clase exacta de tinygrad ni tiene toda su semántica.

**DAG.** Grafo dirigido sin ciclos. Permite ordenar dependencias de expresiones puras. Los bucles de control necesitan una representación de recurrencia, no un ciclo arbitrario introducido en el evaluador topológico.

**SSA.** Organización donde cada nombre estático tiene una definición. Facilita seguir productores y usos; no elimina alias ni convierte memoria mutable en una expresión pura.

**Reescritura.** Sustitución de una parte de la representación por otra que preserva el contrato bajo hipótesis. Una identidad real no es automáticamente una reescritura válida de FP32 estricto.

**Lowering o bajada.** Traducción a una representación con decisiones más concretas. Debe definir qué operaciones desaparecen, cuáles aparecen y qué invariantes conserva.

**Renderer.** Parte que escribe una representación destino, como C. Puede apoyarse en análisis anteriores y en un compilador externo que todavía realizará selección de instrucciones.

## De las coordenadas a la máquina

**Shape.** Tamaño de cada eje lógico. Dos tensores con el mismo número de elementos pueden tener formas diferentes y admitir operaciones distintas.

**Stride.** Avance en almacenamiento al aumentar una coordenada. Un stride cero permite lecturas repetidas; uno negativo exige un origen lógico apropiado.

**Layout.** Regla completa que relaciona coordenadas con almacenamiento o propiedad por participantes. Incluye más información que la forma.

**Tile.** Región de trabajo o datos que se procesa como unidad. Su tamaño afecta reutilización, bordes y recursos; no es necesariamente un bloque físico de GPU.

**Upcasting.** En el contexto del compilador, convertir parte de un eje de iteración en trabajo explícito de varios valores o registros. No debe confundirse con convertir un número a un tipo de más precisión.

**SIMD.** Una instrucción opera sobre varias posiciones de datos. Tener un bucle con varios elementos no demuestra que el compilador haya generado una instrucción vectorial.

**SIMT.** Modelo donde varios hilos ejecutan de forma coordinada instrucciones sobre sus datos. La divergencia y las operaciones colectivas requieren comprender quién participa.

**Memoria compartida o local al grupo.** Almacenamiento usado para cooperación de un grupo de ejecución. El nombre concreto depende del entorno; no equivale siempre a «local» en otros lenguajes o APIs.

**Barrera.** Relación de sincronización entre participantes bajo un alcance. No es un remedio universal: debe corresponder a los productores y consumidores correctos.

**Tensor core o instrucción matricial.** Capacidad especializada para operaciones sobre fragmentos. Impone contratos de tipos, layouts y participación; no es una llamada mágica para cualquier matriz.

**Kernel de cálculo.** Unidad de código que realiza trabajo numérico, a menudo en un acelerador. No es sinónimo del kernel de Linux ni de un controlador.

**Runtime.** Gestiona buffers, módulos, argumentos, ejecución y finalización. Su corrección es necesaria aunque la fórmula del kernel sea perfecta.

**Firmware.** Software de control próximo al dispositivo. Puede interpretar protocolos o inicializar componentes; no reemplaza automáticamente las funciones de una biblioteca de ML.

## Del programa al aprendizaje

**Autodiferenciación.** Transformación que aplica reglas de derivación a un programa. No es estimar todos los gradientes mediante diferencias finitas ni pedirlos a un LLM.

**VJP.** Producto de un vector de contribuciones de salida por el jacobiano en la orientación del modo inverso. Permite propagar una pérdida sin materializar todo el jacobiano.

**Gradiente acumulado.** Suma de contribuciones por todos los caminos de uso. Es indispensable para nodos compartidos, broadcasting y embeddings repetidos.

**Checkpoint.** Estado persistido suficiente para el objetivo de recuperación declarado. Reproducir una trayectoria exige más que los pesos: optimizador, paso, datos y azar pueden importar.

**Token.** Unidad de la secuencia del modelo. Puede representar un carácter, bytes o una pieza aprendida. Las pérdidas por token no son comparables entre tokenizaciones sin más análisis.

**Logit.** Puntuación anterior a normalizar probabilidades. No es ya una probabilidad ni tiene que estar entre cero y uno.

**Atención causal.** Atención que restringe las claves disponibles a posiciones permitidas por el pasado. La máscara debe coincidir en forward, backward e inferencia.

**KV cache.** Estado de claves y valores conservado durante generación para evitar recomputación. No es el mismo caché que guarda una biblioteca compilada.

**Prefill y decode.** Procesar inicialmente una secuencia y generar posteriormente nuevos tokens. Tienen formas de trabajo y límites de memoria diferentes.

**MoE.** Arquitectura que selecciona expertos para parte del cómputo. Añade decisiones de agrupación y comunicación; parámetros totales y activos dejan de ser la misma cifra.

## De la observación a la conclusión

**Benchmark.** Experimento con frontera medida, carga y referencia. Un número sin esas condiciones no permite interpretar rendimiento.

**Ablación.** Comparación que retira o cambia un mecanismo para estudiar su efecto. Modificar a la vez cinco componentes no aísla una causa.

**Oráculo.** Referencia usada para comprobar una salida. También puede tener limitaciones; conviene combinar implementaciones y argumentos cuando la propiedad importa.

**Prueba diferencial.** Comparación de implementaciones sobre entradas concretas. Detecta discrepancias, pero no garantiza por sí sola todos los casos posibles.

**Resultado negativo.** Evidencia de que una hipótesis no se cumple en las condiciones estudiadas. Bien documentado puede ser más informativo que una mejora seleccionada sin controles.

**SOTA.** Estado del arte respecto a una tarea, fecha, suite y condiciones. No es una propiedad que se obtenga por usar una arquitectura moderna o una GPU reciente.

::: comprueba El mapa mental final
Una intención matemática se expresa como programa; el compilador la transforma; el runtime la ejecuta sobre una máquina; las pruebas y medidas permiten evaluar qué se ha conservado y qué se ha mejorado. El entrenamiento añade gradientes, estado y datos, pero no elimina ninguna de esas obligaciones.
:::
