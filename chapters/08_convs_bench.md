# Convoluciones y kernels de reducción rápidos {#ch:convs}

## Reutilización que se solapa entre ventanas

En una convolución de filtro 3×3 y paso uno, dos ventanas vecinas comparten gran parte de sus entradas. Una implementación que vuelve a traer toda la ventana desde memoria lejana desaprovecha esa estructura. El problema se parece a GEMM, pero el patrón de acceso incorpora geometría, bordes y canales.

El primer objetivo sigue siendo una referencia correcta. El segundo es escoger una organización que reutilice entradas y pesos. El tercero es decidir si una especialización compensa frente a una descomposición en GEMM. No hay una única «convolución rápida» independiente de forma, layout y hardware.

## Contrato completo de conv2d

Una firma debe especificar layout de entrada, layout de pesos, stride, padding, dilatación, grupos, tipo y forma de salida. Una convolución con grupos divide canales en subconjuntos; una depthwise utiliza una estructura aún más particular. No basta con declarar que el filtro mide tres.

En la referencia del laboratorio usaremos NCHW y pesos OIHW, sin grupos. Las coordenadas de entrada para una posición de salida `(oy,ox)` y filtro `(ry,rx)` son `iy=oy*stride+ry*dilation-padding` e `ix=ox*stride+rx*dilation-padding`. Si esas coordenadas quedan fuera de la imagen, aportan cero bajo el padding elegido.

```c
float acc = 0.0f;
for (int c = 0; c < Cin; c++)
  for (int ry = 0; ry < R; ry++)
    for (int rx = 0; rx < S; rx++) {
      int iy = oy*stride + ry*dilation - padding;
      int ix = ox*stride + rx*dilation - padding;
      if (0 <= iy && iy < H && 0 <= ix && ix < W)
        acc += X[((n*Cin+c)*H+iy)*W+ix]
             * F[((o*Cin+c)*R+ry)*S+rx];
    }
```

El fragmento es el cuerpo de una salida; los bucles sobre n, o, oy y ox lo rodean. El kernel completo del paquete valida tamaños y calcula todas las salidas.

## Separar interior y bordes

Las posiciones interiores tienen ventanas completamente válidas y no necesitan comprobar cada acceso. Los bordes sí. Un compilador puede generar un kernel o región rápida para el interior y otro camino general para el borde.

El beneficio depende del tamaño de la imagen y del filtro. En imágenes diminutas, gran parte de las posiciones son borde y la separación puede añadir complejidad sin ganar. En imágenes grandes, eliminar ramas del interior puede facilitar vectorización y optimización de direcciones.

No elimines la máscara del borde basándote en que «normalmente el padding es pequeño». La prueba debe derivar el rango exacto de coordenadas de salida cuyo filtro queda dentro de límites.

## Transformar a GEMM paso a paso

Para cada posición de salida, enumera los valores de su ventana a través de canales y coordenadas de filtro. Esa lista tiene longitud `Cin*R*S`. Colócala como una fila de una matriz P. Aplana cada filtro de salida como una columna de una matriz W. Entonces `P@W` produce los canales de salida para todas las posiciones.

La transformación conserva el orden de los productos si se define una enumeración consistente. El resultado debe reorganizarse al layout de salida. El coste de construir P, su tamaño y sus copias forman parte del operador, no son un gasto que se pueda ocultar al comparar con una convolución directa.

::: ejemplo Un caso de una dimensión
La señal `[1,2,3,4]` y filtro `[2,1]` producen P con filas `[1,2]`, `[2,3]`, `[3,4]`. Multiplicar P por la columna `[2,1]` da `[4,7,10]`. El ejemplo muestra qué se duplica: los valores 2 y 3 aparecen en varias filas de P.
:::

## Convolución implícita

Una versión implícita no escribe P completa. El microkernel solicita un elemento de P y una función de índices lo traduce a X o a cero de padding. Esto reduce almacenamiento intermedio, pero introduce cálculos de dirección y puede complicar cargas vectoriales.

El compilador puede precomputar partes de esos índices, desenrollar tamaños de filtro pequeños o empaquetar paneles en lugar de toda la matriz. Es un espacio de decisiones más rico que «directa o im2col», y permite proyectos de autotuning bien delimitados.

## Layouts de canales

En NCHW, posiciones espaciales vecinas de un mismo canal son contiguas. En NHWC, los canales de un píxel son contiguos. Una implementación puede preferir una u otra según cómo vectoriza y cómo alimenta instrucciones matriciales.

Cambiar todo el modelo a otro layout tiene un coste si algunas capas necesitan conversiones. Un kernel aislado más rápido puede empeorar la red completa al introducir transposiciones adicionales. El benchmark de una capa debe complementarse con una medida de la cadena donde se utiliza.

## Reducciones especializadas

Una suma por fila y una suma por columna tienen la misma clase algebraica, pero diferentes accesos en un layout dado. Un kernel rápido puede asignar varios elementos contiguos a cada carril, acumular localmente y reducir entre carriles. Para columnas puede convenir una transposición por bloques o una distribución distinta.

El tamaño de la reducción también cambia la estrategia. Una reducción de ocho valores no necesita el mismo diseño que una de un millón. El coste de lanzamiento y la cantidad de filas disponibles para paralelismo orientan la selección.

## Fusionar estadísticas y epílogos

En normalización, el cálculo de estadísticas y la transformación final están relacionados. Mantener una fila o un bloque cerca permite evitar lecturas repetidas. Sin embargo, si la fila no cabe en memoria local, pueden ser necesarias varias pasadas.

Para varianza, fórmulas algebraicamente equivalentes pueden tener estabilidad diferente. Calcular media de cuadrados menos cuadrado de la media puede perder precisión cuando los valores tienen una media grande y una dispersión pequeña. Una optimización de reducción debe analizar también ese régimen numérico.

## Convolución y backward

El gradiente respecto a pesos suma contribuciones de todas las ventanas. El gradiente respecto a entrada acumula contribuciones de las salidas cuyas ventanas incluían cada posición. Esa segunda operación tiene solapamientos: varias salidas influyen en una misma entrada.

Una estrategia de referencia puede asignar un elemento de gradiente de entrada a cada trabajador y recorrer sus contribuciones. Otra puede repartir salidas y acumular mediante atomics. Ambas pueden ser correctas bajo sus contratos, pero difieren en trabajo, regularidad y determinismo numérico.

## Pruebas indispensables

Utiliza filtros asimétricos, imágenes no cuadradas, stride mayor que uno, dilatación y padding que creen bordes. Comprueba un caso pequeño a mano. Después compara una referencia directa con la descomposición im2col en FP64 o FP32 según el contrato.

Para backward, realiza diferencias finitas en unas pocas posiciones de entrada y pesos. No necesitas perturbar millones de parámetros para detectar una inversión del filtro o una condición de borde incorrecta.

## Objetivo de rendimiento

La meta de semanas 5–6 no es declarar que toda convolución generada gana a una biblioteca. Es construir una ruta desde una referencia general hasta especializaciones evaluables. Para una afirmación competitiva, la matriz de casos debe incluir tamaños reales, layouts y costes de conversión.

En el libro, la convolución directa se aporta como laboratorio de referencia y base de extensión; el núcleo de entrenamiento del decoder no utiliza conv2d. Esta distinción evita confundir cobertura del temario con dependencias de un modelo concreto.

::: practica Elegir una estrategia
Tienes un filtro 1×1 sobre muchos píxeles y numerosos canales. ¿Qué simplifica respecto a un filtro 5×5? ¿Por qué puede parecerse especialmente a GEMM?
:::

**Solución.** No hay una ventana espacial de varios píxeles: cada salida combina canales del mismo píxel, salvo el efecto de stride y otros parámetros. Reorganizando píxeles como filas y canales como características, la operación se acerca a una proyección matricial. El layout y las conversiones siguen importando.

::: comprueba Antes de continuar
Debes poder escribir la coordenada de una ventana, explicar qué repite im2col y detectar por qué un gradiente de entrada necesita acumulación. Después propone una prueba que distinga correlación cruzada de convolución con filtro invertido.
:::

# Medir, perfilar y autotunear sin engañarse {#ch:benchmark}

## El cronómetro también tiene un contrato

Antes de medir, escribe qué intervalo temporal te interesa. ¿Primera llamada con compilación? ¿Kernel caliente con datos residentes? ¿Operador completo con packing? ¿Entrenamiento por token incluyendo carga de datos? Son preguntas diferentes y todas pueden ser útiles.

Un benchmark no es bueno por tener muchos decimales. Es bueno cuando la cifra corresponde a una pregunta clara, el procedimiento es reproducible y la incertidumbre se conserva. No redondees una variación grande hasta convertirla en una mejora aparente pequeña.

## Calentamiento

Las primeras llamadas pueden incluir carga de módulos, asignaciones, compilación diferida, cachés frías y cambios de frecuencia. Un benchmark caliente suele realizar varias llamadas antes de medir. Un benchmark de arranque debe conservar esos costes en lugar de eliminarlos.

No existe un número universal de warmups. Comprueba que el comportamiento se estabiliza para el caso estudiado y registra la política. En el script de la entrega se utilizan tres llamadas previas como procedimiento sencillo, no como garantía de estacionariedad de cualquier dispositivo.

## Sincronizar lo que se mide

En una GPU, medir el tiempo del lanzamiento con un reloj CPU puede contar únicamente cuánto tarda el host en encolar trabajo. Para medir ejecución necesitas eventos del dispositivo o una sincronización bien situada. Sin esa precaución, puedes obtener un kernel «más rápido» que todavía no ha terminado.

Tampoco debes sincronizar más de lo que requiere el escenario real y llamar a ese resultado «rendimiento asíncrono». Una medida de kernels aislados y otra de una cadena con solapamiento deben describir sus puntos de sincronización.

## Muestras y estadísticas

Recoge varias muestras. La mediana reduce la influencia de algunos valores extremos, pero no explica por sí sola la distribución. Conserva mínimo, máximo, cuantiles y, cuando sea útil, intervalos de confianza calculados sobre un procedimiento adecuado.

Si seleccionas la mejor de mil ejecuciones de una versión y la media de diez de otra, la comparación favorece artificialmente a la primera. Aplica la misma política. Intercalar versiones o aleatorizar el orden puede reducir efectos de deriva térmica y de carga externa.

## Corrección antes de tiempo

Un candidato que no produce la salida correcta queda fuera de la comparación de rendimiento del mismo operador. No puede ganar por omitir una reducción, usar etiquetas equivocadas o escribir solo una parte de la salida.

Las pruebas de corrección se realizan fuera del intervalo medido, pero con los mismos tipos, formas y parámetros del candidato. No pruebes un kernel escalar seguro y midas otro vectorizado sin validarlo. El hash del código ayuda a unir evidencia de corrección y evidencia de tiempo.

## Throughput y latencia

La latencia mide cuánto tarda una tarea. El throughput mide cuántas tareas o tokens se procesan por unidad de tiempo. Aumentar batch puede mejorar throughput y empeorar latencia por petición. En entrenamiento suele interesar tokens por segundo a una configuración de batch y secuencia; en generación interactiva también importa el tiempo por token y la primera respuesta.

La cantidad de FLOP/s es una métrica de utilización bajo una convención de conteo. No es una medida directa de calidad del modelo ni de eficiencia económica. Un algoritmo que realiza menos operaciones puede ser más rápido y mostrar menos FLOP/s.

## El comparador PyTorch

Para comparar con PyTorch, identifica si utiliza ejecución eager, compilación, una biblioteca fusionada o un kernel especial. Fija dtype, TF32 cuando corresponda, layout, dispositivo y forma. Una llamada llamada `matmul` puede escoger implementaciones distintas según esos detalles.

No atribuyas a PyTorch el coste de transferir datos desde CPU mientras tu candidato recibe datos residentes, salvo que esa sea precisamente la comparación de escenarios que deseas estudiar y la expliques. El usuario necesita saber qué costes pagará en su aplicación.

## Autotuning

Autotuning explora configuraciones de scheduling: tamaños de tile, número de hilos, unrolling, stages, layout y elección de algoritmo. Cada candidato pasa validación y medición. Después se conserva la configuración elegida para una clave de problema.

El espacio debe limitarse mediante restricciones de recursos y conocimiento del hardware. Probar combinaciones ilegales consume tiempo y puede bloquear o fallar. Una estrategia útil comienza con candidatos seguros y utiliza modelos para priorizar, sin confundir predicción de coste con tiempo observado.

```text
Para cada configuración candidata:
    comprobar restricciones estáticas
    generar y compilar
    verificar resultados
    medir con la política acordada
    registrar artefacto, error y tiempos
Elegir según el objetivo y guardar la clave completa
```

## Coste de buscar

Si autotunear tarda diez segundos y ahorra un milisegundo por ejecución, hacen falta aproximadamente diez mil ejecuciones para recuperar ese coste, ignorando otros gastos. La cuenta es un ejemplo de amortización, no una razón para prohibir búsqueda.

Un servicio que ejecuta la misma forma millones de veces puede beneficiarse mucho. Una herramienta interactiva con formas casi siempre nuevas puede preferir una buena heurística inicial. El compilador puede separar un modo de arranque rápido y otro de optimización persistente.

## Overfitting del autotuner

Escoger el tile que gana en un único tamaño puede producir una política frágil. Un proyecto de selección por forma debe separar casos de ajuste y casos de evaluación. También debe distinguir entre memorizar cada forma exacta y aprender una regla que generalice a formas nuevas.

El mismo principio se aplica a agentes que generan kernels. Si el agente ve los tests finales y puede especializarse a sus datos concretos, el resultado no demuestra una implementación general del operador. Deben variar valores y formas dentro del contrato.

## Perfilado

Un perfilador ayuda a identificar dónde se gasta tiempo, cuánta memoria se mueve y qué recursos limitan. Empieza por una vista de alto nivel: tiempo por kernel, número de lanzamientos, transferencias y sincronizaciones. Después profundiza en el kernel dominante.

No optimices una operación que representa el uno por ciento del tiempo esperando multiplicar por dos el programa completo. La ley de Amdahl formaliza esta intuición: incluso eliminar por completo una parte pequeña deja casi todo el tiempo intacto.

```math
S=\frac{1}{(1-f)+f/s}.
```

Aquí f es la fracción original mejorada y s su factor de aceleración. Si f=0,1 y s=10, el speedup total es aproximadamente 1,10, no diez. La fórmula presupone que el resto no cambia; en sistemas reales puede haber efectos secundarios que también deban medirse.

## Energía y coste

Un kernel más rápido puede consumir más potencia instantánea y aun así menos energía total, o no. La energía integra potencia durante el tiempo. Un presupuesto de entrenamiento debe considerar horas, precio de recursos, almacenamiento y transferencias, no solo el pico de FLOP/s anunciado.

El libro no fija precios de nube a octubre de 2026 sin una consulta específica y una región: cambian y dependen de reservas, disponibilidad y proveedor. Para planificar un proyecto, utiliza un presupuesto parametrizado y actualiza sus entradas al contratar recursos.

## Una tabla de resultados honesta

Incluye forma, tipo, backend, estrategia, error, mediana, dispersión, memoria y coste de compilación. Conserva además configuración del sistema y fuente generada. Una tabla sin estas columnas puede servir como borrador personal, pero no sostiene una afirmación comparativa fuerte.

Los informes JSON del paquete registran muestras y artefactos del compilador. No incluyen contadores de hardware ni una comparación con bibliotecas externas; esas ausencias se declaran en lugar de rellenarse con estimaciones presentadas como mediciones.

::: practica Interpretar una mejora
Una versión reduce el kernel de 1 ms a 0,5 ms, pero añade un packing de 0,8 ms en cada llamada. La original no necesitaba packing. ¿Cuál gana si el panel se usa una vez? ¿Y si se usa cien veces sin volver a empaquetar?
:::

**Solución.** Una vez, la nueva cuesta 1,3 ms y pierde frente a 1 ms. Cien veces, la original cuesta 100 ms y la nueva 0,8+50=50,8 ms, suponiendo que el panel puede reutilizarse sin otros cambios. La reutilización es una precondición del segundo resultado.

::: comprueba Antes de continuar
Debes poder escribir el intervalo medido, separar latencia y throughput, justificar calentamiento y explicar el coste de buscar una configuración. Un benchmark debe permitir que tu hipótesis pierda.
:::
