# De letras a predicciones: construir un modelo de lenguaje {#ch:lenguaje}

::: idea Un juego de completar una secuencia
Escribimos «el gato» y pedimos al modelo una distribución sobre lo que podría venir después. No le entregamos una regla gramatical escrita a mano. Ajustamos sus parámetros usando ejemplos donde sí conocemos la continuación.
:::

## Texto, token e identificador

Un ordenador necesita una representación numérica del texto. En el ejemplo didáctico, cada carácter del corpus tiene un identificador entero. La letra a podría ser el número dos y el espacio el número cero. Esos identificadores son etiquetas: no significa que a sea el doble de un espacio.

Un token puede ser un carácter, un byte, una palabra o una pieza de palabra según el tokenizador. El vocabulario es la colección de tokens disponibles. Su tamaño, V, determina cuántas alternativas puede predecir el modelo en cada posición.

El entrenamiento del paquete usa caracteres para mantener visible todo el proceso. Un tokenizador por subpalabras es más habitual en grandes modelos, pero añade decisiones sobre normalización, bytes, tokens especiales y segmentación. Cambiar el tokenizador cambia tanto los datos como las dimensiones de la entrada y salida.

::: cuidado No ordenar palabras por su identificador
Si «gato» tiene identificador 17 y «perro»18, esa cercanía numérica no representa por sí sola parecido semántico. El embedding es la tabla de vectores que el entrenamiento aprende a asociar a esas etiquetas.
:::

## La tabla de embeddings

Cada fila de una tabla de tamaño V×D contiene D números. D es el ancho de la representación. Para convertir una secuencia de identificadores en vectores, seleccionamos sus filas. Esa selección es exactamente el gather que ya implementamos.

Si los tokens son [2,0,2], seleccionamos dos veces la fila dos y una vez la fila cero. Durante el retroceso, las contribuciones de ambas apariciones de la fila dos se suman. El pequeño ejercicio de scatter-add del capítulo anterior ya estaba preparando esta operación real de un modelo de lenguaje.

Un lote de B secuencias de longitud T tiene identificadores de forma B×T. Después del embedding, la forma es B×T×D. B identifica ejemplos, T posiciones y D características. Aunque todas sean dimensiones de un tensor, tienen funciones diferentes y no deben intercambiarse accidentalmente.

## La etiqueta está desplazada una posición

Para una secuencia de identificadores [4,1,7,3,2], podemos usar como entrada [4,1,7,3] y como etiquetas [1,7,3,2]. En cada posición se intenta predecir el token siguiente.

| Posición | Token de entrada | Etiqueta que debe predecir |
|---|---|---|
| 0 | 4 | 1 |
| 1 | 1 | 7 |
| 2 | 7 | 3 |
| 3 | 3 | 2 |

Si usamos la misma secuencia como entrada y etiqueta sin desplazamiento, el modelo podría aprender a copiar el token actual. La pérdida bajaría, pero no estaríamos entrenando la tarea pretendida. Por eso la construcción del batch es una parte verificable del sistema, no un detalle administrativo.

## Logits y probabilidades

La salida de una posición son V puntuaciones reales llamadas logits. No son todavía probabilidades: pueden ser negativas y no tienen por qué sumar uno. Softmax las convierte en cantidades positivas normalizadas.

Para logits [0,0,0], cada alternativa recibe un tercio. Para [0,1,0], la segunda recibe más peso, pero no certeza. Añadir la misma constante a todas las puntuaciones no cambia las probabilidades: importa su diferencia relativa.

La pérdida de entropía cruzada de una etiqueta correcta t es el negativo del logaritmo de su probabilidad. Si se le asigna 0.5, la pérdida es aproximadamente 0.693; si se le asigna 0.1, aproximadamente 2.303. Castigamos más una respuesta que da poca probabilidad a lo observado.

En el código evitamos calcular una probabilidad diminuta y después su logaritmo si podemos usar una forma estable con log-sum-exp. El capítulo de números ya justificó restar el máximo. La estadística y la estabilidad numérica se encuentran aquí en una operación concreta.

## Una derivada especialmente útil

Para softmax seguido de entropía cruzada de una etiqueta, la derivada respecto a cada logit es la probabilidad predicha menos la etiqueta one-hot. One-hot significa un vector con uno en la alternativa correcta y cero en las demás.

Con probabilidades [0.2,0.5,0.3] y etiqueta primera, el gradiente es [-0.8,0.5,0.3]. El signo negativo en el primer logit indica que aumentarlo reduce la pérdida localmente. Los otros signos indican que conviene disminuirlos. Si promediamos varias posiciones, se incorpora el factor de esa media.

Esta fórmula permite comprobar el motor de autodiff sin confiar en otro framework. En las pruebas del paquete se usan logits grandes positivos y negativos para comprobar también que la estabilización funciona.

## Causalidad: no mirar la solución

Durante entrenamiento disponemos de toda la secuencia, incluidas las etiquetas futuras. Eso permite calcular muchas posiciones en paralelo, pero no autoriza a que una posición use información que no tendría al generar texto.

Para predecir después de la posición i, solo puede utilizar posiciones hasta i. Una máscara causal elimina las contribuciones posteriores. El triángulo permitido incluye la diagonal porque el token actual sí forma parte de la entrada conocida.

```text
Posicion que pregunta -> columnas que puede consultar
0 : 1 0 0 0
1 : 1 1 0 0
2 : 1 1 1 0
3 : 1 1 1 1
```

El test de causalidad cambia el último token y exige que las salidas anteriores permanezcan iguales dentro de la tolerancia. Es una prueba mucho más directa que observar una pérdida sospechosamente baja. También puede detectar una permutación de ejes que aplica la máscara al lugar equivocado.

## Qué significa «entrenar un LLM» en este curso

Construimos una arquitectura decoder-only con componentes usados en modelos de lenguaje modernos y hacemos que su avance, retroceso y actualización se ejecuten mediante nuestro compilador. El ejemplo pequeño sirve para comprobar toda esa cadena.

Pero «modelo de lenguaje» y «gran modelo de lenguaje de calidad competitiva» no son sinónimos. Un modelo de decenas de miles de parámetros y un corpus de unas frases no tiene la capacidad ni la evidencia de un sistema de frontera. Las siguientes partes desarrollan la arquitectura y los mecanismos necesarios para escalar; los resultados locales se etiquetan por su tamaño real.

::: comprueba Una definición operacional del hito
El estudiante alcanza el hito cuando puede explicar cómo se construyen las etiquetas, verificar causalidad y gradientes, ejecutar actualizaciones compiladas, guardar y reanudar el estado, y evaluar sobre datos definidos. Una muestra de texto llamativa por sí sola no satisface esos requisitos.
:::

# Atención: consultar recuerdos con pesos aprendidos {#ch:attention}

## Tres papeles para una misma secuencia

Imagina que cada palabra entrega tres fichas. La primera dice qué información busca; la segunda, qué tipo de información ofrece; la tercera contiene la información que puede aportar. Las llamaremos consulta Q, clave K y valor V.

Las fichas son vectores calculados mediante proyecciones aprendidas. No son frases que el modelo escriba en lenguaje natural. La analogía nos ayuda a separar funciones: Q y K deciden pesos; V aporta el contenido que se mezcla.

La atención del Transformer utiliza productos entre consultas y claves para construir puntuaciones, normaliza esas puntuaciones y combina valores. El trabajo original del Transformer es la referencia histórica de esta organización; el ejemplo numérico siguiente es una construcción didáctica propia. [@attention]

## Una consulta y dos recuerdos

Tomemos una consulta de dos componentes q=[1,0]. Dos claves son k0=[1,0] y k1=[0,1]. Los productos escalares valen 1 y 0. Si dividimos por la raíz de la dimensión, las puntuaciones son aproximadamente 0.707 y 0.

Softmax asigna aproximadamente 0.670 a la primera y 0.330 a la segunda. Con valores v0=[2,0] y v1=[0,4], la mezcla es aproximadamente [1.340,1.321]. No hemos elegido un único recuerdo: hemos construido una combinación ponderada.

Cambiar los valores sin cambiar consultas y claves conserva los pesos, pero cambia el contenido mezclado. Cambiar una clave puede cambiar los pesos aunque su valor permanezca igual. Esta distinción ayuda a depurar una implementación que accidentalmente intercambia K y V.

## Todas las consultas juntas

Para T posiciones y dimensión de cabeza d, Q y K tienen forma T×d. El producto $QK^T$ tiene forma T×T: cada fila corresponde a una consulta y cada columna a una clave.

Aplicamos softmax a cada fila, no a toda la matriz a la vez. Cada consulta debe tener su propia distribución sobre recuerdos. Después multiplicamos por V, de forma T×d, y obtenemos otra matriz T×d.

```math
S=\frac{QK^T}{\sqrt d}+M,\qquad
P=\operatorname{softmax}_{\text{filas}}(S),\qquad
O=PV.
```

M representa la máscara: cero donde se permite consultar y un valor que excluye la posición donde no. En una formulación matemática exacta se usa menos infinito. En una implementación con un número finito muy negativo hay que verificar el rango de puntuaciones y el comportamiento numérico; no es una equivalencia sin condiciones.

## Por qué aparece la raíz de d

El producto escalar suma d productos. Bajo un modelo simplificado de componentes independientes, centradas y de varianza comparable, su dispersión crece con la dimensión. Escalar ayuda a mantener las puntuaciones en un rango manejable para softmax. No es una normalización que garantice una distribución fija para cualquier dato real.

Si todas las puntuaciones se separan demasiado, softmax puede concentrar casi todo el peso en una alternativa y producir sensibilidades muy pequeñas para otras. El factor forma parte de la arquitectura que queremos reproducir; omitirlo cambia el modelo, aunque las formas de los tensores sigan siendo válidas.

## Múltiples cabezas: varias consultas en paralelo

Podemos dividir la representación en H cabezas, cada una con dimensión d=D/H. Cada cabeza aprende proyecciones y mezclas distintas. Después concatenamos sus resultados y aplicamos una proyección de salida.

En la implementación usamos forma B×H×T×d para las operaciones de atención. Antes veníamos de B×T×D. El reshape separa D en H y d; la permutación coloca H antes de T. Al volver, se invierte ese movimiento antes de reunir H y d otra vez.

Un error común consiste en hacer un reshape directo entre B×T×H×d y B×H×T×d sin permutar. El número total de elementos coincide, pero el significado de sus posiciones no. Una prueba con valores que codifican sus coordenadas permite verlo sin entrenar nada.

::: ejemplo Codificar coordenadas para detectar una permutación
Asigna a cada elemento el valor 1000·b+100·t+10·h+j. Después de reorganizarlo, puedes leer el número y reconstruir a qué batch, posición, cabeza y componente pertenecía. Un array de ceros no revela ese error.
:::

## GQA: compartir claves y valores

Grouped-query attention permite más cabezas de consulta que de claves y valores. Varias consultas comparten el mismo grupo de K y V. La familia incluye configuraciones intermedias entre una cabeza KV compartida y una por cada consulta. [@gqa]

Nuestro ejemplo exige que H sea divisible por el número de cabezas KV, HK. Con H=4 y HK=2, cada grupo KV sirve a dos cabezas de consulta. La asociación entre grupos debe mantenerse al expandir o indexar.

La implementación didáctica expresa la repetición mediante movimientos de tensor. Es correcta para el contrato del modelo, pero una implementación de alto rendimiento puede evitar materializar copias. Esa diferencia vuelve a separar una arquitectura matemática de su realización eficiente.

## Qué memoria crece más deprisa

Q, K, V y O crecen linealmente con T para B,H,d fijos. La matriz de puntuaciones crece como T². Duplicar T multiplica por cuatro sus elementos.

Con B=1,H=8,T=4096 y FP32, una sola matriz T×T por cabeza ocupa $1\cdot8\cdot4096^2\cdot4$ bytes, unos 512 MiB. Si además guardamos probabilidades y gradientes semejantes, la presión aumenta. Esta es una cuenta de tamaño, no una medición de un framework concreto.

La atención eficiente que estudiaremos después no elimina necesariamente los productos entre todas las parejas. Evita materializar algunas matrices grandes y reorganiza el acceso a memoria. Confundir reducción de almacenamiento con reducción de complejidad aritmética lleva a expectativas equivocadas.

## El retroceso de atención por etapas

Si O=PV, el adjunto de V es $P^T\overline O$ y el de P es $\overline O V^T$. Para una fila de softmax, el adjunto de S puede calcularse sin una matriz jacobiana completa:

```math
\overline S=P\odot\left(\overline P-
\operatorname{sum}(P\odot\overline P,\text{eje final})\right).
```

La suma se conserva como dimensión de tamaño uno para que pueda expandirse sobre la fila. El símbolo $\odot$ representa multiplicación por elemento.

Finalmente, para S=QKt dividido por raíz de d, obtenemos los gradientes de Q y K mediante dos productos matriciales y el mismo factor. Las posiciones enmascaradas deben contribuir cero.

En Lumbre, la operación de atención online usa precisamente un VJP que reconstruye una atención densa para el retroceso. Eso proporciona gradientes comprobables, pero no un backward de memoria lineal. El proyecto final desarrolla cómo mejorar esa limitación sin esconderla.

# RMSNorm, RoPE y SwiGLU: entender cada pieza del decoder {#ch:decoder-piezas}

## Normalizar sin borrar la identidad de la fila

Las activaciones pueden cambiar de escala durante el entrenamiento. RMSNorm divide cada fila por una medida de su magnitud y aplica un peso aprendido por componente. A diferencia de una normalización que resta la media, esta operación usa la raíz de la media de cuadrados. [@rmsnorm]

Para x=[3,4], la media de cuadrados es 12.5 y su raíz aproximadamente 3.536. Sin el pequeño epsilon y con pesos uno, la salida es aproximadamente [0.849,1.131]. No obligamos a que la media sea cero; controlamos otra propiedad de escala.

```math
r=\left(\frac1D\sum_j x_j^2+\epsilon\right)^{-1/2},\qquad
 y_j=x_jr\gamma_j.
```

El peso $\gamma$ tiene D componentes y se comparte entre todas las filas. Su gradiente suma contribuciones de batch y posiciones. Este es otro uso concreto de la regla de unbroadcast.

## Derivar RMSNorm para comprobar el grafo

Sea g el adjunto de y y definamos $u_j=g_j\gamma_j$. El gradiente de x resulta:

```math
\overline x_j=r u_j-\frac{r^3x_j}{D}\sum_k u_kx_k.
```

El primer término es la dependencia directa de $x_jr$. El segundo refleja que cambiar cualquier componente modifica la magnitud r que se comparte con toda la fila. Si tratáramos r como una constante, perderíamos ese segundo término.

La implementación de Lumbre no escribe esta fórmula como kernel especial: construye RMSNorm a partir de multiplicación, media, suma, raíz, división y peso. El autodiff produce la derivada compuesta. La fórmula cerrada sirve como referencia independiente para un futuro kernel fusionado.

## RoPE: rotar parejas según la posición

Sin información posicional, una colección de vectores no expresa por sí sola el orden que queremos distinguir. RoPE aplica rotaciones dependientes de la posición a consultas y claves. [@rope]

Empieza por una pareja [a,b] y un ángulo theta. Su rotación es:

```math
[a',b']=[a\cos\theta-b\sin\theta,
          a\sin\theta+b\cos\theta].
```

Con ángulo cero no cambia nada. Con un cuarto de vuelta, [1,0] se convierte en [0,1]. Las normas se conservan en la aritmética ideal. En el modelo, diferentes parejas usan frecuencias diferentes y theta depende de la posición.

La implementación del paquete empareja la primera mitad de componentes con la segunda mitad. Otras implementaciones emparejan componentes consecutivas. Ambas convenciones pueden expresar rotaciones, pero sus pesos y layouts no son intercambiables sin una conversión coherente.

::: cuidado Una forma correcta no demuestra una convención correcta
Dos implementaciones de RoPE pueden devolver exactamente el mismo shape y utilizar los mismos senos y cosenos, pero rotar parejas distintas. Importar un checkpoint exige comprobar esa convención, no solo el tamaño de las matrices.
:::

## Posición absoluta y relación relativa

Si rotamos q con un ángulo asociado a la posición i y k con uno asociado a j, su producto depende de la diferencia de esas rotaciones. Esa propiedad motiva el uso de RoPE para introducir relaciones posicionales en la atención.

No se deduce que un modelo entrenado con contexto corto funcione perfectamente con un contexto arbitrariamente largo. El comportamiento fuera de la distribución y las técnicas de extensión de contexto requieren evaluación. Una fórmula que acepta un índice grande no constituye una garantía de comprensión a esa longitud.

En nuestro ejemplo, senos y cosenos se preparan para la longitud estática del programa y se pasan como parámetros fijos. El trabajo inicial de construcción de esas tablas no es una operación diferenciable de entrenamiento.

## SwiGLU: una puerta que modula contenido

Después de la atención, el bloque aplica una red por posición. SwiGLU construye dos proyecciones: una produce contenido y otra una puerta suave. La puerta se transforma con SiLU y multiplica el contenido, antes de una proyección de vuelta a D. [@swiglu]

```math
u=xW_u,\qquad a=xW_g,\qquad
\operatorname{SiLU}(a)=\frac{a}{1+e^{-a}},\qquad
z=(\operatorname{SiLU}(a)\odot u)W_d.
```

La dimensión intermedia F puede ser distinta de D. Wu y Wg tienen forma D×F; Wd, F×D. La operación se aplica independientemente a cada posición después de que la atención haya mezclado información entre posiciones.

No es equivalente a aplicar ReLU a una sola proyección. Tampoco es una operación matricial única: tiene dos GEMMs iniciales, trabajo elementwise y otro GEMM. Un compilador puede fusionar algunos epílogos, pero debe respetar esa dependencia.

## Conexiones residuales: sumar una corrección

Un bloque residual produce x más una transformación de x. Podemos imaginar que la transformación propone una corrección en lugar de reemplazar por completo la representación. La forma de ambos sumandos debe coincidir.

En el decoder del paquete hay una normalización antes de la atención, una suma residual, otra normalización antes de SwiGLU y otra suma residual. El orden define una arquitectura pre-norm. Mover la normalización al otro lado de la suma no es una optimización algebraica gratuita: cambia el modelo.

```diagram
x: representación | Atención sobre norm(x) | x + corrección
```

El dibujo se repite para la subcapa feed-forward. En el código completo se conservan los nombres de cada peso, lo que permite relacionar una celda del checkpoint con la operación que utiliza.

## Embedding compartido con la salida

La tabla de embeddings transforma identificadores en vectores D. Al final, su transpuesta puede transformar una representación D en V logits. Compartir esos pesos reduce el número de parámetros y une dos usos de la misma tabla.

Para el autodiff significa que el mismo parámetro recibe contribuciones desde la entrada y desde la salida. El motor debe sumarlas. Si se crean dos parámetros diferentes con valores inicialmente iguales, no se ha implementado weight tying: durante entrenamiento podrían divergir.

En Lumbre, ambos usos apuntan al mismo UOp de parámetro. Esta decisión muestra por qué la identidad y el compartido de nodos eran importantes desde las primeras semanas.

# Ensamblar el Transformer y comprobarlo antes de entrenar {#ch:decoder-completo}

## El contrato del constructor

`Config` define vocabulario, batch, contexto, ancho, cabezas, cabezas KV, capas y dimensión intermedia. El constructor rechaza dimensiones no positivas, un ancho no divisible por las cabezas, cabezas no divisibles por HK y una dimensión de cabeza impar para la convención RoPE utilizada.

Esas comprobaciones forman parte de la interfaz. Un mensaje claro durante construcción es mejor que una lectura fuera de rango en el kernel número cuatrocientos.

```python
from lumbre.nn import Config, Decoder

config = Config(vocab=24, batch=2, context=32, dim=64,
                heads=2, kv_heads=1, layers=2, hidden=128)
model = Decoder(config, seed=7)
print(model.parameter_count())
print(model.logits.shape, model.loss.shape)
```

En esta configuración el recuento es 75,584 parámetros. Los logits tienen forma(2,32,24) y la pérdida es escalar. Son valores verificables con el constructor incluido.

## Contar parámetros para detectar duplicaciones

El embedding compartido aporta V·D. Cada bloque aporta matrices de consulta y salida de D×D, dos matrices KV de D×(HK·d), tres matrices del feed-forward con un total 3·D·F y dos vectores de normalización de tamaño D. Al final hay otro vector de normalización.

```math
N_{\text{param}}=VD+L\left(2D^2+2D(H_Kd)+3DF+2D\right)+D.
```

Para V=24, D=64, H=2, HK=1, d=32, F=128, L=2, el embedding aporta 1536; cada bloque 36992; la normalización final 64. Sumamos 1536+73984+64=75584.

Las tablas fijas de RoPE y la máscara no son parámetros aprendidos y no se incluyen en ese recuento. Si tu suma duplica el embedding de salida, probablemente has perdido la compartición de pesos.

## Seguir una capa con shapes

| Etapa | Forma de salida | Operación principal |
|---|---|---|
| Identificadores | B×T | Entrada int32. |
| Embedding | B×T×D | Gather. |
| Consultas | B×H×T×d | Proyección y movimientos. |
| Claves y valores | B×HK×T×d | Proyección y movimientos. |
| Atención por cabeza | B×H×T×d | Productos y softmax causal. |
| Reunión de cabezas | B×T×D | Permutación y reshape. |
| Feed-forward | B×T×D | SwiGLU. |
| Logits | B×T×V | Producto con embedding transpuesto. |

Escribe esta tabla para cualquier nueva configuración antes de ejecutar. Una dimensión que parece pequeña puede esconder una expansión costosa, como repetir K y V hasta H o materializar una matriz T×T.

## Prueba uno: el avance tiene números finitos

Inicializa los pesos con la semilla declarada. Carga todos los parámetros fijos que el programa utiliza y un batch de tokens válidos. Exige logits y pérdida finitos.

Un valor NaN inicial puede venir de una operación inválida, una fila de softmax completamente enmascarada, un parámetro sin inicializar o un índice fuera del vocabulario. No se corrige aumentando el número de pasos de entrenamiento.

La API de Lumbre detecta parámetros no cargados antes de ejecutar. En una aplicación más grande conviene diferenciar parámetros aprendidos, constantes y entradas de cada iteración para que el diagnóstico sea todavía más específico.

## Prueba dos: cambiar el futuro no cambia el pasado

Construye dos secuencias iguales salvo su último token. Compara logits anteriores. Deben coincidir dentro de la tolerancia. La posición final sí puede cambiar, porque ahora recibe un token distinto dentro de su contexto permitido.

Después cambia un token inicial. Es razonable que cambien posiciones posteriores: la prueba de causalidad no exige independencia del pasado. Un test que exige igualdad de toda la salida ante cualquier cambio de entrada validaría precisamente un modelo que ignora sus datos.

## Prueba tres: reducir a una instancia diminuta

Para gradientes numéricos usa una capa, ancho pequeño, pocas posiciones y vocabulario diminuto. Selecciona algunas coordenadas de matrices diferentes y del embedding compartido. Compara diferencias centrales con el gradiente compilado.

No es necesario perturbar todos los parámetros de un modelo grande para empezar. La prioridad es cubrir tipos de operación y caminos compartidos. Después se amplía la muestra o se usa una referencia independiente para una comprobación más extensa.

## Prueba cuatro: sobreajustar un batch deliberadamente

Fija un batch pequeño y repite actualizaciones sobre él. La pérdida debería poder descender de forma clara con una configuración adecuada. Este experimento no mide generalización: elimina variabilidad para comprobar que existe una señal de aprendizaje y que la actualización la aprovecha.

Si falla, vuelve a problemas menores: un parámetro escalar, una capa lineal, una clasificación de pocas clases. Un modelo grande combina demasiadas hipótesis para ser el primer depurador.

## Prueba cinco: el checkpoint conserva el significado

Guarda una ejecución corta, cierra el proceso y reanuda con la misma configuración. Verifica que no se reinicien los momentos de Adam ni el contador. Cambia deliberadamente el vocabulario o la longitud de contexto y comprueba que se rechaza una reanudación incompatible cuando ese es el contrato del formato.

El resultado final de estas pruebas es más valioso que un pantallazo de texto generado: demuestra que el compilador ha unido correctamente todas las etapas que empezaron con una suma de vectores.
