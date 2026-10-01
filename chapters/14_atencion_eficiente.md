# Atención online: resolver una fila sin guardar todos sus pesos {#ch:online}

::: idea Una media que cambia de escala
Queremos calcular una mezcla ponderada mientras llegan nuevas puntuaciones. El problema es que softmax necesita una normalización global. La solución será mantener un resumen que podamos reescalar cuando aparezca una puntuación mayor.
:::

## La receta densa guarda más de lo necesario

La atención densa calcula una fila de puntuaciones, obtiene sus exponenciales normalizadas y multiplica esos pesos por los valores. Si solo necesitamos la mezcla final, quizá podamos evitar almacenar todos los pesos a la vez.

Para una fila, la salida puede escribirse como un cociente:

```math
O=\frac{\sum_j e^{s_j}v_j}{\sum_j e^{s_j}}.
```

El numerador es un vector, porque cada valor v tiene varias componentes. El denominador es un escalar. Si las exponenciales fueran siempre manejables, podríamos acumular ambos conforme llegan términos. La dificultad es que $e^{s_j}$ puede desbordarse.

## Restar un máximo que todavía no conocemos

En softmax estable restábamos el máximo de todas las puntuaciones. En una lectura online, aún no sabemos si la próxima será mayor. Podemos usar el máximo visto hasta ahora y corregir la escala de lo acumulado cuando cambie.

Mantendremos tres objetos: m, máximo visto; l, suma de exponenciales en la escala de m; y u, suma ponderada de valores en esa misma escala. La salida parcial es u/l.

Al recibir una puntuación s con valor v, el nuevo máximo es $m'=\max(m,s)$. Lo antiguo estaba medido respecto a m y ahora debe medirse respecto a m'. El factor de conversión es $\alpha=e^{m-m'}$. La contribución nueva pesa $\beta=e^{s-m'}$.

```math
l'=\alpha l+\beta,\qquad u'=\alpha u+\beta v,\qquad m\leftarrow m'.
```

No hemos aproximado el denominador descartando términos. Hemos cambiado su escala común. En aritmética real, el cociente representa exactamente la misma atención sobre los elementos visitados. En coma flotante, el orden de operaciones puede introducir diferencias que debemos medir.

## Una ejecución entera con tres puntuaciones

Usamos puntuaciones 0,log 2 y log 4, y valores escalares 1,3 y 5. Sus exponenciales sin normalizar serían 1,2 y 4. La respuesta directa es $(1+6+20)/(1+2+4)=27/7$.

Primera puntuación: m=0, l=1 y u=1. La salida parcial es 1.

Segunda puntuación: el máximo cambia a log 2. El factor de lo antiguo es un medio y el nuevo peso es uno. Por tanto, l=0.5·1+1=1.5 y u=0.5·1+3=3.5. La salida parcial es 7/3, exactamente la mezcla de las dos primeras contribuciones.

Tercera puntuación: el máximo cambia a log 4. Volvemos a reescalar por un medio. Ahora l=0.5·1.5+1=1.75 y u=0.5·3.5+5=6.75. El cociente 6.75/1.75 es 27/7.

::: comprueba El invariante que demuestra la corrección
Después de visitar cualquier prefijo, m es su máximo, l es la suma de exponenciales de ese prefijo restando m y u es la suma de esas exponenciales multiplicadas por sus valores. Reescalar por el mismo factor mantiene ambos sumatorios en la escala nueva.
:::

## Inicialización y filas vacías

Podemos iniciar m en menos infinito, l en cero y u en cero. La primera puntuación finita produce factor antiguo cero y peso nuevo uno. Así no necesitamos un caso especial para el primer término, siempre que las operaciones y dominios estén controlados.

Una fila sin posiciones válidas no tiene una distribución softmax ordinaria: el denominador es cero. Hay que definir si se rechaza, se devuelve cero o se aplica otra convención. Nuestro operador causal de longitudes positivas incluye la posición propia, por lo que cada fila tiene al menos una contribución.

Si añadimos máscaras arbitrarias, ya no basta esa garantía. Los tests deben incluir una fila totalmente enmascarada y comprobar la política elegida.

## De un elemento a un bloque

Para trabajar con tiles, un bloque de puntuaciones puede producir su máximo local, suma exponencial local y suma ponderada local. Se combinan dos resúmenes con un máximo común, reescalando cada uno.

Sean los resúmenes A y B. Tomamos m igual al máximo de sus máximos. Multiplicamos la suma y el numerador de A por $e^{m_A-m}$ y los de B por $e^{m_B-m}$. Después sumamos. Es la misma idea, ahora aplicada a grupos.

Esa propiedad permite organizar una reducción por bloques sin materializar la matriz completa. El tamaño del bloque se elige según recursos y patrón de cómputo, no porque la matemática exija un número concreto.

## La operación nueva en Lumbre

La función `online_attention(q,k,v)` acepta tensores FP32 de igual forma terminados en T×D, con T y D positivos. Aplica atención causal y devuelve la misma forma. La implementación materializa sus operandos y asigna un resultado por fila.

```python
import numpy as np
from lumbre import param, Program, online_attention

q = param("q", (1,2,5,4))
k = param("k", (1,2,5,4))
v = param("v", (1,2,5,4))
o = online_attention(q,k,v)
rng = np.random.default_rng(63)
feed = {name:rng.normal(size=q.shape).astype("float32")
        for name in ("q","k","v")}
with Program([o], gemm="blocked") as p:
    result = p.run(feed)[0]
    assert result.shape == q.shape
    assert np.isfinite(result).all()
```

La prueba de finitud no basta para aceptar el operador. La batería compara además con atención densa y compara los gradientes de una pérdida cuadrática respecto a Q,K,V para varias formas, incluidas dimensiones irregulares.

## Qué memoria ahorra exactamente

El forward online no crea un tensor global T×T de puntuaciones o probabilidades. Su salida y operandos siguen teniendo tamaño proporcional a T·D. Nuestro helper usa el buffer de salida como acumulador de la fila.

Ese helper no es una implementación GPU de alto rendimiento: una ruta con un hilo por fila y escrituras frecuentes del acumulador no organiza la reutilización como un kernel FlashAttention especializado. La mejora de complejidad de almacenamiento no garantiza una buena realización física.

La diferencia entre algoritmo, mapping y hardware es justamente el tema del curso. La misma recurrencia puede formar parte de un kernel excelente o de uno lento según cómo se distribuyan sus datos y trabajo.

# Del algoritmo online a FlashAttention y sus derivados {#ch:flash}

## Qué idea investigamos en los artículos

FlashAttention plantea la atención exacta con una organización consciente del tráfico entre memoria rápida y memoria de mayor capacidad. FlashAttention-2 estudia además el reparto del trabajo, y FlashAttention-3 adapta el diseño a mecanismos de Hopper, entre otros cambios. No tratamos sus resultados de rendimiento como mediciones propias. [@flash1;@flash2;@flash3]

La lección común es que contar multiplicaciones no basta. Una implementación puede hacer un número similar de operaciones y tardar mucho menos si deja de materializar intermediarios y coordina mejor transferencia, cómputo y paralelismo.

El repositorio consultado en 2026 incluye rutas recientes basadas en CuTe DSL. Eso no autoriza a inventar una publicación científica para cada nombre de versión del código ni a suponer que toda GPU soporta las mismas rutas. [@flash-repo]

## Reutilizar Q mientras recorremos K y V

Supón que un bloque de trabajo posee varias filas de Q. Las mantiene cerca de la unidad de cómputo y recorre K,V por bloques. Cada bloque produce puntuaciones parciales, actualiza máximos y denominadores y añade una contribución al resultado.

La forma de los tiles controla varias cosas a la vez. Más filas de Q pueden aumentar reutilización de K,V, pero consumen más acumuladores. Un bloque más ancho en K puede aprovechar mejor una operación matricial, pero requiere más memoria temporal. Hay que equilibrar recursos, no maximizar cada dimensión por separado.

```diagram
Tile de Q residente | Recorrer tiles K y V | Resumen online y salida
```

En la región causal, algunos bloques quedan completamente fuera del triángulo permitido y pueden omitirse. Los bloques que cruzan la diagonal necesitan una máscara por elemento. Separar esos dos casos evita realizar comprobaciones innecesarias en todo el dominio.

## El coste no aritmético puede dominar

Una puntuación requiere productos y sumas, pero softmax requiere máximo, exponencial, suma y reescalado. Esas operaciones no tienen necesariamente el mismo throughput que la multiplicación matricial.

Si aceleramos mucho los GEMMs, el resto puede convertirse en el cuello de botella. La optimización deja de consistir en «hacer más tensor cores» y pasa a organizar el trabajo escalar o vectorial, reducir conversiones y solapar etapas.

La lectura de un kernel debe contar quién ejecuta cada parte. ¿Todos los warps participan en el GEMM? ¿Quién actualiza el máximo? ¿Dónde vive el acumulador de salida? ¿Hay participantes esperando mientras otros hacen softmax? Estas preguntas conectan el algoritmo con el perfil temporal.

## Guardar log-sum-exp para el retroceso

Para reconstruir una probabilidad basta con conocer su puntuación y el logaritmo del denominador correspondiente. Si guardamos por fila $LSE=m+\log l$, podemos recuperar $P_{ij}=e^{S_{ij}-LSE_i}$ cuando volvemos a calcular un bloque de puntuaciones.

Así evitamos guardar toda P durante el avance. En el retroceso recomputamos bloques y acumulamos gradientes. Cambiamos memoria por cómputo adicional, pero ese cómputo puede ser barato comparado con trasladar una matriz grande.

No basta con eliminar P del checkpoint del forward. El backward debe estar diseñado para no volver a materializarla completa. Nuestra primera operación online todavía usa un VJP denso; por eso su ahorro de memoria durante entrenamiento es mucho menor que el de un forward aislado.

## Una derivación que reduce un término del backward

Recordemos que $\overline P=\overline O V^T$. Para una fila i, el término de reducción de softmax es $D_i=\sum_jP_{ij}\overline P_{ij}$. Sustituyendo la definición y reordenando sumas obtenemos:

```math
D_i=\sum_d \overline O_{id}\left(\sum_jP_{ij}V_{jd}\right)
=\sum_d\overline O_{id}O_{id}.
```

Podemos calcular D a partir de la salida O y su adjunto, sin guardar todas las derivadas de P. Después cada bloque usa $\overline S_{ij}=P_{ij}(\overline P_{ij}-D_i)$.

Esta igualdad es una herramienta concreta para diseñar el backward. Es una transformación algebraica de sumas; en coma flotante, el nuevo orden puede cambiar el redondeo y necesita tolerancias acordes.

## La propiedad de los gradientes importa

Un bloque de consultas puede acumular su propio gradiente de Q con facilidad. Pero varios bloques de consultas contribuyen a las mismas filas de K y V. Necesitamos una estrategia para reunir esas contribuciones sin carreras.

Una opción usa acumulación atómica. Otra asigna a un kernel la propiedad de un tile de K,V y recorre las consultas necesarias. Otra produce parciales y ejecuta una reducción posterior. Cada estrategia cambia memoria, paralelismo, determinismo y número de lanzamientos.

No existe una solución universalmente mejor. Para un proyecto de semestre, una reducción explícita de parciales puede ser más fácil de verificar antes de introducir atomics o kernels persistentes.

::: cuidado La carrera no se arregla con tolerancia numérica
Dos escrituras no coordinadas pueden perder contribuciones. El resultado quizá cambie entre ejecuciones y a veces parezca cercano. Aumentar `atol` no convierte una carrera en una aproximación legítima.
:::

## Dropout y reproducibilidad del retroceso

Si añadimos dropout a los pesos de atención, el backward necesita aplicar la misma selección aleatoria que el forward. Guardar toda la máscara consume memoria; regenerarla requiere un generador y un mapeo de contadores reproducibles.

Cambiar el tiling no debería cambiar sin explicación qué números aleatorios corresponden a cada posición si queremos comparar la misma ejecución. Una semilla global no basta si el orden de consumo del generador depende del scheduler.

El modelo incluido no usa dropout. Al incorporarlo, se añade una operación con estado o una generación funcional por contador, junto a sus pruebas. No se considera una modificación puramente elementwise sin consecuencias sobre el runtime.

## Atención de entrenamiento y de generación

En entrenamiento suelen existir muchas consultas por secuencia. En generación incremental puede haber una consulta nueva y un historial largo de K,V. El patrón de reutilización y paralelismo cambia.

Una configuración excelente para prefill no tiene por qué serlo para decode. Si el número de filas de Q es pequeño, quizá no haya suficientes bloques para ocupar la máquina. Se pueden dividir otras dimensiones o usar varias particiones del historial con una combinación de resúmenes.

La máscara también necesita una definición para longitudes de Q y K diferentes. Nuestro operador educativo exige la misma longitud y causalidad simple; no se presenta como una API general de atención con caché o longitudes irregulares.

## Reproducción de un artículo frente a inspiración

Implementar la recurrencia online y medirla en CPU es una reproducción de una idea algorítmica, no una reproducción completa de los resultados de un artículo GPU. Para esto último harían falta arquitectura, tipos, tamaños, baseline y protocolo compatibles, además de la implementación correspondiente.

En el proyecto final usamos tres etiquetas: derivación comprobada, implementación local validada y comparación externa pendiente. Mantener esas etiquetas separadas permite realizar investigación útil incluso con hardware limitado, sin inflar la conclusión.

# Kernels persistentes, planificación y límites del autotuning {#ch:persistentes}

## Cuando los tiles no tienen el mismo trabajo

En una matriz regular, podemos asignar un tile de salida a cada bloque. En una mezcla de expertos o una atención con longitudes distintas, algunos tiles pueden requerir mucho más trabajo que otros. Una partición estática puede dejar unidades ociosas mientras unas pocas terminan.

Un kernel persistente mantiene un conjunto de trabajadores que van tomando tareas. La idea se parece a una cola de pedidos: al terminar uno, el trabajador solicita otro. Así puede adaptarse mejor a una distribución irregular.

Pero la cola cuesta. Hay sincronización, metadatos, decisiones y recursos ocupados durante más tiempo. Para una operación pequeña, ese mecanismo puede ser más caro que el trabajo que reparte.

## Un scheduler de tareas como modelo independiente

Antes de escribir un kernel persistente, crea un simulador pequeño que reciba tareas con costes conocidos y compare asignación fija con una cola. El simulador permite estudiar balance idealizado, no latencias reales de la GPU.

Para tareas de costes [8,1,1,1,1] y dos trabajadores, una asignación por bloques contiguos puede dejar un reparto 9 frente 3. Una cola que toma la primera tarea larga y reparte las pequeñas entre el otro trabajador termina en 8, ignorando costes de coordinación. El máximo de las cargas limita la finalización.

Ahora añade un coste de 0.5 por tarea tomada. La ganancia cambia. Este ejemplo obliga a incluir overhead antes de asumir que una planificación dinámica siempre gana.

## El orden de tareas también afecta a la caché

Dos tiles cercanos pueden reutilizar partes de A o B. Una cola que equilibra cómputo pero destruye localidad puede trasladar más datos. El scheduler debe considerar no solo cuánto cuesta una tarea aislada, sino qué datos comparte con otras.

En problemas agrupados, ordenar por tamaños o agrupar tareas compatibles puede reducir cambios de configuración. Sin embargo, ordenar también cuesta y puede aumentar la latencia de un pedido que esperaba ser atendido primero.

Un compilador de servicio tiene objetivos diferentes de un benchmark de throughput total. Es necesario especificar si optimizamos la suma de tiempos, el tiempo de la última tarea o una latencia por solicitud.

## Especializar sin compilar infinitas variantes

Cada tamaño, dtype, layout y arquitectura podría producir una configuración distinta. Compilar todas las combinaciones no es viable. Se necesitan familias de especialización y una alternativa genérica.

Una política sencilla agrupa tamaños por intervalos y alinea tiles. Otra utiliza un pequeño conjunto de candidatos elegidos por un modelo de coste. La caché debe identificar todas las propiedades que cambian el significado del programa, no solo el tamaño total de elementos.

Si dos tensores tienen el mismo número de elementos pero strides distintos, reutilizar un kernel especializado para contigüidad puede ser incorrecto. Si el destino cambia de arquitectura, un binario anterior puede ser incompatible aunque los shapes coincidan.

## Autotuning con un presupuesto explícito

Supongamos que probar cien candidatos tarda diez segundos y ahorra un microsegundo por ejecución. Harían falta unos diez millones de ejecuciones para recuperar ese coste, sin contar almacenamiento o compilación adicional. Es una cuenta de amortización, no una opinión sobre si el tuning es elegante.

Un autotuner puede decidir no buscar cuando la operación solo se ejecutará unas veces. En entrenamiento de muchos pasos, el mismo coste puede amortizarse bien. La frecuencia esperada de uso forma parte del modelo de decisión.

El proyecto de autotuning del libro exige registrar también candidatos descartados por incorrectos y por incompatibilidad. Seleccionar el más rápido entre programas que no calculan lo mismo es una optimización inválida.

## El mejor candidato de una muestra puede sobreajustar

Si elegimos una configuración usando solo matrices cuadradas múltiplos de 128, quizá falle o pierda mucho en dimensiones de borde. Reservamos un conjunto de formas no usado durante selección y comprobamos allí la política.

También necesitamos estabilidad. Una configuración que gana por una diferencia menor que el ruido temporal no merece una conclusión fuerte. Podemos preferir la opción más simple o robusta cuando las medidas no distinguen claramente.

::: comprueba El resultado de esta semana
Propón kernels con guardas, selección y respaldo. Explica qué preserva su corrección y cuántas ejecuciones amortizan la búsqueda.
:::
