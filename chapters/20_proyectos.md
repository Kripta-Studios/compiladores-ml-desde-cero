# Proyecto final A. Atención con memoria acotada: una réplica semántica completa {#project:atencion}

::: idea Pregunta de investigación
¿Podemos sustituir la atención causal densa de nuestro compilador por una recurrencia online que conserve resultados y reduzca almacenamiento? Después, una pregunta distinta: ¿esa sustitución mejora también el tiempo en la CPU disponible?
:::

## Alcance y contribución del proyecto de referencia

Este es un proyecto desarrollado, no únicamente una lista de tareas futuras. El paquete contiene el operador `online_attention`, su generación de C, una regla de gradiente por recomposición, pruebas numéricas y un benchmark con resultados locales. La contribución es una **implementación didáctica original de una recurrencia conocida**, integrada en un compilador propio y estudiada con hipótesis separadas de corrección, memoria y velocidad.

La fuente conceptual de atención eficiente por reorganización del tráfico es la familia FlashAttention. La réplica de este proyecto es semántica y acotada: no reproduce íntegramente sus kernels GPU, su partición entre warps ni su backward optimizado. Esa distinción es fundamental para atribuir correctamente qué se ha conseguido. [@flash1;@flash2;@flash3]

## Definir la función exacta

Las entradas `Q,K,V` tienen la misma forma `(B,H,T,D)`, tipos FP32 y dimensiones positivas. Cada fila de consulta en posición $i$ utiliza claves $0\ldots i$, con factor de escala $1/\sqrt D$. La salida tiene la misma forma. No se admite en este operador una máscara arbitraria ni longitudes diferentes de consulta y clave.

La referencia densa calcula puntuaciones, añade una máscara causal, normaliza mediante softmax estable y multiplica por los valores. El candidato procesa claves por fila sin conservar toda la matriz de puntuaciones. La equivalencia buscada es numérica dentro de tolerancias justificadas para el conjunto de prueba; no se exige identidad bit a bit entre órdenes de reducción distintos.

## Derivación del estado que basta conservar

Para un conjunto de claves ya procesadas, guardamos $m$, el máximo de puntuaciones; $l$, la suma de exponenciales desplazadas; y $a$, el vector de suma ponderada de valores. La salida parcial normalizada sería $a/l$. Si llega una puntuación $s$ con valor $v$, el nuevo máximo es $m'=\max(m,s)$.

Todas las contribuciones antiguas estaban expresadas respecto a $m$. Para expresarlas respecto a $m'$, se multiplican por $\alpha=\exp(m-m')$. La nueva contribución pesa $\beta=\exp(s-m')$. Por tanto:

```math
l'=\alpha l+\beta,\qquad a'=\alpha a+\beta v.
```

No hemos aproximado la normalización: hemos cambiado una representación intermedia. El factor común que se introduce en numerador y denominador se cancela. En FP32 aparecen redondeos, de ahí la comparación tolerada. La estabilidad mejora frente a calcular exponenciales de puntuaciones sin desplazar, pero no elimina todos los problemas posibles de precisión.

::: ejemplo Una prueba inductiva, con su base
Antes de procesar claves, toma denominador cero y acumulador cero. La primera clave válida establece el máximo y aporta peso uno en la escala elegida. Después de cada incorporación, el denominador y acumulador representan exactamente, en aritmética real, todas las claves procesadas respecto al máximo actual.

El paso inductivo es la reescala anterior: transforma cada contribución antigua por el mismo factor y añade la nueva. Al procesar la última clave permitida, dividir acumulador entre denominador produce la atención causal definida. La prueba requiere al menos una clave válida por fila; una fila completamente enmascarada necesitaría otro contrato.
:::

## Integración en el compilador

La operación online actúa como una frontera de kernel. Sus operandos se materializan contiguos para que la función C reciba el layout que espera. El renderer genera un recorrido por filas y, dentro de cada una, un bucle sobre claves permitidas. El acumulador de salida mantiene $D$ componentes y se normaliza al final.

Esta elección conserva la comprensión del algoritmo, pero renuncia a algunas optimizaciones. No hay un reparto avanzado entre hilos CPU, ni tiles vectoriales de claves, ni pipeline GPU de copias. El mismo operador tiene una ruta GPU simple en el renderer, pero no se ha validado en dispositivo. Ninguna tabla de este proyecto debe mezclar esa ruta no ejecutada con resultados CPU.

El backward del operador se expresa con primitivas densas. Esa decisión permite integrar entrenamiento y comprobar gradientes sin escribir de inmediato otro kernel complejo. El precio es que el grafo de gradientes puede recuperar almacenamiento cuadrático. La memoria de forward y la de un paso completo se informan por separado.

## Plan de pruebas que acompaña al resultado

La batería compara forward con atención densa para varias formas y compara gradientes respecto a las tres entradas. Los tamaños pequeños permiten identificar una primera coordenada incorrecta. También se verifica causalidad en el decoder: cambiar el futuro no debe alterar salidas anteriores.

Para fortalecer el proyecto, el estudiante puede añadir entradas construidas para hacer cambiar varias veces el máximo, valores grandes con signos opuestos y longitudes cercanas al límite de un tile. Cada caso debe responder a una hipótesis de fallo. Repetir mil veces una forma cómoda con valores normales no sustituye una prueba donde la primera clave domina y las siguientes son muy negativas.

En una réplica más avanzada, se guarda también el logaritmo del denominador en la escala original, conocido como log-sum-exp. Permite reconstruir probabilidades en backward sin conservarlas todas. La fórmula $p_j=\exp(s_j-L)$ requiere que $L$ corresponda exactamente a la fila y máscara del forward. Un error de índice de fila puede producir probabilidades plausibles que suman mal o mezclan ejemplos.

## Resultados locales del forward

La tabla se obtiene del informe `reports/kernels.json`. Los tiempos son medianas de veinte repeticiones, después de tres calentamientos, con entradas residentes. La arena es la definida por Lumbre; no es el pico total del proceso. Todas las filas usan `(B,H,D)=(1,2,32)`.

| Contexto T | Densa: microsegundos / bytes | Online: microsegundos / bytes |
|---|---|---|
| 16 | 10,15 / 22.528 | 9,70 / 16.384 |
| 64 | 106,44 / 131.072 | 136,31 / 65.536 |
| 128 | 437,53 / 393.216 | 624,61 / 131.072 |

En contexto 128, la arena online es un tercio de la densa, pero el tiempo es aproximadamente 1,43 veces el de la densa. En contexto 64 ocurre también una pérdida de velocidad. El primer tamaño es demasiado pequeño para convertir una diferencia reducida de tiempo en una conclusión amplia sin estudiar dispersión y más repeticiones.

La conclusión válida es doble: **la implementación evita el intermedio denso y reduce la arena observada; esa mejora no garantiza aceleración en esta CPU**. No se publican cifras de ancho de banda efectivo ni ocupación GPU porque no se midieron. Tampoco se atribuye una causa concreta sin una ablación adicional.

## Qué experimento realizar después

La siguiente variante debería procesar varias claves juntas. Puede calcular un máximo de bloque, un denominador de bloque y una suma ponderada, y combinar ese estado con el anterior. Así se mantiene la prueba por composición de estados, pero cambia la granularidad del trabajo y se abren oportunidades de vectorización.

Otra variante conserva forward y cambia únicamente la organización de las componentes $D$, buscando que el compilador genere instrucciones vectoriales. Se inspecciona el ensamblador y se miden varios valores de $D$. Si ambas variantes cambian a la vez tamaño de bloque, precisión y número de hilos, ya no se puede identificar qué mecanismo explica el resultado.

Para GPU, la extensión debe diseñar memoria compartida, participantes y barreras antes de copiar un código de atención. La verificación de resultados es necesaria, pero no suficiente para carreras. La ruta de tensor cores requiere además distinguir tipos de entrada, acumulación y layout colectivo. Se utiliza hardware compatible y herramientas de diagnóstico reales; un emulador lógico no certifica esas propiedades.

## Entregables y defensa del proyecto

La memoria del proyecto contiene función, derivación, integración, pruebas, tabla y discusión. El código debe incluir un modo de referencia que no desaparezca cuando el candidato empiece a funcionar. Los resultados en JSON conservan muestras y configuración. Una tabla negativa se mantiene, porque es parte del hallazgo.

La defensa se aprueba cuando el estudiante explica por qué la reescala no pierde contribuciones, por qué su backward todavía puede usar memoria cuadrática y por qué una CPU no reproduce automáticamente los resultados de una publicación GPU. Esas tres respuestas demuestran comprensión de matemática, compilación y evaluación, respectivamente.

# Proyecto final B. Pasaporte de índices para un acelerador desconocido {#project:pasaporte}

## La idea original

Antes de portar un compilador a una máquina que no conocemos, podemos separar una pregunta que sí sabemos resolver: **¿el reparto lógico escribe cada salida válida exactamente una vez y evita direcciones fuera de rango?** El proyecto construye un pasaporte verificable del mapa de índices. No inventa una GPU virtual completa ni simula ciclos; comprueba una propiedad finita y útil antes de gastar tiempo en el hardware.

El paquete incluye `lumbre/passport.py`, pruebas y `examples/verify_launch.py`. Se ejecutó una colección de quince mapas válidos y se rechazó una variante con stride incorrecto. Este resultado es validación del mapa lógico por enumeración CPU, no ejecución CUDA/HIP/Ascend ni una prueba de ausencia de carreras de memoria compartida.

## Definición del mapa

Una matriz tiene $M$ filas y $N$ columnas. Los tiles tienen $T_M$ filas y $T_N$ columnas. Para un tile $(b_y,b_x)$ y una posición local $(t_y,t_x)$, la salida lógica es $r=b_yT_M+t_y$, $c=b_xT_N+t_x$. Si $r<M$ y $c<N$, su dirección contigua es $rN+c$.

El pasaporte enumera participantes lógicos, aplica la máscara y cuenta escritores de cada dirección. Deben cumplirse tres condiciones: ninguna dirección válida sin escritor, ninguna con varios escritores y ninguna escritura fuera de la región. El conteo permite distinguir pérdida de cobertura, duplicación y desbordamiento.

```python
from lumbre.passport import Launch, verify

correcto = verify(Launch(17, 23, 16, 16))
assert correcto["status"] == "LOGICAL_MAP_VALIDATED"
assert correcto["valid_writes"] == 17 * 23

erroneo = verify(Launch(3, 5, 2, 4),
                 lambda fila, columna, s: fila * s.rows + columna)
assert erroneo["status"] == "FAILED"
print(erroneo["missing"], erroneo["duplicate_writes"])
```

El segundo mapa multiplica por número de filas cuando debía multiplicar por número de columnas. En una matriz cuadrada ambos números coinciden y el fallo queda oculto. Por eso las formas rectangulares son esenciales en una prueba de layouts.

## Resolver el contraejemplo completo

Para $M=3,N=5$, la dirección correcta de la segunda fila empieza en 5. El mapa defectuoso la empieza en 3. La primera fila escribe 0,1,2,3,4; la segunda escribe 3,4,5,6,7; la tercera 6,7,8,9,10. Se repiten 3,4,6,7 y quedan sin escribir 11,12,13,14.

No hace falta ejecutar aritmética para encontrar el fallo: la propiedad de propiedad de salida ya es falsa. Si después un kernel devolviese algunos números correctos, sería accidental. Aumentar tolerancias no repara direcciones duplicadas. La corrección es sustituir el stride por $N$, y después volver a comprobar la colección completa.

## Del ensayo finito a un argumento general

La enumeración prueba las formas elegidas. Podemos añadir una demostración para el mapa correcto. Cada fila $r$ tiene una descomposición única en cociente y resto respecto a $T_M$; cada columna $c$, respecto a $T_N$. Por tanto, cada coordenada válida corresponde a un único tile y una única posición local.

Además, para $0\le r<M$ y $0\le c<N$, la dirección $rN+c$ queda entre 0 y $MN-1$. Si dos coordenadas producen la misma dirección, sus cocientes y restos respecto a $N$ coinciden, luego fila y columna son iguales. Hay cobertura, unicidad y rango. La máscara excluye participantes que no corresponden a una coordenada válida.

::: ejemplo Qué no demuestra el argumento
No demuestra que un compilador GPU traduzca bien los índices, que todos los hilos lleguen a una barrera, que una instrucción tensorial tenga el layout correcto ni que el dispositivo admita el número de participantes. La prueba se refiere al mapa matemático definido. Un port debe conectar esa prueba con el código generado y con capacidades verificadas del hardware.
:::

## Por qué sirve para hardware extraño

Un acelerador puede organizar trabajo en warps, wavefronts, grupos o unidades matriciales con nombres diferentes. Antes de utilizar sus características, el pasaporte conserva una representación neutral de qué salidas posee cada participante lógico. Después una capa de mapeo asigna participantes a unidades reales.

Si una unidad real procesa varios participantes lógicos, debe explicar cómo los recorre. Si varios participantes cooperan para una salida, el pasaporte simple de un escritor por elemento deja de ser suficiente: hay que representar contribuciones y una reducción o una escritura final única. El contrato debe evolucionar; no basta con desactivar la comprobación de duplicados para que aparezca verde.

En un port Ascend, por ejemplo, habrá que estudiar por separado las unidades de cómputo, buffers y mecanismos de sincronización de la versión concreta del entorno. El material de Huawei y los repositorios Ascend de DeepSeek ayudan a identificar esos contratos, pero no se consideran una interfaz intercambiable con CUDA. [@ascend-samples;@deepgemm-ascend;@deepep-ascend]

## TDD del proyecto

La primera prueba exige que una matriz vacía no produzca accesos. La segunda utiliza un elemento. La tercera coincide con un tile completo. La cuarta obliga a enmascarar bordes. La quinta usa una forma rectangular grande respecto al tile. Después se añaden fallos intencionales: stride incorrecto, desplazamiento de uno y dirección fuera de rango.

El verificador tiene límites explícitos de tamaño para evitar enumeraciones accidentales enormes. Un rechazo por límite de recursos no debe etiquetarse como mapa incorrecto: es una solicitud fuera del dominio del verificador. Del mismo modo, los límites didácticos de tile no afirman cuáles son los límites físicos de CUDA o HIP.

```bash
python examples/verify_launch.py --output reports/mi_pasaporte.json
python -m pytest tests/test_passport.py -q
```

El informe conserva el alcance de la comprobación en cada resultado. La etiqueta `LOGICAL_MAP_VALIDATED` es deliberadamente más estrecha que «backend validado». Una buena nomenclatura evita que un informe de simulación se convierta después, por descuido, en una afirmación de pruebas en hardware.

## Extensión a layouts de fragmentos

Una ampliación interesante describe la propiedad de cada elemento como un par «participante, posición local». Para una instrucción matricial, el verificador puede comprobar que la distribución de fragmentos cubre las coordenadas esperadas y que las transformaciones entre layouts son biyectivas dentro de su dominio.

No conviene empezar por una instrucción enorme. Construye una matriz $4\times4$ y dos repartos diferentes entre cuatro participantes. Escribe las tablas completas y su conversión. Después añade un tamaño no divisible y decide si hay padding, máscara o una ruta de fallback. Esta metodología convierte el port en una secuencia de contratos pequeños en lugar de una búsqueda ciega de código que compile.

## Entrega del proyecto

El resultado mínimo es el verificador ejecutado, quince casos válidos, tres clases de fallos y la demostración del mapa contiguo. El resultado avanzado conecta el pasaporte con un renderer de Lumbre y compara sus expresiones de índices con la tabla esperada. El nivel de hardware exige además compilación, ejecución y herramientas de diagnóstico del dispositivo, y permanece pendiente hasta contar con esa evidencia.

# Proyecto final C. Competitividad CPU y GPU con un contrato medible {#project:competitivo}

## Convertir «competitivo» en una pregunta que pueda fallar

El temario propone producir código competitivo con implementaciones modernas. Es un objetivo exigente y valioso, pero necesita una definición operacional. Este proyecto no declara que Lumbre ya iguale a PyTorch, BLAS o los mejores kernels CUDA. Diseña la comparación necesaria y proporciona una base CPU mejorada que el estudiante puede extender.

Elige una familia de carga: GEMM de proyecciones de un decoder, reducciones de normalización o convoluciones de una red pequeña. Define formas y tipos antes de ajustar. Una colección útil mezcla matrices pequeñas, grandes, irregulares y relaciones entre dimensiones distintas. Una suite formada solo por la forma que mejor le va a tu kernel no permite una conclusión general.

## Baseline de tres niveles

El primer baseline es el bucle de referencia del compilador, porque ayuda a aislar el efecto de cada transformación propia. El segundo es una implementación madura de la misma operación en el entorno disponible. El tercero es el paso del modelo completo, donde importan fusión, llamadas y memoria.

Estas comparaciones responden a preguntas diferentes. Ganar al bucle propio muestra una mejora interna. Acercarse a una biblioteca madura muestra competitividad de kernel. Mejorar el paso completo muestra impacto de sistema. Ninguna implica automáticamente las otras. Una biblioteca puede ganar el GEMM aislado y perder tiempo de conversión de layout al integrarse en una cadena concreta.

Para una referencia PyTorch, conserva código, versión y configuración, y comprueba si se utiliza ejecución eager o compilada. No compares una versión compilada del candidato con una referencia que incluye transferencias o inicialización sin declararlo. La referencia debe tener una oportunidad razonable de usar sus rutas habituales.

## Una escalera de optimizaciones CPU

La base entregada ofrece un microkernel con acumuladores y bloques de reducción. La primera extensión puede separar packing de la ejecución: reorganizar $B$ en un layout que permita accesos contiguos en el microkernel. Mide packing por separado y extremo a extremo; si la matriz cambia cada llamada, no puedes amortizarlo como si fuera un peso constante.

La segunda extensión usa vectorización explícita o verifica la autovectorización del compilador. La tercera reparte tiles entre hilos sin que dos escriban la misma región. La cuarta estudia afinidad, memoria NUMA y tamaño de trabajo. No implementes todas antes de obtener una referencia correcta de una sola hebra: perderías un punto de comparación interpretable.

Un diseño de packing debe describir su dirección. Para un bloque de $K_B\times N_B$, un formato simple copia filas contiguas de $B$ y rellena el borde con cero. El microkernel usa el mismo orden de $K$ si se quiere conservar cierta comparabilidad numérica. Formatos más complejos pueden intercalar grupos para SIMD, pero requieren otra prueba de mapa.

## Repartir trabajo sin false sharing innecesario

Si cada hilo recibe tiles de salida completos, las escrituras no se solapan lógicamente. Aun así, dos tiles pequeños podrían compartir líneas de caché en sus bordes y perjudicar rendimiento. Esa interferencia no es una carrera de datos si las posiciones son distintas, pero puede aumentar coherencia y tráfico.

No confundas las dos clases de problema: una carrera compromete corrección; false sharing puede comprometer tiempo. El remedio depende del diagnóstico. Añadir un mutex a todas las escrituras puede evitar una carrera real, pero destruir paralelismo y no ser la solución apropiada a un reparto mal diseñado.

## Escalera GPU y puertas de aceptación

Empieza por un hilo por salida, después tiling compartido, luego microtiles por hilo y finalmente instrucciones matriciales si el dispositivo las admite. Cada nivel mantiene la versión anterior como oráculo de comparación y fallback. No actives el nivel siguiente solo porque compila.

| Puerta | Evidencia exigida | Motivo de rechazo |
|---|---|---|
| Semántica | Valores y formas en una batería diseñada | Error fuera de tolerancia |
| Memoria | Bordes, alias y sincronización revisados | Acceso inválido o carrera |
| Recursos | Arquitectura y lanzamiento compatibles | Tile que excede recursos |
| Rendimiento | Mediana y dispersión frente a baseline justo | Mejora no reproducible |
| Integración | Paso completo y gradientes correctos | Kernel aislado que rompe el modelo |

Una instrucción tensorial no es una victoria si obliga a convertir repetidamente datos para una matriz pequeña. La comparación debe incluir la ruta de conversión cuando forme parte del uso real. También debe mantener la misma precisión de acumulación o informar explícitamente el cambio de contrato.

## Función de puntuación sin premiar errores

Una propuesta es calcular la media geométrica de aceleraciones por forma solo entre candidatos que superen todas las pruebas requeridas. Si una forma obligatoria falla, el proyecto no puede ocultarla eliminándola de la media. Se informa cobertura, tasa de fallos y tiempos por forma, además del agregado.

La media geométrica evita que una aceleración enorme en una sola forma domine linealmente el resumen, pero tampoco sustituye la tabla. Para una aplicación con frecuencias conocidas, una métrica ponderada por el tiempo real de la carga puede ser más relevante. El peso debe fijarse antes de ajustar, no después de ver qué favorece al candidato.

::: practica Umbral de competitividad elegido por el curso
Como criterio docente, se puede exigir estar dentro de un factor dos de una referencia fuerte en al menos el 80 por ciento de la suite, sin fallos de corrección, y explicar los peores casos. Ese umbral es una propuesta de evaluación, no una definición universal de SOTA. Un proyecto que lo alcance todavía debe declarar hardware, suite y precisión.
:::

## Calendario de cuatro semanas

En la primera semana se fija la suite y se reproduce la referencia. En la segunda se implementa una transformación con pruebas y se registra un baseline intermedio. En la tercera se estudia el recurso limitante y se realiza una ablación. En la cuarta se integra en el modelo y se redactan resultados, incluidos los negativos.

Cada semana termina con una decisión verificable. Si no se consigue reproducir la referencia, no se dedica la siguiente a ajustar contra una medición dudosa. Si el candidato es incorrecto, no se aumenta el presupuesto de búsqueda de velocidad. Si mejora el kernel pero no el modelo, se estudia Amdahl y overhead, no se oculta la diferencia.

## Qué resultado haría original el trabajo

La originalidad puede estar en un selector de kernels sensible a formas irregulares, una política de packing que se amortice con pesos reutilizados o un análisis de por qué una transformación pierde en una región de tamaños. No necesita afirmar una técnica inédita mundialmente. Sí necesita una pregunta propia, una implementación identificable y evidencia que vaya más allá de ejecutar un tutorial ajeno.

La entrega debe permitir activar y desactivar la contribución con una opción, de modo que se pueda realizar una ablación limpia. Un conjunto de cambios inseparables dificulta aprender cuál importa. Esa disciplina es especialmente valiosa en compiladores, donde una mejora aparente puede proceder de un detalle accidental de generación.

# Proyecto final D. Un agente propone kernels; el contrato decide {#project:agente}

## Motivación científica y límite de la automatización

Los trabajos recientes que exploran generación o bajada de kernels con modelos de lenguaje sugieren que un proponente puede encontrar transformaciones distintas de las de un pipeline tradicional. El artículo AI as a Compiler estudia precisamente una ruta de Triton a PTX acompañada de evaluación y verificación. KernelBench ofrece otra referencia útil para pensar tareas, comprobación y rendimiento de kernels generados. [@taic;@kernelbench]

El proyecto del estudiante consiste en separar el proponente del evaluador. El proponente puede ser un LLM, una búsqueda enumerativa o un conjunto de reglas. El evaluador no acepta un resultado porque venga explicado con confianza: verifica el contrato y mide en un entorno definido. El núcleo del diseño es esa separación, no el nombre del modelo utilizado.

## Elegir un espacio de búsqueda seguro y pequeño

Para empezar, no permitas programas arbitrarios que puedan leer archivos, acceder a la red o ejecutar comandos del host. Una opción es permitir únicamente parámetros de un generador de kernels conocido: tamaños de tile, orden de ejes y grado de desenrollado dentro de límites. Así el candidato es un objeto estructurado, no código con autoridad ilimitada.

```json
{
  "operation": "gemm_fp32",
  "tile_rows": 4,
  "tile_columns": 8,
  "reduction_block": 64,
  "strategy": "blocked",
  "numeric_contract": "fp32_reference_tolerance"
}
```

El validador comprueba tipos, rangos y combinaciones legales antes de generar fuente. Los campos desconocidos se rechazan. No basta con pedir al agente que sea cuidadoso: la herramienta debe impedir que una respuesta inesperada cambie el contrato, borre resultados o ejecute una acción fuera del espacio permitido.

## Un evaluador en fases

Primero valida el esquema. Después construye el código con un generador controlado. Compila con límite de tiempo. Ejecuta en un proceso separado y entorno restringido. Comprueba formas y números. Solo si pasa, mide varias repeticiones. Finalmente conserva candidato, fuente, diagnóstico y resultado.

El pseudocódigo siguiente describe arquitectura, no una función ya integrada en Lumbre:

```text
evaluar(candidato):
    comprobar_esquema_y_limites(candidato)
    fuente = generar_desde_parametros_conocidos(candidato)
    binario = compilar_con_presupuesto(fuente)
    resultado = probar_en_proceso_aislado(binario, casos_ocultos)
    si resultado no es correcto:
        registrar_fallo_y_terminar
    medidas = cronometrar_con_calentamiento(binario)
    registrar(candidato, fuente, resultado, medidas)
```

Un proceso separado no es por sí solo una frontera de seguridad completa. El proyecto debe usar el aislamiento y permisos que ofrezca su entorno. Para un primer laboratorio local, es preferible el espacio estructurado de parámetros a compilar código libre de origen no confiable dentro del mismo proceso del estudiante.

## No enseñar todas las pruebas al proponente

Si un candidato conoce únicamente tres entradas fijas, puede especializarse indebidamente a ellas. Separa casos de desarrollo y casos reservados. Cambia semillas y valores, pero mantén casos estructurales: tamaños cero donde proceda, uno, bordes, negativos, magnitudes variadas y formas irregulares.

Tampoco permitas que el candidato cambie tolerancias, elimine pruebas o modifique la referencia. La evaluación debe tener un hash o versión independiente. Un agente que corrige su propia nota cambiando el examen no ha mejorado el kernel. El informe debe distinguir propuestas, fallos de compilación, fallos de corrección y candidatos válidos.

## Un presupuesto que no desaparece del artículo

Fija número máximo de candidatos, tiempo de compilación y tiempo total de búsqueda. Conserva intentos fallidos. Compara con una búsqueda aleatoria o una cuadrícula del mismo presupuesto. Si el agente solo gana porque prueba cien veces más variantes, la conclusión sobre su estrategia es distinta.

El rendimiento final y el coste de encontrarlo son dos ejes. Una búsqueda cara puede ser rentable para una operación que se repetirá millones de veces. Para formas efímeras, puede perder frente a una heurística rápida. Calcula el punto de amortización: coste adicional de búsqueda dividido entre ahorro por ejecución, siempre que ambos estén medidos en unidades compatibles.

## Detectar soluciones tramposas o frágiles

Un kernel que devuelve constantes aprendidas de entradas públicas no implementa la operación. Uno que utiliza una biblioteca prohibida incumple el alcance de la tarea aunque sea correcto. Uno que omite sincronización puede pasar ocasionalmente y fallar en otra intercalación. El evaluador debe revisar tanto resultados como condiciones del experimento.

Las restricciones deben declararse antes de la búsqueda. No prohíbas después la técnica del ganador porque no era la que esperabas, ni aceptes una precisión inferior sin reflejarla. Si se permite llamar a una biblioteca, el resultado se etiqueta como selección o integración de biblioteca, no como generación completa de un kernel propio.

::: practica Experimento resuelto de control
Un agente encuentra un tile más rápido que el inicial. Para comprobar si su razonamiento aporta algo, se ejecuta una búsqueda aleatoria con el mismo número de propuestas legales y el mismo evaluador. Si ambas encuentran rendimientos similares, ¿ha fracasado el proyecto?

**Solución:** no. Se ha obtenido evidencia de que, en ese espacio y presupuesto, no se observa una ventaja clara del agente sobre el control. La comparación evita atribuir a inteligencia del proponente lo que podría explicar una exploración sencilla. El proyecto puede aportar un evaluador robusto y una caracterización del espacio de búsqueda.
:::

## Entrega y conexión con los LLM del curso

El estudiante entrega el generador controlado, evaluador, suite separada, historial completo y una comparación con control. No necesita entrenar un LLM para generar kernels si el proyecto estudia evaluación; puede usar una API o un modelo local autorizado, registrando versión y configuración. Esas llamadas no forman parte de las pruebas ejecutadas de esta edición.

La conexión con el decoder propio puede ser usarlo como carga objetivo: escoger una de sus operaciones y mejorarla sin cambiar la pérdida ni los gradientes. Así el proyecto une semántica, generación, validación y entrenamiento, en lugar de terminar en una demostración que solo produce código bonito.

# Proyecto final E. Del decoder pequeño al entrenamiento de un LLM mayor {#project:llm}

## Qué significa ampliar sin cambiar de problema

El núcleo del curso ya entrena un decoder pequeño mediante C generado. Este proyecto desarrolla la ruta hacia una escala mayor, sin fingir que aumentar dos números en la configuración resuelve memoria, rendimiento, datos y evaluación. El resultado esperado del estudiante es un hito de escala medido y una arquitectura de entrenamiento reproducible, no una promesa de competir con modelos de frontera sin recursos ni evidencia.

Las técnicas de paralelismo de modelos, partición de estados y planificación de cómputo se estudian en Megatron-LM, ZeRO y trabajos sobre asignación de presupuesto de entrenamiento. Son herramientas para razonar sobre la escala, no órdenes mágicas que conviertan el runtime educativo en un sistema distribuido completo. [@megatron;@zero;@chinchilla]

## Tres tamaños con objetivos distintos

El tamaño de **depuración** usa pocas capas y dimensiones para probar todos los gradientes y reproducir checkpoints. El tamaño de **ingeniería** debe ocupar una parte relevante del dispositivo disponible y revelar costes reales de memoria y lanzamiento. El tamaño de **calidad** necesita suficientes datos y presupuesto para evaluar generalización en una tarea lingüística definida.

No impongas que los tres tamaños coincidan. Un modelo suficientemente grande para saturar un acelerador puede ser demasiado caro para diferencias finitas de todos sus parámetros. Un modelo diminuto puede validar el optimizador sin decir nada sobre eficiencia de memoria a gran escala. Cada hito usa el tamaño adecuado a su pregunta.

## Un presupuesto de memoria antes de reservar

Cuenta pesos, gradientes, momentos, copias maestras cuando existan, activaciones, buffers temporales y estado del runtime. Si se usa AdamW con cuatro arreglos FP32 por parámetro entre pesos, gradientes y dos momentos, solo ese conjunto ocupa aproximadamente 16 bytes por parámetro, antes de activaciones. En una configuración mixta, las copias y tipos cambian el cálculo; no se reutiliza ciegamente el mismo coeficiente.

Para 100 millones de parámetros, 16 bytes por parámetro son 1.600 millones de bytes, aproximadamente 1,49 GiB. Esa cifra no demuestra que el modelo quepa en una GPU de dos GiB: faltan activaciones, logits, atención, workspace y reservas. El presupuesto debe incluir un margen operativo y contrastarse con el pico observado.

Construye una tabla por tensor y por fase del paso. La arena estática de Lumbre da una primera estimación para su propio grafo, pero no resuelve automáticamente buffers externos de bibliotecas o comunicación. Un modelo de memoria debe reconciliar estimación y medición, no declarar victoria cuando una de las dos es menor.

## Orden de las extensiones

Primero reduce materializaciones innecesarias y fusión de operaciones elemento a elemento. Después mejora GEMM y normalizaciones. Luego implementa atención eficiente también en backward, si esa matriz domina el pico. Añade checkpoint de activaciones cuando recomputar sea preferible a conservar. Solo entonces decide qué estado repartir entre dispositivos.

La precisión mixta se introduce con pruebas de estabilidad: tipos de almacenamiento, acumulación, escalado y detección de valores no finitos. FP8 o FP4 requieren contratos y escalas específicos; no son una sustitución textual de `float` por un tipo más pequeño. Los repositorios modernos de DeepSeek y NVIDIA muestran implementaciones especializadas, pero sus restricciones deben conservarse al integrarlas. [@deepgemm;@cutlass;@deepseek-v3;@deepseek-v4]

## Datos y evaluación que no se puedan confundir con la demo

Sustituye el corpus repetido por documentos con autorización de uso y una partición por unidades independientes. Guarda hashes o un manifiesto reproducible, reglas de limpieza y tokenizador. Define qué evaluación se ejecutará antes de comenzar la búsqueda de hiperparámetros. Mantén un conjunto final reservado para la conclusión.

Además de pérdida, registra número de tokens procesados, tokens por segundo, tiempo frío, tiempo por paso y pico de memoria. Para comparar dos compiladores, mantén arquitectura, inicialización, datos, orden de lotes y optimizador. Si cambias todo a la vez, una diferencia de calidad no puede atribuirse al compilador.

La muestra de texto se conserva como ilustración cualitativa. No sustituye métricas ni análisis de contaminación. Un modelo puede producir una frase convincente y tener un entrenamiento incorrecto; también puede tener un pipeline correcto y producir texto pobre por escala o datos insuficientes.

## Distribuido: un paso debe conservar su significado

Con paralelismo de datos, cada réplica calcula contribuciones de un lote local. Hay que definir si las pérdidas son medias locales o sumas y cómo se combinan gradientes. Un promedio doble puede reducir indebidamente la magnitud; una suma sin normalización puede cambiar la tasa efectiva al aumentar dispositivos.

Con partición tensorial, una operación se divide entre dispositivos y aparecen colectivos en puntos específicos. Su orden debe coincidir entre participantes y sus buffers vivir hasta finalización. Un fallo de colectivo puede parecer un bloqueo del modelo cuando realmente hay secuencias distintas entre procesos. Las bibliotecas NCCL y RCCL pertenecen a esta capa, no al algoritmo de autodiferenciación. [@nccl;@rccl]

El primer ensayo distribuido debe ser minúsculo y compararse con una ejecución de un solo dispositivo que use el mismo lote global. No se comienza con el modelo máximo esperando depurar a partir de un timeout. Se comprueban pérdidas, gradientes y una actualización antes de medir escalabilidad.

## Puertas de aceptación del proyecto LLM

La primera puerta exige que el modelo pequeño conserve pruebas de causalidad, gradientes y checkpoint. La segunda exige una ejecución mayor sin valores no finitos y con presupuesto de memoria explicado. La tercera compara rendimiento justo contra una referencia definida. La cuarta evalúa calidad con datos independientes. Ninguna puerta se sustituye por haber completado un número de epochs.

Una configuración que cabe pero tarda demasiado aporta información: identifica qué operación domina. Una que es rápida pero cambia la trayectoria por un error numérico no supera corrección. Una que reduce pérdida en entrenamiento y empeora evaluación obliga a analizar datos y sobreajuste, no a celebrar automáticamente que el compilador «entrena SOTA».

## Informe final y posible contribución

Un proyecto sólido puede aportar un backward de atención por tiles, un planificador de activaciones, una integración de GEMM con conversiones amortizadas o un runtime que reusa lanzamientos. Debe mostrar el efecto en un paso completo y conservar equivalencia semántica dentro del contrato elegido.

La meta de entrenar LLM actuales se cubre mediante arquitectura, autodiferenciación, kernels, datos, precisión, memoria, paralelismo y evaluación. El resultado local de este libro sigue siendo el modelo pequeño documentado. El estudiante que avance la escala debe añadir su propia evidencia a esa cadena, no heredar una afirmación de rendimiento que todavía no ha medido.
