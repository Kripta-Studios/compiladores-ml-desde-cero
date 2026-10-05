# Contrato de UOps: de la especificación a una IR verificable {#curso:contrato}

## Tres representaciones con responsabilidades distintas

El syllabus define `UOp(op, src:tuple[UOp, ...], arg)`. Este capítulo explica cómo razonar sobre ese contrato sin imponer clases, lenguaje ni biblioteca. Las propiedades `dtype`, `shape` y `addrspace` se infieren a partir de la operación, sus fuentes y su argumento. Una implementación puede memorizarlas, pero no debe permitir que una caché contradiga el cálculo de tipos.

Conviene distinguir el grafo puro con formas, el grafo que hace explícitos índices y efectos, y la secuencia final del renderer. El primero expresa el modelo y permite autodiff; el segundo conserva dependencias de memoria y rangos; la última decide el orden de emisión. Confundir esas responsabilidades hace que una mejora del renderer termine cambiando el significado del modelo.

Lumbre tiene una IR propia con forma y dtype almacenados, operaciones tensoriales y planificación específica. No implementa literalmente todo el catálogo del syllabus ni un lector universal `uop v1`. Aquí se especifica el trabajo que debe realizar el estudiante y se explica su relación con los mecanismos del libro. Los ejemplos de UOps son notación didáctica salvo cuando se reproduce expresamente el fragmento de intercambio original.

## Hash-consing con tipos, ámbitos y efectos

Dos nodos puros con la misma operación, fuentes canónicas y argumento semántico pueden compartir identidad. Esto permite que `(x+2)*(x+2)` contenga una sola suma. El argumento debe preservar toda distinción relevante: un literal booleano no es un entero, aunque el lenguaje anfitrión compare `True` e `1` como iguales. Una representación robusta de constantes incluye etiqueta de tipo y representación del valor.

Si el contrato conserva cero con signo, las constantes FP32 deben distinguir sus bits. Para NaN hay que decidir qué representaciones se admiten y si se canonicalizan. No uses igualdad ordinaria de coma flotante como definición universal de identidad estructural: NaN no es igual a sí mismo y ceros de signo distinto pueden compararse iguales.

`PARAM` necesita identidad de ámbito y posición. Los parámetros cero de dos funciones diferentes no se vuelven el mismo argumento por compartir nombre. El syllabus no detalla `ParamArg` ni `CallArg`; el estudiante debe fijar sus campos y comprobarlos. Una posibilidad es que `ParamArg` identifique ámbito, slot y tipo esperado, con tamaño y espacio de direcciones cuando proceda. Esa es una decisión de implementación, no un campo añadido al estándar original.

La tabla de interning no puede borrar eventos distintos de memoria. Dos `ALLOC` de igual tamaño pueden requerir almacenamiento diferente. Dos escrituras textualmente iguales pueden ocurrir en momentos diferentes. Para mantener hash-consing, la identidad o las dependencias del evento deben formar parte de su estructura; alternativamente, la implementación debe representar efectos en una capa que no los confunda con expresiones puras. En ambos casos, explica cómo se conserva esa distinción.

::: practica Una clave que parece suficiente
Una caché utiliza `(op, src, arg)` y `arg=1` tanto para un booleano verdadero como para el entero uno. ¿Qué prueba revela el fallo?

**Solución.** Construye ambos literales y consulta su `dtype`; deben seguir siendo distintos. Después construye `MUL` booleano e integer con entradas del tipo correspondiente. La regla de interning tiene que incluir la etiqueta semántica del literal, aunque la sintaxis pública del nodo siga teniendo tres campos.
:::

## Semana 1: operaciones puras y sus dominios

| Familia | Operaciones | Obligación del verificador |
|---|---|---|
| Entradas y aplicación | `PARAM`, `CALL`, `CONST` | Resolver ámbito y aridad; `CALL.src[0]` es el cuerpo y el resto son argumentos |
| Unarias | `CAST`, `NEG`, `RECIP`, `EXP2`, `LOG2` | Fijar tipo de resultado, conversiones y dominios admitidos |
| Binarias | `ADD`, `MUL`, `CMPLT`, `CMPNE`, `MAX` | Validar tipos y producir bool en comparaciones |
| División entera | `FLOORDIV`, `FLOORMOD` | Definir redondeo, divisor cero y desbordamiento |
| Selección | `WHERE` | Condición booleana y ramas con tipos compatibles |

El syllabus enumera operaciones y tipos, pero no publica todas las sobrecargas posibles. No deduzcas, por ejemplo, que `ADD` booleano significa OR: solo establece las equivalencias `MUL=AND`, `MAX=OR` y `CMPNE=XOR`. Las combinaciones adicionales deben rechazarse o especificarse expresamente. Tampoco aceptes conversiones implícitas diferentes en el evaluador y en el código C.

El plegado de constantes debe seguir la precisión del destino. Si `f32` significa redondeo binario32 tras cada operación, evaluar una cadena en doble precisión y convertir al final puede dar un resultado diferente. El tratamiento de FMA, infinitos, NaN, subnormales y ceros con signo se documenta antes de introducir reglas que dependan de él. Para un primer entregable, es válido limitar el dominio si el rechazo es verificable y está declarado.

`RECIP(x)` expresa el recíproco. `EXP2(x)` y `LOG2(x)` usan base dos. Reducir `LOG2(EXP2(x))` a `x` necesita hipótesis sobre rango, precisión y dominio; no es una identidad universal del programa FP32. `WHERE` selecciona valores, pero el syllabus no determina por sí solo si sus ramas se evalúan de forma perezosa. Esa distinción será decisiva para una carga enmascarada de la semana 4.

## División hacia menos infinito y C

Para enteros con divisor distinto de cero, una definición de división por suelo exige $a=bq+r$, con $q=\lfloor a/b\rfloor$. Así, para $a=-7$ y $b=3$, resultan $q=-3$ y $r=2$. Con $a=7$ y $b=-3$, resultan $q=-3$ y $r=-2$. El resto tiene el signo del divisor o es cero.

La división entera de C trunca hacia cero. Por eso imprimir `/` y `%` directamente no implementa esos ejemplos. Si se parte de cociente y resto truncados, y el resto es no nulo con signo distinto del divisor, la corrección es restar uno al cociente y sumar el divisor al resto. Antes debe tratarse el divisor cero y el caso del mínimo entero dividido entre menos uno, cuyo resultado no cabe en el mismo entero con signo.

::: comprueba Tabla mínima del evaluador
Para `(7,3)`, `(-7,3)`, `(7,-3)` y `(-7,-3)`, los pares cociente/resto por suelo son `(2,1)`, `(-3,2)`, `(-3,-2)` y `(2,-1)`. Comprueba también divisiones exactas. Las cuatro filas deben coincidir entre plegador y programa generado.
:::

## Reescribir de abajo arriba y saber cuándo detenerse

Visita primero las fuentes, reconstruye el nodo con sus versiones normalizadas y aplica reglas locales orientadas. Si una regla crea otro nodo reducible, vuelve a considerar el resultado con una política explícita. Conserva memoización por nodo y detecta ciclos o agotamiento de un presupuesto de reescritura. Un presupuesto sirve como diagnóstico; no demuestra que las reglas terminen.

Para una familia sencilla, puede justificarse terminación mediante una medida que disminuya: cantidad de operaciones constantes pendientes y, después, tamaño de una representación canónica. No mezcles `a+b -> b+a` con su inversa. Ordenar operandos con una clave estable evita ese ciclo cuando la conmutatividad esté permitida por el contrato numérico.

Una regla como `x*0 -> 0` requiere atención en flotantes: con `x` infinito, la expresión original puede producir NaN. Cambiar asociación de sumas también modifica redondeos. Mantén reglas exactas para el dominio elegido y reglas bajo relajación numérica identificadas por separado. Si el modo cambia, debe cambiar la clave del artefacto compilado.

El plegador puede evaluar un `CALL` puro con argumentos constantes ligando los `PARAM` del cuerpo al entorno correspondiente. Evita sustituir parámetros de una llamada anidada por coincidencia de nombre. La recursión general, si se admite, necesita su propia semántica; no se obtiene gratuitamente de un evaluador recursivo de DAG.

## Semana 2: tipo de valor, forma y dirección

`dtype` responde qué clase de dato representa un nodo; `shape` describe sus ejes lógicos; `addrspace` distingue memoria de valores de cálculo. Una dirección a un escalar `f32` y un valor `f32` no son intercambiables aunque compartan forma `()`. `void` representa ausencia de valor utilizable, como una escritura.

| Operación | Contrato del syllabus | Consecuencia práctica |
|---|---|---|
| `BUFFER` | Buffer global persistente, nunca definido dentro de `CALL` | Su vida supera una invocación |
| `ALLOC` | Almacenamiento local de llamada | No puede escaparse una dirección que ya no sea válida |
| `STACK` | Nuevo eje inicial; fuentes de igual forma | Con m fuentes de forma S, la salida tiene forma `(m,) + S` |
| `INDEX` | Consume ejes por la izquierda mediante fuentes índice | Con k índices válidos, quedan los ejes desde k en adelante |
| `LOAD` | Una fuente | Obtiene un valor desde memoria válida |
| `STORE` | Dos fuentes | Escribe un valor en el destino compatible |
| `AFTER` | `AFTER(buf, store)` | Hace depender el uso posterior del buffer de la escritura |
| `LINEAR` | Fuentes en orden antes del renderer | La emisión debe respetar la secuencia elegida |

La tabla no impone una sintaxis textual para cada constructor. En particular, el orden destino/valor de `STORE`, los campos de tipos y el tratamiento exacto de `STACK` en cada espacio deben declararse al completar la especificación del proyecto. En los ejemplos siguientes usaremos destino primero y valor después.

Si `x` tiene forma `(2,3)`, indexarlo con `i` produce forma `(3,)`; indexarlo con `i,j` produce `()`. Hay que verificar índices enteros y, cuando se conocen estáticamente, sus límites. Una forma escalar tiene un elemento. Una forma `(0,)` tiene cero. Confundirlas provoca reservas y accesos incorrectos.

## Una función C sin bucles y una dependencia real

El primer programa de memoria puede tener esta salida C, bajo el contrato de que `p` apunta a un elemento válido y el llamador admite modificarlo:

```c
float incrementar(float *p) {
    float anterior = p[0];
    p[0] = anterior + 1.0f;
    return p[0];
}
```

En una notación conceptual, indexa el buffer, carga el valor anterior, construye la suma, crea `STORE(destino, suma)` y alimenta la lectura final desde una dirección obtenida del buffer dependiente `AFTER(buffer, escritura)`. Así la lectura posterior tiene una dependencia explícita. Volver a cargar desde la dirección original sin ese vínculo permite que una optimización la trate indebidamente como la primera lectura.

Con entrada 4, la primera llamada devuelve 5 y deja 5 en memoria; la segunda devuelve 6. Esta prueba detecta un runtime que no conserva buffers o un compilador que congela entradas como constantes. También debe existir una prueba con dos direcciones alias si la ABI permite aliasing. Si lo prohíbe, ese contrato debe verificarse o mantenerse como precondición pública, no esconderse en el renderer.

Un orden topológico cualquiera conserva dependencias del grafo, pero no inventa las que falten. `LINEAR` es una lista final de emisión, no una reparación de aliasing. El verificador comprueba que toda fuente necesaria aparece antes, que los efectos ordenados conservan su relación y que ninguna operación fuera de ámbito usa un local de una llamada terminada.

## Semana 3: rangos, cierre y acumulación

`RANGE(end, nombre)` representa el intervalo semiabierto desde cero hasta `end`. `END(passthrough, range)` vincula un resultado o efecto con el cierre de ese rango. El nombre por sí solo no debe mezclar dos ámbitos. Verifica correspondencia entre aperturas y cierres, anidamiento y disponibilidad de los valores usados después.

La reducción suma exige inicialización una sola vez por salida. Este pseudocódigo fija el significado de una GEMM rectangular; no prescribe una representación interna de acumuladores:

```text
para i en [0, M):
    para j en [0, N):
        acc = 0
        para k en [0, K):
            acc = acc + A[i,k] * B[k,j]
        C[i,j] = acc
```

Inicializar `acc` dentro del rango k dejaría solo el último producto; inicializarlo antes del rango j mezclaría columnas. La prueba con K igual a cero, si se admite, obliga a producir ceros para la suma sin leer entradas. La prueba con M o N igual a cero no debe escribir salida alguna.

Para una convolución 1D válida sin invertir el filtro, usa `x=[1,2,3,4]` y `w=[2,-1]`. La suma de productos sobre ventanas produce `[0,1,2]`. Esta convención es correlación cruzada, habitual en redes neuronales; si se quiere convolución matemática, hay que invertir el filtro. El nombre de la operación no sustituye esa decisión.

## El fragmento de intercambio proporcionado

El syllabus aporta este ejemplo literal:

```text
# uop v1 256x256 GEMM
%0 = buffer 65536 : dtype=f32 slot=0
%1 = buffer 65536 : dtype=f32 slot=1
%2 = reshape %0, (256, 256, 1)
%3 = reshape %1, (256, 1, 256)
%4 = mul %2, %3
%5 = reduce %4 : op=add pop=1
```

Son seis nodos. Las dos vistas conservan 65.536 elementos cada una. Bajo broadcasting, el producto lógico tiene forma `(256,256,256)`: contiene 16.777.216 valores si se materializa, 64 MiB en FP32. Una bajada que acumule directamente puede evitar esa materialización.

Para interpretar la cuenta necesitamos declarar qué ejes elimina `pop=1` y cómo se interpretan los buffers. **En este desarrollo didáctico adoptamos reducción de los ejes iniciales.** Si `%0` contiene `A[k,i]` y `%1` contiene `B[k,j]`, el producto es `A[k,i]*B[k,j]` y la salida es la suma sobre k. Con ambos buffers vistos como matrices fila a fila, esto representa $A^T B$. Para representar $A B$ con A almacenada como `[i,k]`, se requiere la permutación correspondiente. El nombre GEMM sigue siendo adecuado, pero las dimensiones cuadradas no revelan esta diferencia.

El fragmento también presupone broadcasting en `mul`. Una IR que exija formas idénticas debe insertar `EXPAND` durante la normalización. El ejemplo no especifica una gramática completa, escapes, codificación de literales, salidas, ABI ni política de errores. El alumno debe acordar esos detalles antes de prometer interoperabilidad.

## Un intercambio que se pueda comprobar

Como extensión docente, el lector admite solo versiones declaradas, identificadores definidos una vez y referencias a nodos ya definidos. El impresor recorre un orden topológico estable y conserva tipos y argumentos. Las salidas se declaran con una convención documentada; no se asume para todos los programas que el último nodo sea la única salida.

La prueba de ida y vuelta compara estructura semántica: leer, imprimir, volver a leer y comprobar operación, fuentes, argumentos, tipos y salidas. Los números `%0`, `%1` pueden cambiar por una renumeración válida. No deben cambiar slots externos, constantes, relaciones de dependencia ni disposición lógica de datos.

Añade casos inválidos: versión desconocida, referencia inexistente, identificador duplicado, operación no admitida, tipo incompatible, reshape que cambia elementos y slot incompatible con la firma. El diagnóstico debe señalar nodo y propiedad. Estos casos prueban el verificador; no son excusas para aceptar silenciosamente una interpretación distinta en cada backend.

# De rangeify al entrenamiento: preservar el contrato {#curso:rangeify}

## Movimientos como mapas de coordenadas

`RESHAPE` cambia la descomposición de un índice lineal conservando el número de elementos. `EXPAND` replica lógicamente un eje de tamaño uno; su coordenada de entrada siempre es cero. `PERMUTE` reordena ejes mediante una permutación válida. `FLIP` convierte la coordenada i de un eje de longitud n en `n-1-i`. `SHRINK` desplaza la coordenada por el inicio de una región. `PAD` amplía el dominio y distingue posiciones interiores de posiciones de relleno.

Estas operaciones describen valores, no necesariamente nuevas reservas. Por ejemplo, permutar una matriz `(2,3)` a `(3,2)` transforma la coordenada de salida `(j,i)` en la entrada `(i,j)`. Al consumirla en un cálculo, el compilador puede componer ese mapa con el acceso original. No es obligatorio copiar la matriz para obtener su traspuesta lógica.

Para `PAD`, el predicado de validez forma parte del mapa. Con x de longitud tres, un cero a la izquierda y dos a la derecha, la salida tiene longitud seis. Su coordenada de entrada es `i-1`, válida solo si `1 <= i < 4`. En otro caso el resultado es cero. No ejecutes primero una carga fuera de límites y después un `WHERE`: si las ramas se evalúan ávidamente, el acceso inválido ya ocurrió. La bajada debe producir carga predicada o control de flujo que garantice seguridad.

## Reducción por número de ejes

El syllabus define `REDUCE` mediante cantidad de ejes y operación, sin completar aquí todos sus detalles. Adoptamos para los ejemplos una reducción de los p ejes iniciales. Una suma sobre `(K,M,N)` con `p=1` da `(M,N)`. Para reducir otro conjunto, primero permuta los ejes deseados a la izquierda. Mantener o eliminar dimensiones unitarias es una decisión que debe reflejarse en el tipo de salida.

Una reducción suma tiene identidad cero. Para máximo, el dominio y el tipo determinan si existe una identidad admisible o si se rechaza un eje vacío. No inicialices indiscriminadamente el máximo a cero: sobre `[-4,-2]` daría cero en lugar de -2. Tampoco cambies orden de acumulación FP32 sin considerar la política numérica.

## Derivar GEMM con un ejemplo rectangular

Sean $A=\left(\begin{smallmatrix}1&2&3\\4&5&6\end{smallmatrix}\right)$ y $B=\left(\begin{smallmatrix}7&8\\9&10\\11&12\end{smallmatrix}\right)$. Para calcular AB con reducción inicial, transforma A de `(M,K)` a `(K,M)`, añade un eje final y transforma B de `(K,N)` a `(K,1,N)`. Expande ambos a `(K,M,N)`, multiplica y reduce k.

```text
A: (2,3) -> PERMUTE(1,0) -> (3,2) -> RESHAPE(3,2,1)
B: (3,2) -> RESHAPE(3,1,2)
EXPAND ambos a (3,2,2)
MUL -> REDUCE(add, pop=1) -> (2,2)
resultado: [[58,64], [139,154]]
```

La coordenada lógica `(k,i,j)` se traduce a `A[i*K+k]` y `B[k*N+j]` en almacenamiento contiguo. Rangeify introduce los rangos i, j y k y un acumulador por salida. El producto lógico de doce elementos no obliga a reservar doce posiciones: los productos pueden consumirse inmediatamente.

Esta prueba rectangular detecta intercambios de ejes que una multiplicación de matrices cuadradas podría ocultar. Amplíala con valores distintos por fila y columna, K igual a uno y dimensiones no múltiplos del tile. Los ejemplos numéricos son deducciones exactas con estos enteros pequeños, no mediciones de rendimiento.

## Derivar convolución sin un operador privilegiado

Una convolución 1D de entrada de longitud L y filtro de longitud F, sin padding y con paso uno, puede expresarse como suma de F vistas desplazadas multiplicadas por cada peso. Para el desplazamiento r, `SHRINK` selecciona `x[r:r+L-F+1]`. `STACK` agrupa esas vistas en forma `(F,L-F+1)`. El filtro se reinterpreta como `(F,1)`, se expande, se multiplica y se reduce el eje inicial.

Así se obtiene el ejemplo `[0,1,2]` del capítulo anterior sin introducir una primitiva opaca de convolución. El uso explícito de `STACK` puede hacer crecer el grafo con F; una extensión con índices simbólicos evita desplegar filtros grandes. El ejercicio inicial busca demostrar semántica y composición, no escoger ya el mejor kernel industrial.

Con padding, dilatación o paso distinto de uno, escribe primero la coordenada `salida*paso + r*dilatacion - padding`. Después determina su máscara. El gradiente del filtro suma contribuciones de todas las ventanas y el de la entrada acumula las ventanas que la tocan; una copia de los índices del forward no basta para construir ambos.

## Qué elimina rangeify y qué debe conservar

El resultado del pase expresa el cálculo con valores escalares, índices y rangos. Las operaciones aritméticas ya no portan los ejes tensoriales originales como una ejecución implícita. Las direcciones conservan información suficiente para generar accesos y el runtime conserva tamaños de buffers y restricciones. «Eliminar shapes de todas las ops» no significa olvidar cuántos bytes necesita el programa.

Valida tres aspectos por separado: toda coordenada admitida encuentra el valor correcto, ninguna coordenada inválida produce una carga o escritura, y cada salida se calcula la cantidad de veces exigida. Un programa puede respetar límites y aun así escribir dos veces una posición dejando otra sin calcular.

Conserva un dump antes y después del pase para un caso pequeño. La prueba diferencial ejecuta el grafo puro con un evaluador sencillo y el programa bajado con los mismos datos. El evaluador de referencia no debería llamar al mismo generador de índices que pretende comprobar, porque reproduciría sus errores.

## STAGE: pagar memoria para evitar trabajo repetido

Si un intermedio caro tiene dos consumidores, fusionarlo en ambos puede duplicar su cálculo. `STAGE` permite materializarlo y hacer que los consumidores lo indexen. Decide su tamaño, productor, consumidores, ubicación y vida. La reserva solo puede reutilizarse cuando todos los consumidores han terminado.

Supón que un intermedio cuesta C, y que escribirlo y leerlo para ambos consumidores cuesta T. Recomponerlo dos veces cuesta aproximadamente 2C; materializarlo cuesta C+T. Si C supera T, materializar podría resultar favorable. Son costes orientativos: lanzamientos, caché, vectorización y capacidad de memoria pueden cambiar la decisión, por lo que el modelo debe contrastarse con medidas.

Una barrera de materialización no es equivalente a una regla de autodiff que detenga el gradiente. Mantén el grafo matemático puro para diferenciar; decide stages y efectos al bajar forward y backward. Si se inserta una anotación de stage antes, debe aclararse que no cambia la función y cómo la trata el diferenciador.

::: practica Dos consumidores de una reducción
Sea `s=sum(x*x)` y dos salidas `s+1` y `s+2`. ¿Es mejor generar un kernel por salida?

**Solución.** Si cada kernel vuelve a recorrer x, duplica la reducción. Un plan calcula s una vez y lo comparte; otro fusiona las dos salidas en un productor común. Elegir depende de la representación y del destino. La prueba de rendimiento debe contar recorridos, bytes y lanzamientos, además del número de kernels.
:::

## Autodiff antes de introducir efectos

Recorre el grafo puro en orden topológico inverso y acumula contribuciones para cada entrada diferenciable. En `y=a*a`, a recibe dos contribuciones. Sustituir la suma de gradientes por una asignación conserva solo una y produce un error por factor dos. Los tipos enteros y booleanos se usan para índices y condiciones; la entrega debe declarar qué nodos admiten derivación.

Para `z=x+b` con x de forma `(B,D)` y b de forma `(D,)`, el gradiente de b suma sobre el eje de lote. En un `EXPAND`, el backward reduce ejes replicados. En una permutación, aplica la inversa. En un reshape, restaura la forma. En un flip, invierte el mismo eje. En padding y shrink, el backward recorta o rellena según corresponda. Acumula las contribuciones de vistas solapadas.

En un máximo con empate o una función no diferenciable, documenta la convención. No compruebes diferencias finitas exactamente en el punto de quiebre y después atribuyas el desacuerdo al compilador. Para funciones suaves y entradas pequeñas, compara gradientes analíticos, autodiff y diferencias centrales con una perturbación y tolerancia justificadas.

`STORE`, `CALL`, `AFTER`, `RANGE` y `END` quedan fuera del autodiff descrito por el syllabus. Aunque un `CALL` matemáticamente puro pudiera diferenciarse mediante otra extensión, esa no es la ruta exigida aquí. Expande o expresa el modelo en el grafo puro admitido antes de diferenciar, y vuelve a aplicar planificación y rangeify al grafo resultante.

## El primer entrenamiento: MNIST como entrega

Una propuesta mínima de laboratorio es un clasificador lineal con imágenes aplanadas de 784 componentes, pesos de forma `(784,10)` y sesgo `(10,)`. Con lote B, los logits son `XW+b`, de forma `(B,10)`. Este modelo deja visibles GEMM, broadcasting, reducción y gradientes. Una capa oculta puede añadirse después como extensión.

Usa entropía cruzada estable. Para una fila de logits z y etiqueta y, resta su máximo m antes de exponenciar y calcula `m + log(sum(exp(z-m))) - z[y]`. Con las primitivas del syllabus, `exp(x)` puede expresarse como `EXP2(x*log2(e))` y `log(x)` como `LOG2(x)*ln(2)`, bajo la precisión acordada. La resta se expresa con `ADD` y `NEG`.

La selección de etiqueta puede formularse con una máscara one-hot externa y una reducción si todavía no existe gather diferenciable. Los datos y etiquetas son entradas; no se necesita derivar respecto a ellos para optimizar los pesos. Normaliza la pérdida una sola vez por lote y aplica una actualización simple antes de introducir un optimizador más complejo.

Primero comprueba un lote artificial, luego sobreajusta un conjunto diminuto de ejemplos y finalmente utiliza una partición de entrenamiento y evaluación. Registra procedencia de datos, normalización, semilla, tamaño de lote, pasos, pérdida y exactitud. No uses evaluación para ajustar indefinidamente el modelo. El libro no fija una exactitud conseguida: el estudiante aporta su resultado y la configuración que lo produce.

## Semana 6: LOOP, UPCAST, THREAD y GROUP

Un rango `LOOP` se ejecuta mediante iteración. `UPCAST` expone varias posiciones como operaciones o acumuladores distintos, permitiendo desenrollado y posible vectorización. `THREAD` reparte trabajo entre trabajadores. Esta clasificación describe cómo ejecutar un eje; no autoriza a repartir una reducción con escrituras concurrentes sin combinar sus parciales.

Para sumar un vector, una propuesta segura reparte bloques disjuntos, calcula una suma privada por trabajador y combina las sumas en un orden definido. Usar `total += x[i]` desde todos los hilos sin sincronización introduce una carrera. Si cambia el orden FP32, la prueba compara según el contrato numérico, no exige identidad de bits accidental entre planes diferentes.

`GROUP` reúne múltiples resultados `void` sin establecer orden entre sus fuentes; su forma es `()`. Si dos salidas escriben posiciones independientes, pueden agruparse. Si una lee lo que otra escribe, hace falta una dependencia explícita. Ni el orden textual de argumentos de `GROUP` ni un renderer determinista sustituyen ese vínculo.

## Contar operaciones y localizar el límite

Con la convención habitual de contar multiplicación y suma como dos operaciones, una GEMM M por K por N realiza aproximadamente $2MKN$ FLOPs. Para 256 en los tres ejes son 33.554.432 FLOPs. FLOPs es cantidad de trabajo; FLOP/s es tasa. Divide por segundos de la fase medida, no por tiempo de compilación salvo que estudies expresamente el coste completo.

Un tile de salida de 16 por 16 que recorre K necesita reutilizar cada valor cargado tantas veces como permita su organización. Aumentar tile puede mejorar reutilización y a la vez consumir más registros o caché. Mide formas pequeñas, grandes e irregulares; un único cuadrado no describe la carga del modelo.

La comparación CPU debe fijar hilos, precisión, biblioteca de referencia y fronteras de medición. Separa ejecución residente, preparación y extremo a extremo. Para declarar competitividad con el estado del arte, la referencia debe ser pertinente y los casos desfavorables deben permanecer en el informe. Los resultados CPU existentes de Lumbre no se convierten en esa afirmación por añadir una tabla de metas.

## Atención online y el objetivo FlashAttention

El artículo FlashAttention estudia atención exacta organizada por tiles para reducir tráfico entre niveles de memoria. Ese principio conecta directamente con la semana de jerarquías de memoria. Implementar la recurrencia en CPU permite comprobar la matemática antes de abordar un kernel GPU especializado. [@flash1]

Para una fila de scores, mantén máximo m, suma normalizadora l y numerador vectorial o. Al recibir otro bloque, actualiza el máximo, reescala el estado anterior y añade las contribuciones del bloque. El resultado final es o dividido por l. Atiende explícitamente el primer bloque no vacío y las filas completamente enmascaradas para evitar operaciones indeterminadas con infinitos.

La forma densa es un buen oráculo para longitudes pequeñas. Compara salidas y gradientes con máscara causal y longitudes no múltiplos del bloque. Un forward con menos memoria no garantiza un backward con la misma propiedad: la validación actual de Lumbre recompone operaciones densas en backward. Para alcanzar el hito completo hay que diseñar y medir también esa fase.

## GPU: la dependencia lógica se convierte en ejecución

En las semanas 7–8, un eje de salidas independientes puede mapearse a bloques e hilos. Un eje de reducción necesita coordinación y una estrategia de parciales. Las dependencias `AFTER` y los tiempos de vida deben conservarse cuando un lanzamiento es asíncrono: regresar al host no implica que el dispositivo haya terminado de leer un temporal.

Una barrera de bloque solo coordina ese bloque. Las dependencias entre kernels necesitan el orden que establezca el runtime en sus colas o eventos. Las operaciones matriciales especializadas añaden restricciones de tipos, layouts y participación de hilos. El tutorial CUDA desarrolla ejercicios para convertir estos requisitos en criterios de selección y pruebas.

## Formas simbólicas y el paso de entrenamiento

Una dimensión simbólica representa una familia de tamaños con restricciones. Si el kernel requiere K múltiplo de ocho, el compilador puede demostrarlo, comprobarlo al lanzar o escoger otro kernel. No debe reutilizar una variante suponiendo que el tamaño anterior sigue vigente. Incluye en la caché las decisiones que afectan al código: destino, tipos, especializaciones y modo numérico.

Antes de entrenar un LLM, fija el contrato de un paso: entradas y etiquetas desplazadas, máscara causal, pérdida normalizada, gradientes, actualización y contador. Un checkpoint de reanudación conserva pesos, estados del optimizador, contador y estados aleatorios o del flujo de datos necesarios. Guardar solo pesos permite inferencia o reinicio, pero no acredita continuar exactamente la misma trayectoria.

La aceptación de las semanas 9–10 incluye una prueba de causalidad, comprobación de gradientes en un caso pequeño, reanudación comparada y medición del paso completo. El objetivo de escala se elige según memoria y tiempo disponibles. Se conserva el modelo pequeño como regresión incluso cuando el proyecto final trabaje con un modelo mayor.
