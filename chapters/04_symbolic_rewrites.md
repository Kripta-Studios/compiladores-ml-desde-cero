# Índices simbólicos: calcular con lo que todavía no sabemos {#ch:symbolic}

## Una casilla cuyo número depende de otra

Supón que una fila tiene ocho elementos y quieres acceder al elemento situado dos posiciones después de i. Si i vale tres, la dirección lógica es cinco. Pero el compilador puede estar construyendo un programa que servirá para todos los valores de i entre cero y cinco. Necesita representar `i+2` sin conocer todavía i.

Una expresión simbólica es una receta para calcular un valor cuando se conozcan sus variables. No es una predicción estadística ni una cadena de texto opaca. Podemos analizar su estructura, simplificar partes y deducir límites.

Los índices de Lumbre se estudian con un pequeño lenguaje entero separado del de valores FP32. Esa separación evita aplicar por accidente reglas enteras a números de coma flotante. En el paquete, `symbolic.py` funciona como laboratorio autocontenido de expresiones e intervalos; el renderer tensorial principal utiliza sus propias fórmulas de índices estáticos. No se presenta el laboratorio como un solver general ya integrado en todos los pases.

## Variables, constantes y operaciones

Una constante como 8 tiene valor conocido. Una variable como i tendrá valor en ejecución. Una suma une dos expresiones. La estructura de `i*8+j` contiene una multiplicación y una suma, igual que el grafo tensorial, pero ahora cada nodo describe un índice entero.

```python
from lumbre.symbolic import var, const, simplify

i = var("i")
j = var("j")
expresion = i * 8 + j
print(expresion.evaluate({"i": 2, "j": 3}))
```

La intención del ejemplo es obtener 19: dos filas completas de ocho posiciones más tres posiciones dentro de la siguiente. El método evalúa recursivamente la receta con los valores del diccionario.

## Por qué los intervalos son útiles

Sabemos que i está entre cero y tres, y j entre cero y siete. Entonces `i*8+j` está entre cero y treinta y uno. Hemos demostrado que el índice cabe dentro de un array de treinta y dos elementos sin probar todas las combinaciones una por una.

El intervalo es una aproximación del conjunto de valores posibles. Para ciertas expresiones la cota será exacta; para otras perderá información. Esa pérdida puede impedir una optimización segura, pero no debe permitir una optimización insegura.

::: traduccion Propagar límites
Si $a\in[a_{\min},a_{\max}]$ y $b\in[b_{\min},b_{\max}]$, entonces la suma pertenece a $[a_{\min}+b_{\min},a_{\max}+b_{\max}]$. Para el producto general se comparan los cuatro productos de extremos. Si sabemos que una constante multiplicadora es positiva, el orden de extremos se conserva.
:::

La notación `a ∈ [mínimo,máximo]` dice que el valor de a se encuentra dentro de ese intervalo. No significa que a sea un array ni que todos los valores del intervalo deban alcanzarse realmente.

## División y resto: encontrar fila y columna

Si una tabla tiene ocho columnas y un índice lineal q vale diecinueve, la fila es `q//8 = 2` y la columna `q%8 = 3`. Hemos separado cuántas filas completas caben y cuánto sobra. Esta pareja de operaciones permite convertir un único bucle lineal en coordenadas multidimensionales.

Para índices no negativos y divisor positivo, la identidad es `q = (q//8)*8 + q%8`, con resto entre cero y siete. La restricción de signos importa porque los lenguajes pueden tener convenciones diferentes para división y resto de enteros negativos.

```python
for q in range(32):
    fila, columna = q // 8, q % 8
    assert q == fila * 8 + columna
    assert 0 <= fila < 4
    assert 0 <= columna < 8
```

Este test cubre un dominio finito pequeño. El argumento general procede de la definición de cociente y resto bajo el contrato indicado. Ambos son útiles: el argumento explica la regla y el test detecta errores en nuestra implementación.

## Simplificaciones que necesitan hipótesis

`(i*8+j)//8 -> i` es válida si j está entre cero y siete y trabajamos con enteros sin desbordamiento. Sin esa hipótesis, j podría valer nueve y añadir una fila. Una reescritura debe registrar las condiciones bajo las que se aplica.

Análogamente, `(i*8+j)%8 -> j` necesita ese mismo intervalo de j. Si solo sabemos que j es no negativo, podríamos simplificar a `j%8`, pero no a j. Aprender a conservar una condición es más importante que memorizar una colección de patrones.

::: ejemplo Un contraejemplo mínimo
Para i=2 y j=9, `(i*8+j)//8` vale 3, mientras i vale 2. La regla sin guardia cambia la dirección. Un test con j=0,1,2 no detectaría el fallo porque permanece dentro del intervalo donde sí es válida.
:::

## Máscaras como proposiciones

Un kernel vectorizado puede generar índices que exceden el tamaño real. Para N=10 y grupos de cuatro, el último grupo considera posiciones 8,9,10,11. Solo las dos primeras son válidas. Una máscara representa la condición `q<N`.

El compilador puede simplificar una máscara a verdadera si demuestra que todos los índices del grupo están dentro de rango. Puede simplificarla a falsa si demuestra que ninguno lo está. Si no puede demostrar ninguna de esas dos cosas, debe conservarla.

No confundir «no he encontrado un índice inválido» con «he demostrado que no existe». El análisis conservador prefiere mantener una comprobación cuando falta información. Perder una optimización es un coste de rendimiento; eliminar una comprobación necesaria es un error de corrección y potencialmente de seguridad.

## Aritmética matemática y anchura de máquina

El lenguaje simbólico puede razonar con enteros Python, que crecen según sea necesario. El código generado utiliza enteros de anchura fija. Una prueba de límites solo se traslada si las expresiones intermedias no desbordan esa anchura, o si la semántica modela explícitamente el desbordamiento.

Aunque el índice final quepa, un producto intermedio puede no caber. Por ejemplo, multiplicar dimensiones antes de comprobarlas exige cuidado. Un frontend robusto verifica tamaños y límites de dirección antes de reservar o generar memoria. No puede confiar en que una reserva fallida repare por sí sola una expresión desbordada.

## Rangos relacionados: lo que un intervalo pierde

Si j es exactamente `7-i` y i está entre cero y siete, la suma `i+j` siempre vale siete. Un análisis que solo recuerde los intervalos independientes `[0,7]` y `[0,7]` concluirá que la suma está entre cero y catorce. La cota es segura, pero imprecisa.

Para conservar relaciones más ricas se utilizan dominios abstractos y análisis afines. No los necesitamos para el primer compilador, pero debemos reconocer el límite de nuestro análisis. Un motor de intervalos no se convierte automáticamente en un demostrador de todas las identidades aritméticas.

## Expresiones afines y transformaciones de bucles

Una expresión afín combina variables multiplicadas por constantes y una constante adicional, como `3*i+2*j+5`. Los accesos afines permiten analizar regiones y dependencias de manera estructurada. Divisiones y módulos por constantes introducen extensiones útiles para tiling y layouts.

Los compiladores de ML aprovechan estas estructuras para generar bucles, transformar índices y vincular ejes lógicos con ejes físicos. MLIR ofrece representaciones y dialectos para expresar distintos niveles de abstracción; su tutorial Toy es una ruta de comparación con nuestro compilador pequeño. [@mlir-toy]

## Cómo probar un simplificador simbólico

Primero genera expresiones pequeñas sobre variables con dominios acotados. Después evalúa la original y la simplificada en todas las combinaciones de esos dominios. Añade pruebas específicas de fronteras: cero, último índice válido, primer índice inválido y divisores que cambian un cociente.

No permitas divisores cero en los generadores cuando el lenguaje los prohíbe. Pero sí prueba que el frontend rechaza esas expresiones. Un test no debe descartar silenciosamente todas las entradas que incomodan al algoritmo; debe representar el contrato real.

::: practica Demostrar o conservar
Sabes que `0 <= t < 16`. Simplifica `(32*b+t)//16` y `(32*b+t)%16`, suponiendo b no negativo. Después explica qué cambia si t puede valer 16.
:::

**Solución.** La primera expresión vale `2*b` y la segunda t, porque `32*b` es múltiplo de 16 y t es un resto válido. Si t puede valer 16, la primera puede ser `2*b+1` y la segunda cero. La frontera superior estricta es una parte esencial de la regla.

::: comprueba Antes de continuar
Debes poder transformar un índice lineal en coordenadas, propagar una cota de suma y encontrar una hipótesis que falte en una reescritura. El objetivo es justificar direcciones, no practicar álgebra simbólica sin dominio.
:::

# Reescrituras: mejorar el programa conservando su contrato {#ch:rewrites}

## Cambiar el plan, no el resultado pedido

Un compilador transforma representaciones. Algunas transformaciones hacen el grafo más pequeño; otras lo hacen más grande para exponer paralelismo o generar instrucciones legales. La palabra **optimización** no significa siempre reducir el número de nodos.

Para una suma entera, `x+0` puede reemplazarse por x bajo la semántica correspondiente. Para una transposición seguida de su inversa, podemos recuperar el tensor original si no hay efectos que debamos conservar. Para una multiplicación matricial, expandirla en bucles produce muchos elementos de IR pero acerca el programa al hardware.

Una transformación necesita dos descripciones: qué patrón reconoce y qué condición garantiza que su reemplazo sea válido. La condición puede depender de tipos, formas, intervalos, aliasing, valores especiales y política de coma flotante.

## Pattern matching explicado con piezas

Imagina una caja de piezas donde buscas «una suma cuyo segundo operando es cero». No te interesa el nombre concreto del primer operando. Un patrón permite poner una variable, por ejemplo x, en esa posición y capturar la pieza encontrada.

```text
Patrón:      ADD(x, CONST(0))
Condición:   dominio entero exacto compatible
Reemplazo:   x
```

La variable del patrón no es necesariamente una variable de entrada del programa. Puede capturar un subgrafo completo. Esa distinción permite escribir reglas que no dependan de la complejidad interna del operando.

Tinygrad utiliza `UPat`, `PatternMatcher` y recorridos de reescritura dentro de un pipeline de UOps. Lumbre enseña la idea mediante componentes más pequeños; no copia una lista universal de reglas suponiendo que todas son seguras para su propia semántica. [@tinygrad-codegen]

## Reescribir de abajo hacia arriba

Si una expresión es `(3+4)*x`, simplificar primero la suma produce `7*x`. Un recorrido de abajo hacia arriba transforma las entradas antes de examinar el nodo que las utiliza. Esto permite que una regla descubra constantes que antes estaban escondidas.

El recorrido debe conservar el compartido del grafo. Si un nodo común se transforma, todos sus usos compatibles deberían referirse a la nueva versión, no a copias incoherentes. Una tabla que asocie nodo antiguo y nodo nuevo ayuda a garantizar esa consistencia.

```python
def reescribir(orden, regla):
    nuevos = {}
    for viejo in orden:
        entradas = tuple(nuevos[x] for x in viejo.src)
        candidato = reconstruir(viejo, entradas)
        nuevos[viejo] = regla(candidato) or candidato
    return nuevos
```

Este es un esquema de algoritmo. `reconstruir` debe utilizar el constructor canónico y conservar los metadatos que no cambian. Devolver `None` indica «la regla no se aplica», no «el resultado es un tensor vacío».

## Punto fijo y riesgo de bucles

Una pasada puede hacer aplicable otra regla. Por eso algunos motores repiten hasta que el grafo deja de cambiar, un **punto fijo**. Pero si incluimos reglas opuestas, como expandir y factorizar la misma expresión sin control, podemos alternar indefinidamente.

Una solución es ordenar fases con propósitos diferentes. Otra consiste en orientar reglas según una medida que disminuye. También puede imponerse un límite de iteraciones con diagnóstico explícito. El límite evita un bloqueo, pero no sustituye una argumentación sobre terminación o estabilidad del pipeline.

::: ejemplo Dos reglas que se pelean
Regla A: `x*(y+z) -> x*y+x*z`. Regla B: `x*y+x*z -> x*(y+z)`. Aplicarlas repetidamente sin estrategia no elige una versión óptima; produce un ciclo. Además, en FP32 ni siquiera se garantiza equivalencia exacta sin permisos numéricos adicionales.
:::

## Constant folding

Calcular una subexpresión formada por constantes durante la compilación se llama constant folding. Para enteros matemáticos es directo mientras respetemos el dominio. Para FP32, evaluar con un `float` de Python puede usar otra precisión y otro orden de redondeo que el programa objetivo.

Un compilador cuidadoso define cómo redondea constantes y operaciones plegadas. En Lumbre las constantes escalares se convierten al tipo FP32 al construirlas y se almacenan mediante una representación hexadecimal. Esto preserva su identidad con más fidelidad que un texto decimal abreviado arbitrariamente.

Plegar funciones trascendentes introduce una dificultad adicional: la biblioteca matemática de la máquina anfitriona puede no coincidir con la del dispositivo objetivo. No basta con que ambas funciones se llamen `exp` para garantizar el mismo último bit.

## Eliminar trabajo muerto

Si una operación no contribuye a ninguna salida y no tiene efectos observables, puede eliminarse. En un grafo puro basta recorrer desde las salidas hacia las entradas para encontrar lo necesario. Una suma construida y abandonada no debe generar un kernel solo porque existe un objeto Python que la describe.

Si una operación escribe un archivo, modifica un parámetro o comunica con otro dispositivo, la ausencia de una salida numérica no significa que sea irrelevante. Por eso la representación de efectos y dependencias de control es necesaria en compiladores más generales.

Lumbre separa la evaluación funcional de nuevos valores y el compromiso de actualizaciones mediante `assign`. Durante la evaluación, el grafo lee el estado viejo. Después se copian las salidas declaradas hacia los parámetros. Esta disciplina hace más fácil razonar sobre qué trabajo es observable.

## Equivalencia: qué debemos conservar

La equivalencia puede significar identidad bit a bit, igualdad matemática bajo hipótesis, proximidad dentro de tolerancia o equivalencia de comportamiento con efectos. No debemos cambiar de criterio a mitad de una comparación.

Para movimientos de memoria, además de valores pueden importar aliasing y escrituras futuras. Para entrenamiento, además de forward importa backward. Una transformación que conserva la salida pero destruye el gradiente no es una optimización válida del programa de entrenamiento.

::: traduccion Un contrato de transformación
Una regla se documenta con entradas admitidas, precondiciones, salida esperada, efectos, semántica numérica y propiedades conservadas. «Probada con cien tensores» describe evidencia experimental; «válida para toda forma estática no negativa bajo estas hipótesis» requiere un argumento general.
:::

## E-graphs y saturación de igualdades

Un motor de reescrituras dirigido elige una representación y puede perder alternativas útiles. Un e-graph agrupa expresiones equivalentes y conserva varias formas. La saturación de igualdades aplica reglas para ampliar ese conjunto y después extrae una opción según un modelo de coste.

El trabajo de `egg` proporciona una implementación y técnicas de mantenimiento de esas clases de equivalencia; su publicación original es de 2021, aunque existe una presentación en Communications of the ACM de 2026. No se debe presentar esa reedición como si el algoritmo hubiera nacido en 2026. [@egg]

La idea no elimina las precondiciones numéricas. Introducir una igualdad falsa en un e-graph contamina las alternativas. Tampoco garantiza automáticamente el mejor rendimiento real: la extracción depende del coste que se haya modelado y de los límites de búsqueda.

## Costes que no se suman de manera local

Supón que dos expresiones comparten una suboperación costosa. Elegir por separado la mejor forma de cada una puede duplicarla. El coste global depende de reutilización, materialización, tamaños y presión de memoria. Una regla «menos nodos siempre es más rápido» no captura estas interacciones.

También hay costes de lanzamiento, compilación y transferencia. Un kernel algo más lento puede ser preferible si evita una transferencia grande. Un grafo muy fusionado puede perder frente a dos kernels porque consume demasiados registros. Las decisiones necesitan modelos y medición.

## Pruebas diferenciales y metamórficas

Una prueba diferencial compara dos implementaciones del mismo contrato: intérprete y C generado, versión sin optimizaciones y versión optimizada, o referencia FP64 y candidato FP32 bajo tolerancia. Una prueba metamórfica utiliza una relación que debe mantenerse, como transponer dos veces o cambiar una dimensión de lote independiente.

Las relaciones metamórficas también tienen hipótesis. Permutar los términos de una reducción no garantiza igualdad bit a bit en FP32. Escalar todas las entradas puede cambiar el régimen de desbordamiento. El test debe declarar qué observa y qué propiedades espera.

## Un expediente de regla

Para cada nueva reescritura, crea una ficha pequeña. La primera línea da el patrón; la segunda, las precondiciones; la tercera, un ejemplo válido; la cuarta, un contraejemplo cuando se elimina una precondición. Añade un test que falle antes de corregir una implementación defectuosa.

Esta disciplina evita colecciones de trucos sin explicación. También facilita revisar código generado por un asistente: una propuesta puede parecer convincente y, aun así, olvidar el signo del cero o una dependencia de memoria.

::: practica Diseñar una regla completa
Quieres eliminar dos permutaciones consecutivas. La primera cambia ejes `(0,1,2)` a `(2,0,1)` y la segunda aplica el orden `(1,2,0)` al resultado. ¿Qué permutación conjunta obtienes? ¿Puedes eliminarla?
:::

**Solución.** La segunda selecciona, en orden, los ejes originales que quedaron en posiciones 1,2,0: son 0,1,2. La composición es la identidad. En una representación pura de vistas puede recuperarse la entrada. Si entre ambas hay una materialización o efecto exigido por el contrato, hay que analizarlo por separado; no se borra una escritura observable solo porque los valores finales coincidan.

::: comprueba Antes de continuar
Explica cuándo se aplica una regla, qué hipótesis exige y cómo se evitan ciclos. Una optimización debe hacer visibles esas condiciones.
:::
