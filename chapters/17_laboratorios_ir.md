# Laboratorio 1. Tu primera cadena completa: de cuatro números a C {#lab:uno}

## Objetivo y preparación

Este laboratorio corresponde al comienzo de las semanas 1–2. La tarea es calcular $y=(x+2)x$ para cuatro números. No vamos a empezar entrenando nada: queremos observar toda la cadena y saber qué parte se ejecuta en Python y qué parte en C. Necesitas el paquete de Lumbre, Python con NumPy y un compilador de C accesible como `cc`. En Windows, el runtime entregado se utiliza desde un entorno Linux, por ejemplo WSL; no es un cargador de DLL nativas de Windows.

Abre una terminal en la carpeta del código. Ejecuta la batería básica antes de cambiar archivos. Si falla por ausencia de compilador, todavía no es un fallo del grafo. Resuelve primero la dependencia y conserva el mensaje original para distinguir instalación de corrección.

```bash
python -m pytest -q
```

La primera compilación puede tardar más que las siguientes porque el runtime conserva bibliotecas por clave de caché. No compares el tiempo de esa primera ejecución con el de una función ya cargada como si fueran la misma fase del trabajo.

## Predicción antes del programa

Toma $x=(-2,-1,0,3)$. Para cada elemento hay dos operaciones: sumar dos y multiplicar por el original. Los resultados deben ser $(0,-1,0,15)$. El valor original no se sobrescribe al sumar: el producto recibe tanto la suma como la entrada.

| Posición | Entrada | Entrada más dos | Producto esperado |
|---|---|---|---|
| 0 | -2 | 0 | 0 |
| 1 | -1 | 1 | -1 |
| 2 | 0 | 2 | 0 |
| 3 | 3 | 5 | 15 |

Esta tabla ya detecta dos fallos frecuentes. Si el programa devuelve cuadrados de la suma, utilizó dos veces el resultado intermedio. Si devuelve $x+2x$, cambió la agrupación. Un ejemplo con una sola entrada igual a cero no distinguiría bien esas equivocaciones.

## Construcción y ejecución

Guarda el siguiente programa como `laboratorio_01.py` en la raíz del paquete y ejecútalo con Python. El nombre es un archivo que tú creas; no se presupone que exista en la entrega.

```python
import numpy as np
from lumbre import param, Program

x = param("x", (4,))
y = (x + 2.0) * x
entrada = np.array([-2, -1, 0, 3], dtype=np.float32)

with Program([y]) as programa:
    salida = programa.run({"x": entrada})[0]
    print(salida)
    print(programa.report())
    print(programa.source)
    np.testing.assert_array_equal(
        salida, np.array([0, -1, 0, 15], dtype=np.float32)
    )
```

`param` crea una descripción de entrada, no un vector con números. `y` es otro nodo del grafo. `Program` decide materializaciones, genera código, lo compila y prepara el almacenamiento. `run` copia la entrada y llama a la función compilada. El arreglo devuelto es una copia de la salida declarada, de modo que no queda dependiendo de una región temporal que se vaya a reutilizar.

Busca en el C generado el acceso al elemento de `x`, la constante y la escritura de la salida. Los nombres automáticos pueden cambiar; no forman parte del resultado matemático. La propiedad importante es que cada posición válida produce exactamente un valor y que ninguna posición necesita el resultado de otra.

## Comparar dos planificaciones sin cambiar la fórmula

Construye también `Program([y], fuse=False)`. La versión sin fusión materializa más pasos. Debe conservar los resultados para esta entrada. No afirmes que una variante es más rápida porque tiene menos líneas: mide después y separa compilación de ejecución.

```python
with Program([y], fuse=True) as unido:
    a = unido.run({"x": entrada})[0]
    informe_a = unido.report()
with Program([y], fuse=False) as separado:
    b = separado.run({"x": entrada})[0]
    informe_b = separado.report()
np.testing.assert_array_equal(a, b)
print(informe_a["kernels"], informe_b["kernels"])
```

En CPU, «kernel» significa aquí una unidad generada del plan, no un lanzamiento GPU real. Esa precisión evita interpretar automáticamente el número de kernels como latencia de GPU. Las unidades pueden convertirse en funciones C llamadas secuencialmente dentro de una ejecución.

## Ejercicio de reescritura con límites explícitos

Sobre enteros matemáticos, $x+0=x$. En coma flotante, algunas identidades necesitan cuidado con ceros con signo, NaN, infinitos y redondeo. Por eso el primer simplificador algebraico del laboratorio trabaja con expresiones enteras de índices, no activa indiscriminadamente reglas de números reales sobre todos los tensores.

Propón ahora la transformación de $(x+2)x$ a $x^2+2x$. Es algebraicamente correcta sobre reales exactos. ¿Es una regla que deba activarse siempre para FP32? La respuesta es no: cambia el orden y número de redondeos, puede alterar resultados extremos y podría aumentar operaciones. Para usarla en un modo aproximado habría que declarar la política numérica y justificar su utilidad, no solo escribir una igualdad simbólica.

::: practica Error intencional y reparación
Cambia la expresión a `(x + 2.0) * (x + 2.0)` y ejecuta la prueba. Debe fallar. Localiza el primer valor que contradice la tabla y dibuja las dos aristas del producto.

**Solución:** con entrada -2, ambas fórmulas dan cero; ese caso no identifica el error. Con entrada -1, la fórmula pedida da -1 y la defectuosa da 1. El producto debe recibir el nodo de entrada en una de sus aristas, no dos copias de la suma. Reparar el resultado final con un signo añadido solo para esa posición sería un parche al ejemplo, no una corrección del programa.
:::

## Entrega y condición de avance

Conserva el programa, una copia del C generado y una explicación de diez líneas del recorrido. Añade una entrada que no aparezca en la tabla y calcúlala a mano. No hace falta adjuntar una biblioteca compilada dependiente de tu máquina.

Puedes pasar al laboratorio siguiente cuando distingas el momento de construir el grafo del momento de ejecutar los datos, y cuando sepas explicar por qué una operación elemento a elemento puede fusionarse sin comunicar posiciones. Un resultado correcto sin esa explicación todavía no muestra que controles el compilador.

# Laboratorio 2. Índices simbólicos, intervalos y máscaras {#lab:dos}

## El problema que vamos a resolver

Tenemos 35 valores y queremos repartirlos en bloques de ocho posiciones. Habrá cinco bloques. Los primeros cuatro cubren 32 valores y el último contiene tres válidos y cinco sobrantes. La tarea es generar índices, demostrar qué posiciones son válidas y evitar leer los sobrantes.

El índice global se calcula como $i=8b+t$, donde $b$ identifica el bloque y $t$ una posición local entre 0 y 7. No estamos hablando todavía de hilos físicos: primero probaremos una relación aritmética. Después podremos mapearla a un bucle, un vector o un hilo GPU.

## Calcular una tabla completa del borde

Para el bloque cuatro, los índices son 32,33,34,35,36,37,38,39. La condición `i < 35` acepta los tres primeros. La condición incorrecta `i <= 35` aceptaría también el 35, que queda fuera de un arreglo de 35 elementos, cuyos índices terminan en 34.

```python
N, B = 35, 8
vistos = []
for bloque in range((N + B - 1) // B):
    for local in range(B):
        i = bloque * B + local
        if i < N:
            vistos.append(i)
assert vistos == list(range(N))
```

Esta comprobación exige orden y cobertura exacta, no únicamente cantidad. Un algoritmo que duplicase el índice 0 y perdiese el 34 también produciría 35 resultados; comparar solo la longitud no lo detectaría.

## Construir la expresión en el módulo simbólico

```python
from lumbre.symbolic import var, simplify, interval, render_c

b = var("b")
t = var("t")
indice = simplify((b * 8 + t) + 0)
print(indice.evaluate({"b": 4, "t": 2}))
print(interval(indice, {"b": (0, 4), "t": (0, 7)}))
print(render_c(indice))
```

La primera salida numérica es 34. El intervalo es `(0,39)`. Eso nos dice que el índice generado **puede** exceder 34; no que todas las combinaciones lo hagan. El simplificador elimina una suma por cero dentro de su dominio entero. El renderer produce una expresión C, pero su contrato exige las condiciones indicadas para división y resto.

El módulo simbólico es una herramienta de laboratorio separada. El renderer tensorial de Lumbre construye sus expresiones de índices directamente. No se atribuye al compilador completo un sistema general de pruebas simbólicas por el hecho de incluir este pequeño módulo.

## Probar una identidad y sus hipótesis

Para un entero no negativo $i$ y un divisor positivo $B$, se cumple $i=(i//B)B+(i\bmod B)$. El cociente identifica el bloque y el resto la posición local. Con $i=34$, obtenemos bloque 4 y resto 2: $4\cdot8+2=34$.

```python
for i in range(40):
    bloque, local = i // 8, i % 8
    assert bloque * 8 + local == i
    assert 0 <= local < 8
```

La prueba por enumeración cubre esos 40 enteros. El argumento general es la división euclídea: el cociente y el resto se definen de forma que el resto está en el rango permitido. Al traducir a C, restringimos índices no negativos para evitar confundir su división truncada con la división suelo de Python cuando aparezcan valores negativos.

## Una máscara debe proteger el acceso, no solo el resultado

Compara las dos formas siguientes, escritas como pseudocódigo:

```text
Forma incorrecta:
    valor = A[i]
    resultado = valido ? valor : 0

Forma correcta para este contrato:
    resultado = valido ? A[i] : 0
```

En la primera ya se ha producido una lectura fuera de límites antes de descartar el valor. La segunda representa una lectura condicionada: su traducción debe preservar que la dirección inválida no se desreferencia. Una optimización no puede mover la carga fuera de la condición si carece de una garantía adicional de acceso seguro.

En una reducción por suma, cero es un relleno apropiado porque no cambia el acumulado. En una reducción por máximo, rellenar con cero puede ser erróneo si todos los valores válidos son negativos. La máscara depende de la operación, no únicamente del tamaño del arreglo.

::: ejemplo Un borde que cambia el máximo
Los valores válidos del último bloque son -5,-2,-9. Su máximo es -2. Si las cinco posiciones sobrantes se rellenan con cero y participan, el resultado pasa a ser cero. El error no se arregla cambiando la tolerancia numérica: es una diferencia semántica. Debe usarse una identidad apropiada o excluir los elementos inválidos de la reducción.
:::

## Intervalos conservadores y precisión del análisis

Si sabemos $b\in[0,4]$ y $t\in[0,7]$, el intervalo de $8b+t$ es `[0,39]`. Si añadimos la relación `b==4`, el intervalo se estrecha a `[32,39]`. Si añadimos además `t<3`, queda `[32,34]`. Cuanta más información relacional conservamos, más accesos pueden demostrarse seguros.

Pero el módulo de intervalos básico no representa todas las relaciones. Si $a$ y $b$ son realmente la misma variable, tratarlas como dos intervalos independientes puede producir una cota muy amplia. Una cota amplia no es un fallo de corrección si contiene todos los valores; es una pérdida de precisión que puede impedir optimizaciones. Lo peligroso es una cota demasiado estrecha que excluya valores posibles.

## Entrega resuelta y variación

La entrega incluye una función que enumera índices para cualquier `N>=0` y bloque positivo, junto con casos `N=0,1,7,8,9,35`. Con `N=0` no debe ejecutarse ningún acceso. Con `N=8` hay un bloque completo; con `N=9` aparece un borde de un solo elemento. Añade validación que rechace bloque cero antes de dividir.

Como variación, usa un tile bidimensional de $4\times8$ para una matriz $5\times10$. La máscara correcta exige a la vez fila menor que 5 y columna menor que 10. Una única condición sobre el índice lineal puede aceptar elementos que cruzan indebidamente la frontera de una fila si la construcción de coordenadas no coincide con el layout declarado. Escribe primero las dos coordenadas y solo después la dirección.

# Laboratorio 3. Vistas sin copia y reducciones sin confusión {#lab:tres}

## Del dibujo al almacenamiento

Tenemos una matriz de dos filas y tres columnas almacenada como `[0,1,2,3,4,5]`. Su primera fila es `[0,1,2]` y la segunda `[3,4,5]`. Queremos observar la transpuesta sin mover esos seis números. La vista debe traducir coordenadas, no modificar el contenido.

```python
from lumbre.layout import View

original = View.contiguous((2, 3))
transpuesta = original.permute((1, 0))
print(original.shape, original.strides)
print(transpuesta.shape, transpuesta.strides)
assert transpuesta.address((2, 1)) == 5
```

La vista original tiene strides `(3,1)`; la transpuesta `(1,3)`. Para la coordenada `(2,1)` de la transpuesta, la dirección es $2\cdot1+1\cdot3=5$. Corresponde al elemento original `(1,2)`. La forma ha pasado a `(3,2)`, pero la memoria sigue siendo la misma lista.

## Por qué aplanar no siempre es una vista contigua

El recorrido lógico por filas de la transpuesta sería `[0,3,1,4,2,5]`. Esa secuencia no está contigua en la memoria original. No basta con cambiar la forma a `(6,)` conservando stride uno: produciría `[0,1,2,3,4,5]`, que representa otro orden lógico.

El método `contiguous_reshape` de la vista del laboratorio rechaza esa conversión cuando necesita una materialización o una indexación más general. Ese rechazo es una propiedad de corrección, no una carencia que deba ocultarse devolviendo cualquier arreglo de seis elementos.

```python
try:
    transpuesta.contiguous_reshape((6,))
except ValueError as error:
    print("Rechazo esperado:", error)
else:
    raise AssertionError("La vista no era contigua")
```

El tensor IR de Lumbre puede expresar composiciones lógicas mediante transformaciones de índices; eso no significa que su runtime acepte cualquier buffer externo con strides arbitrarios. Las entradas se copian a almacenamiento contiguo. Separar ambos contratos evita atribuir zero-copy a una copia de entrada que realmente existe.

## Broadcasting como stride cero

Una fila de tres valores puede usarse como sesgo de muchas filas. En el modelo de vista, expandir un eje de tamaño uno permite usar stride cero: cambiar esa coordenada lógica no avanza en memoria. Así varias posiciones leen el mismo valor.

```python
fila = View.contiguous((1, 3))
repetida = fila.expand((4, 3))
assert repetida.strides == (0, 1)
assert repetida.address((0, 2)) == 2
assert repetida.address((3, 2)) == 2
```

Leer es sencillo; escribir ya no lo es. Dos posiciones lógicas pueden señalar el mismo elemento físico. Si se permite una operación in-place sobre la vista expandida, distintas escrituras podrían entrar en conflicto. El laboratorio utiliza estas vistas para leer y razonar, no promete un sistema general de actualizaciones in-place sobre alias.

## Reducir una dimensión y conservar las otras

Considera la matriz `[[1,2,3],[4,5,6]]`. Sumar por el último eje produce `[6,15]`: una contribución por fila. Sumar por el primer eje produce `[5,7,9]`: una contribución por columna. Sumar todo produce el escalar 21. El nombre «reducción» no especifica por sí solo qué dimensión desaparece.

```python
import numpy as np
from lumbre import param, Program

x = param("datos", (2, 3))
filas = x.sum(axis=1)
columnas = x.sum(axis=0)
total = x.sum()
with Program([filas, columnas, total]) as p:
    a, b, c = p.run({"datos": [[1, 2, 3], [4, 5, 6]]})
np.testing.assert_array_equal(a, [6, 15])
np.testing.assert_array_equal(b, [5, 7, 9])
assert float(c) == 21.0
```

Escribe el C de referencia para cada caso antes de consultar el generado. Para sumar filas, el bucle exterior escoge la fila y el interior recorre columnas. Para sumar columnas, ocurre lo contrario. En ambos casos el acumulador se inicializa una vez por salida; si se inicializa dentro del bucle de reducción, solo queda la última contribución.

## Composición que obliga a seguir índices

Transpone la matriz, selecciona las dos últimas filas de la transpuesta y suma cada fila. La transpuesta es `[[1,4],[2,5],[3,6]]`; el recorte es `[[2,5],[3,6]]`; la salida debe ser `[7,9]`.

```python
y = x.permute(1, 0).slice(0, 1, 3).sum(axis=1)
with Program([y]) as p:
    salida = p.run({"datos": [[1, 2, 3], [4, 5, 6]]})[0]
np.testing.assert_array_equal(salida, [7, 9])
```

El propósito no es una operación compleja, sino comprobar que cada transformación compone el mapa anterior. Un fallo frecuente aplica el recorte sobre la forma original porque se conservó un eje antiguo. La forma de cada nodo debe describir su salida, no la del primer tensor de la cadena.

::: practica Gradiente del sesgo, adelantando la semana 9
Una matriz de cuatro filas suma el mismo sesgo de tres elementos. La pérdida es la suma de todos los resultados. ¿Qué gradiente recibe cada elemento del sesgo?

**Solución:** cada uno se utilizó cuatro veces y cada uso contribuye uno; el gradiente es `[4,4,4]`. El broadcasting que en forward repite lecturas se convierte en una suma de contribuciones en backward. No se devuelve una matriz de cuatro filas como gradiente de un parámetro cuya forma era `(3,)`.
:::

## Condición de éxito

Entrega los mapas de dirección de la vista original y transpuesta, la explicación del reshape rechazado y tres reducciones comprobadas. Añade una matriz no cuadrada: las cuadradas ocultan algunos intercambios de ejes porque la forma no cambia. Tu argumento debe explicar el resultado para una coordenada arbitraria válida, no solo para las cuatro direcciones que has probado.

# Laboratorio 4. Planificar kernels y reciclar memoria sin romper datos {#lab:cuatro}

## Un grafo con una dependencia compartida

Construye $h=\tanh(x)$ y dos salidas: $a=h+1$, $b=h\cdot h$. El nodo $h$ se usa en varias aristas. Un plan puede calcularlo una vez y conservarlo, o recomputarlo dentro de cada consumidor. La decisión afecta trabajo y memoria; no puede cambiar el resultado matemático pedido.

```python
import numpy as np
from lumbre import param, Program

x = param("x", (101,))
h = x.tanh()
a, b = h + 1.0, h * h
entrada = np.linspace(-2, 2, 101, dtype=np.float32)
with Program([a, b]) as p:
    ya, yb = p.run({"x": entrada})
    print(p.report())
np.testing.assert_allclose(ya, np.tanh(entrada) + 1, atol=1e-6)
np.testing.assert_allclose(yb, np.tanh(entrada)**2, atol=1e-6)
```

Que $h$ tenga varios usos no obliga a materializarlo en todo compilador posible. Es una política razonable para evitar recomputaciones; una política más sofisticada compararía tamaño, coste y presión de memoria. Lumbre usa reglas simples y observables para que el estudiante pueda explicar su decisión.

## Calcular vidas a mano

Usa este plan secuencial imaginario: K0 produce `h`; K1 lee `h` y produce `a`; K2 lee `h` y produce `b`; al final se devuelven `a` y `b`. `h` debe vivir hasta K2. `a` debe conservarse hasta que el usuario lo lea, aunque ningún kernel posterior lo utilice. Las salidas declaradas tienen una vida diferente de un temporal interno.

| Valor | Nace | Último uso necesario | ¿Puede reciclarse antes de K2? |
|---|---|---|---|
| x | Antes de K0 | K0 | Solo bajo una política explícita de entradas |
| h | K0 | K2 | No |
| a | K1 | Lectura final | No |
| b | K2 | Lectura final | No |

Un planificador que solo cuente consumidores dentro del grafo podría considerar `a` muerto justo después de producirlo y sobrescribirlo. Para evitarlo, las salidas deben actuar como raíces retenidas. La interfaz del programa, no solo las aristas internas, determina la vida de memoria.

## Un ejemplo donde sí se puede reutilizar

En una cadena $u_1=f(x)$, $u_2=g(u_1)$, $u_3=h(u_2)$, el almacenamiento de $u_1$ puede reutilizarse después de que K1 haya terminado de leerlo. No puede reutilizarse al inicio del mismo kernel si la escritura se solapa con lecturas futuras y no se ha demostrado seguridad in-place.

Una regla conservadora para kernels secuenciales exige que el último uso del buffer sea **estrictamente anterior** al kernel que lo recibe. La igualdad no basta. Esa desigualdad pequeña evita que una función sobrescriba datos que todavía está recorriendo. En ejecución asíncrona, además hay que traducir el último uso lógico a una garantía de finalización del dispositivo.

## Comparación de las cuatro combinaciones

Construye una cadena de doce pasos `tanh(0.9*u+0.1)`. Ejecuta versiones con fusión y reutilización activadas o desactivadas. Hay dos variables independientes; no cambies ambas y atribuyas toda diferencia a una sola.

```python
u = x
for _ in range(12):
    u = (u * 0.9 + 0.1).tanh()
referencia = None
for fusion in (False, True):
    for reciclaje in (False, True):
        with Program([u], fuse=fusion, reuse=reciclaje) as p:
            resultado = p.run({"x": entrada})[0]
            print(fusion, reciclaje, p.report())
        if referencia is None:
            referencia = resultado
        else:
            np.testing.assert_allclose(
                resultado, referencia, rtol=1e-5, atol=1e-6
            )
```

La arena informada incluye categorías propias de la implementación, como entradas y temporales alineados. No es automáticamente el pico total del proceso Python ni la memoria reservada por un driver. Para comparar dos informes, utiliza la misma definición y explica qué queda fuera.

## Una escritura de parámetros es una transacción

En entrenamiento, varias actualizaciones deben calcularse a partir del mismo estado antiguo. Si se actualiza una matriz mientras todavía se calcula el gradiente de otra que depende de ella, el paso deja de corresponder al optimizador definido. Por eso Lumbre construye salidas de actualización y las aplica después de ejecutar el grafo completo.

No confundas esa organización con una base de datos transaccional general. Aquí la garantía es local y acotada: las entradas del paso permanecen separadas de sus salidas y se confirma la actualización cuando el cálculo ha terminado. Recuperarse de una interrupción del proceso exige además checkpoints completos y una escritura persistente segura.

::: practica Encuentra el fallo sin mirar números
Un planificador asigna el mismo offset a dos buffers de 256 bytes porque tienen el mismo tamaño. El primero se usa en K5; el segundo nace en K4. ¿Qué información falta?

**Solución:** el tamaño solo prueba que caben en la misma región, no que puedan compartirla en el tiempo. Sus vidas se solapan. El segundo no puede ocuparla hasta que termine el último uso del primero. También habría que revisar alineamiento, tipo y restricciones de propiedad; ninguno sustituye la condición temporal.
:::

## Entrega

Entrega un dibujo del DAG, una tabla de vidas y los cuatro informes. Explica qué cambio reduce kernels y cuál reduce bytes. Si un cambio empeora el tiempo, conserva la medida y busca una causa: mayor código, presión de registros, compilación o ruido. Una optimización tiene una intención, pero su nombre no es una prueba de que mejore todas las métricas.
