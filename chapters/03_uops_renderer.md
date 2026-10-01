# UOps: construir un grafo de intenciones {#ch:uops}

## De una fórmula escrita a piezas conectadas

Queremos calcular `y=(x+2)*(x+2)`. Un texto repite dos veces `x+2`. Un dibujo puede mostrar una sola suma cuyo resultado se utiliza dos veces. Esa representación conserva explícitamente la dependencia: la multiplicación necesita la salida de la suma, y la suma necesita x y la constante dos.

```tikz
\begin{center}
\begin{tikzpicture}[node distance=1.3cm and 1.7cm]
\node[dot] (x) {$x$};
\node[dot,right=of x] (c) {$2$};
\node[dot,below right=.8cm and .2cm of x] (a) {$+$};
\node[dot,below=1.2cm of a] (m) {$\times$};
\draw[flow] (x)--(a);\draw[flow] (c)--(a);
\draw[flow] (a) to[bend right=22] (m);\draw[flow] (a) to[bend left=22] (m);
\node[right=.5cm of m,font=\small,align=left] {La misma suma alimenta\\las dos entradas del producto.};
\end{tikzpicture}
\end{center}
```

Cada pieza es una operación con sus entradas. En la tradición de tinygrad se utiliza el nombre **UOp**. Lumbre toma ese nombre didáctico, pero su estructura y su conjunto de operaciones son propios. No prometemos compatibilidad de código con la API interna de tinygrad. [@tinygrad-dev]

## Qué información necesita un nodo

Para interpretar una suma necesitamos saber que es una suma, cuáles son sus operandos, qué forma tiene su salida y qué tipo numérico utiliza. Una constante necesita además su valor. Una permutación necesita el orden de los ejes. Un parámetro necesita un nombre que conecte el grafo con los datos de ejecución.

```python
from dataclasses import dataclass

@dataclass(frozen=True, eq=False)
class Nodo:
    op: str
    shape: tuple[int, ...]
    src: tuple["Nodo", ...] = ()
    arg: object = None
    dtype: str = "float32"
```

Este es un modelo explicativo reducido. El listado completo añade utilidades y validaciones. `frozen=True` expresa que no modificaremos los campos después de crear el nodo. La inmutabilidad facilita reutilizar nodos y razonar sobre transformaciones: una suma antigua no cambia por accidente cuando otro pase la inspecciona.

`src` contiene referencias a otros nodos, no copias de sus valores numéricos. El grafo describe una computación pendiente. Crear el nodo de una suma no exige tener todavía el array que sumaremos.

## Un grafo dirigido sin ciclos

Las aristas van de las entradas hacia sus usos. Si ninguna operación termina dependiendo de sí misma a través de un camino, tenemos un grafo dirigido acíclico, o DAG. El nombre parece abstracto, pero aquí significa que podemos ordenar las operaciones para calcular antes sus entradas.

El grafo de una ejecución tensorial finita no tiene por qué representar todos los bucles del programa original como ciclos. Puede contener una operación de reducción que más adelante se convierta en un bucle. También puede construirse un grafo distinto para cada iteración de entrenamiento, reutilizando su estructura y cambiando parámetros.

Los bucles y el control de flujo general requieren representaciones adicionales si queremos compilarlos sin desplegar cada paso. Lumbre trabaja con grafos estáticos de operaciones tensoriales; esta decisión reduce la superficie inicial y hace visibles las extensiones necesarias.

## Orden topológico explicado como lista de tareas

Imagina una receta: primero necesitas ingredientes, después una mezcla, después el horno. Un orden topológico es una lista donde toda tarea aparece después de sus requisitos. Para `(x+2)^2`, un orden válido es x, constante, suma, multiplicación.

Un recorrido en profundidad puede visitar entradas antes de añadir un nodo a la lista. Hay que recordar qué nodos ya se han incluido, porque una misma suma puede utilizarse dos veces. De lo contrario repetiríamos trabajo y podríamos contar mal los usos.

```python
def ordenar(salidas):
    vistos, resultado = set(), []
    def visitar(nodo):
        if nodo in vistos:
            return
        vistos.add(nodo)
        for entrada in nodo.src:
            visitar(entrada)
        resultado.append(nodo)
    for salida in salidas:
        visitar(salida)
    return resultado
```

Este fragmento supone que el grafo ya es acíclico y es pequeño. Un frontend abierto debe detectar ciclos o impedirlos por construcción. La versión ejecutable usa un recorrido iterativo para no depender de la profundidad máxima de recursión de Python.

## Identidad frente a igualdad estructural

Dos objetos pueden describir la misma suma sin ser el mismo objeto. Compartirlos permite representar una expresión común una sola vez. A esta técnica se la suele llamar **hash-consing**: construir una clave inmutable y consultar si ya existe un nodo con esa descripción.

En Lumbre la clave incluye operación, forma, entradas, argumento y tipo. Las entradas se comparan por identidad de nodo dentro del grafo. Si construimos repetidamente una misma operación sobre las mismas entradas, el constructor con caché puede devolver el mismo nodo.

No podemos compartir indiscriminadamente operaciones con efectos. Dos lecturas de un buffer separadas por una escritura no tienen necesariamente el mismo valor. La deduplicación de operaciones puras es una cosa; el control de dependencias de memoria es otra.

::: ejemplo Contar usuarios correctamente
En `z=a*a`, el nodo a aparece dos veces como entrada de la misma multiplicación. Si contamos usos para decidir materialización, hay dos aristas aunque solo haya una operación consumidora. Un algoritmo debe decidir si necesita contar aristas o consumidores distintos y utilizar esa definición de manera consistente. Lumbre cuenta referencias de entrada para su heurística de fusión.
:::

## El frontend verifica el significado

El constructor de suma calcula la forma resultante mediante broadcasting. El de matmul comprueba la dimensión K. El de permutación valida que cada eje aparezca una vez. Cuanto antes detectemos un contrato inválido, más comprensible será el error.

Un renderer no debería descubrir por primera vez que las matrices no encajan cuando ya está escribiendo una llamada de bajo nivel. La separación deseable es: el frontend define y valida una operación; los pases conservan sus invariantes; el backend la implementa para un dispositivo concreto.

```python
from lumbre import param
A = param("A", (2, 3))
B = param("B", (3, 4))
C = A @ B
assert C.shape == (2, 4)
```

Esto construye un nodo con forma `(2,4)`. No multiplica arrays en Python. Para ejecutar, tendremos que crear un programa, inicializar A y B, y llamar a su runtime.

## Por qué la forma es información del compilador

Conocer dimensiones permite calcular cuánto espacio reservar, qué índices son válidos y qué bucles generar. Una dimensión simbólica introduce otra pregunta: ¿qué sabemos sobre su valor y qué debemos comprobar en ejecución? El primer compilador utiliza formas estáticas para separar ambas dificultades.

Estático no significa constante: A y B pueden cambiar en cada llamada manteniendo sus formas. Así se actualizan pesos sin reconstruir dimensiones.

## Parámetros y constantes

Un parámetro recibe datos al ejecutar; una constante forma parte del programa. El incremento dos puede incorporarse al código o leerse como parámetro. La primera opción especializa más; la segunda reutiliza el ejecutable para distintos valores.

Especializar cada longitud, escala y valor multiplica variantes, caché y compilaciones. La API debe distinguir qué valores identifican el programa y cuáles son datos de ejecución.

## Representación impresa y representación interna

Imprimir el grafo ayuda a depurar, pero sus identificadores pueden depender del orden de creación. Una clave de caché estable utiliza contenido canónico y configuración, no direcciones de objetos Python ni un texto de depuración accidental.

Lumbre utiliza el código generado y la identidad del toolchain para su caché de módulos. Esa estrategia es sencilla y visible. No es la única posible: compiladores grandes también almacenan IR serializada, resultados de autotuning y artefactos intermedios con versiones propias.

## Ejercicios de construcción

::: practica Dibujar antes de programar
Dibuja `r=(a+b)*(a+b)+c`. ¿Qué nodos pueden compartirse? Si a, b y c tienen forma `(8,)`, ¿qué forma tiene cada nodo? ¿En qué orden podrías evaluarlos?
:::

**Solución.** Los parámetros a, b y c son hojas. La suma a+b puede representarse una vez y alimentar las dos entradas del producto. El último nodo suma ese producto y c. Todos tienen forma `(8,)`. Un orden válido es a, b, suma, producto, c, suma final; c también puede aparecer antes porque no depende de los otros cálculos.

::: practica Encontrar el contrato ausente
Alguien construye un nodo `matmul` sin comprobar formas y afirma que el backend decidirá qué significa. ¿Qué problema crea? ¿Basta con que el producto total de elementos coincida?
:::

**Solución.** El mismo nodo podría representar cálculos diferentes o accesos inválidos. La igualdad del número total de elementos no determina las dimensiones de contracción. El frontend debe fijar una semántica inequívoca antes de que el backend escoja una implementación.

::: comprueba Antes de continuar
Explica UOp, DAG, orden topológico, parámetro, constante y operación pura con el ejemplo de una sola fórmula. Después localiza esas ideas en `lumbre/ir.py` sin intentar memorizar todas sus líneas.
:::

# Del grafo al C: renderer, ABI y primera ejecución {#ch:renderer}

## Traducir sin cambiar de tarea

Ya sabemos representar `y=(x+2)*3`. Ahora necesitamos instrucciones que calculen sus valores. Un **renderer** recorre una representación y escribe código en un lenguaje objetivo. En el primer backend el objetivo es C: bucles, accesos a memoria y operaciones escalares.

Para cuatro elementos podemos generar un bucle que calcule toda la fórmula por posición. No hace falta crear arrays intermedios para `x+2` si ninguna otra parte necesita observarlos. La representación tensorial se transforma en una expresión escalar dentro de un recorrido.

```c
void desplazar_escalar(const float* x, float* y, long n) {
    for (long i = 0; i < n; i++) {
        y[i] = (x[i] + 2.0f) * 3.0f;
    }
}
```

El código tiene un contrato: x apunta a n valores legibles; y apunta a n posiciones escribibles; las reglas de aliasing que adoptemos deben respetarse. No basta con imprimir una fórmula correcta si los punteros no apuntan al espacio adecuado.

## Qué es un puntero sin metáforas misteriosas

Un puntero contiene una dirección de memoria. Podemos imaginar la memoria como una calle de bytes y el puntero como el número del primer portal de un array. Para acceder al elemento i de `float*`, C calcula un desplazamiento de i elementos del tamaño de `float`, no de i bytes.

Si un array FP32 comienza en una dirección A, su elemento 3 comienza doce bytes después. Es importante mantener separadas dos unidades: índices en elementos y desplazamientos de memoria en bytes. El planificador de memoria de Lumbre calcula offsets en bytes; las expresiones de acceso dentro de un tensor utilizan índices de elementos.

## Firmas y ABI

Una firma describe argumentos y resultado de una función. La **ABI**, interfaz binaria de aplicación, define cómo se transmiten esos argumentos y resultados en el código máquina de una plataforma. Python y C tienen que ponerse de acuerdo para que una llamada no interprete una dirección como un entero pequeño o un número de coma flotante como un puntero.

En `ctypes` declaramos tipos explícitamente. El siguiente fragmento ilustra una función que recibe dos punteros y una longitud; no incluye la compilación de la biblioteca.

```python
import ctypes as C
biblioteca = C.CDLL("./operaciones.so")
funcion = biblioteca.desplazar_escalar
funcion.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_long]
funcion.restype = None
```

No conviene suponer que `long` tiene el mismo tamaño en todos los sistemas. En el código del proyecto se prefieren tipos de anchura explícita para índices y se define la firma del runtime como un puntero base. La portabilidad necesita comprobar ABI, alineación y plataforma, no solo cambiar la extensión del archivo.

## Compilar una biblioteca compartida

Una biblioteca compartida contiene funciones cargables desde otro programa. Para un ejemplo Linux:

```bash
cc -O3 -std=c11 -shared -fPIC operaciones.c -o operaciones.so
```

`-shared` pide una biblioteca compartida y `-fPIC` genera código adecuado para cargarse en distintas direcciones. Lumbre añade las opciones numéricas y la biblioteca matemática que necesita. La orden real se construye como una lista de argumentos; se captura su salida y se comprueba el código de retorno.

El hecho de que la compilación termine solo garantiza que el compilador C ha producido un artefacto. Todavía faltan carga, inicialización y ejecución. Algunas incompatibilidades aparecen al cargar una biblioteca; otras, al ejecutar una instrucción que el procesador no soporta.

## Un renderer recursivo reducido

Para operaciones puras elemento a elemento, podemos traducir cada nodo a una expresión. Este fragmento es un ejemplo aislado de estructura, no el renderer completo de Lumbre.

```python
def expresion(u, i):
    if u.op == "param":
        return f"{u.arg}[{i}]"
    if u.op == "const":
        return literal_float(u.arg)
    if u.op == "add":
        return f"({expresion(u.src[0], i)} + {expresion(u.src[1], i)})"
    if u.op == "mul":
        return f"({expresion(u.src[0], i)} * {expresion(u.src[1], i)})"
    raise NotImplementedError(u.op)
```

`literal_float` debe imprimir una constante válida del tipo requerido. El renderer real también traduce índices de broadcasting, movimientos, reducciones y operaciones materializadas. La recursión es una forma cómoda de mostrar dependencias, pero un grafo grande necesita controlar profundidad, duplicación y coste de generación.

## Paréntesis y precedencia

La expresión `a+b*c` no significa `(a+b)*c`. Un generador de código debe conservar la estructura del grafo y no confiar en que la forma impresa se «parezca» a la fórmula. Añadir paréntesis explícitos en un renderer inicial simplifica la corrección, aunque produzca texto más largo.

También hay que evitar inyectar nombres arbitrarios en el código. Lumbre utiliza identificadores internos para punteros y valida los parámetros mediante el grafo. Un usuario no debería poder introducir una instrucción C escondida en el nombre de un tensor.

## Fusión: un recorrido en lugar de varios

Considera `a=x+2`, `b=a*3`, `y=b-1`. Una implementación separada realiza tres recorridos y escribe a y b en memoria. Una fusionada calcula `(x[i]+2)*3-1` y escribe únicamente y. El número de operaciones aritméticas puede ser el mismo, pero disminuye el tráfico de memoria intermedia.

La fusión no siempre gana. Una expresión compartida por varios consumidores puede repetirse si se inserta en cada uno. Una expresión excesivamente grande puede aumentar el tiempo de compilación o la presión de registros. Por eso Lumbre utiliza una heurística limitada de profundidad y usos, en lugar de fusionar indiscriminadamente todo el grafo.

::: ejemplo Contar bytes sin medir todavía
Para N elementos FP32, cada recorrido que lee un array y escribe otro mueve aproximadamente 8N bytes a nivel de accesos solicitados. Tres recorridos elementales requieren aproximadamente 24N bytes; uno fusionado, 8N. La cifra es un modelo de tráfico solicitado, no una medición exacta de DRAM: cachés, escrituras y otros detalles pueden cambiar el tráfico físico.
:::

## Primera ejecución completa con Lumbre

```python
import numpy as np
from lumbre import param, Program

x = param("x", (4,))
y = (x + 2.0) * 3.0 - 1.0
with Program([y]) as programa:
    resultado = programa.run({"x": np.array([1, 2, 3, 4], dtype="float32")})[0]
    np.testing.assert_array_equal(resultado, [8, 11, 14, 17])
    print(programa.report())
```

`Program([y])` declara qué salidas necesitamos conservar. `run` inicializa x con los datos recibidos, ejecuta el módulo y devuelve una copia de cada salida solicitada. La lista externa permite programas con varias salidas, como pérdida, norma de gradiente y nuevos parámetros.

La copia de salida es una decisión del runtime didáctico: evita devolver una vista que cambie inesperadamente en la siguiente ejecución. Cuando medimos únicamente kernels residentes podemos pedir `read=[]` para no incluir esa copia. El nombre de la medición debe reflejarlo.

## Caché de compilación

Si el código generado, el compilador y las opciones coinciden, podemos reutilizar una biblioteca existente. Una clave incompleta es peligrosa. Dos módulos con el mismo texto pero compilados para arquitecturas distintas no son necesariamente intercambiables. Tampoco lo son variantes con contratos numéricos incompatibles.

La clave de Lumbre incluye fuente, ruta y versión del compilador, opciones, máquina y sistema. Se escribe primero un artefacto temporal y después se publica mediante renombrado. Esto reduce la posibilidad de observar un archivo a medio escribir. No convierte la caché en un almacén distribuido seguro ni resuelve por sí solo todos los casos de concurrencia.

## Errores que deben resultar visibles

Un compilador correcto rechaza una operación no soportada en lugar de devolver ceros. Un runtime correcto detecta parámetros sin inicializar. Un test correcto verifica salida y forma. Una herramienta reproducible conserva la fuente generada cuando el compilador C falla.

Hay un riesgo particular en los sistemas educativos: hacer que el ejemplo «funcione» delegando silenciosamente a NumPy cuando una operación no se puede compilar. Eso puede ser una estrategia válida si se declara como fallback, pero falsea un benchmark de compilador propio si se oculta. Lumbre no utiliza ese fallback para los kernels de entrenamiento.

::: practica Diseñar un test de integración
Construye `(x+1)*(x-1)` con cinco elementos, incluyendo cero y valores negativos. Compara el resultado generado con el cálculo Python. Después repite la ejecución con otros valores pero la misma forma. ¿Debería recompilarse por haber cambiado x?
:::

**Solución.** La identidad matemática es `x*x-1`, pero el test debe evaluar la expresión que realmente se ha pedido bajo el contrato elegido. Para `[-2,-1,0,1,2]` se obtienen `[3,0,-1,0,3]`. Cambiar los datos de x no modifica el grafo ni sus formas, de modo que el mismo programa compilado puede reutilizarse.

::: comprueba Antes de continuar
Explica qué recibe una función C, qué conserva el grafo y qué identifica la caché. Localiza un bucle en la fuente generada y relaciónalo con una salida del programa. Después distingue el tiempo de compilar del de volver a ejecutar.
:::
