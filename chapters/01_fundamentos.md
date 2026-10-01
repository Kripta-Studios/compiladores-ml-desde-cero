# Un programa que fabrica otro programa {#ch:programa}

::: idea La pregunta de este capítulo
¿Cómo puede una lista de números convertirse en una tarea para un compilador? Empezaremos ejecutando a mano. Después escribiremos instrucciones, y solo al final construiremos un programa que escriba esas instrucciones por nosotros.
:::

## Cuatro temperaturas y una regla

Tenemos los números 10, 12, 15 y 9. Queremos sumar dos a cada uno. La salida debe ser 12, 14, 17 y 11. No estamos ordenando los números ni sumándolos entre sí. Tampoco estamos pidiendo el promedio. Precisar la tarea es el primer paso de cualquier optimización: una versión que resuelve otro problema no es una versión más rápida.

Imagina cuatro casillas numeradas desde cero. La casilla 0 contiene 10; la 1 contiene 12; la 2 contiene 15; la 3 contiene 9. La numeración permite referirse a una posición sin escribir su valor. Esa posición se llama **índice**. El valor puede cambiar sin que cambie la posición.

::: ejemplo Ejecución sin ordenador
En el paso 0 leemos 10 y escribimos 12 en la casilla de salida 0. En el paso 1 leemos 12 y escribimos 14 en la salida 1. Después hacemos lo mismo con 15 y con 9. Cada paso depende de una única entrada. Por tanto, el orden de esos cuatro pasos no afecta a la salida, siempre que cada uno escriba en su propia casilla y las entradas no se destruyan antes de leerlas.
:::

Ya hemos descubierto una propiedad de paralelismo sin usar la palabra GPU. Las tareas son independientes porque los datos que necesitan no son resultados de otras tareas. Más adelante encontraremos operaciones donde esa independencia no existe, como acumular una suma en una única casilla.

## Variables: etiquetas para valores

En Python podemos guardar un número con una etiqueta:

```python
x = 10
y = x + 2
print(y)
```

La primera línea asocia `x` al número 10. La segunda calcula una suma y asocia su resultado a `y`. La tercera muestra 12. El signo `=` en un programa es una instrucción de asignación, no una afirmación matemática eterna. Si después escribimos `x = 50`, no cambia retrospectivamente el cálculo anterior de `y`.

Una lista agrupa varias posiciones. La expresión `temperaturas[2]` accede a la tercera, porque empezamos a contar por cero. Esa convención puede resultar incómoda al principio, pero simplifica direcciones: la primera posición se encuentra a desplazamiento cero respecto al comienzo.

```python
temperaturas = [10, 12, 15, 9]
salida = []
for x in temperaturas:
    salida.append(x + 2)
assert salida == [12, 14, 17, 11]
```

`for` repite el bloque sangrado para cada valor. `append` añade una casilla al final. `assert` pide detenerse si una condición no se cumple. En este ejemplo la aserción convierte una intención verbal en una comprobación automática. No prueba todos los casos posibles; prueba este caso concreto.

## De la lista al array de C

C permite describir un array de valores del mismo tipo. Aquí utilizaremos `float`, un formato de coma flotante que estudiaremos después. El siguiente programa es completo.

```c
#include <stdio.h>

int main(void) {
    float x[4] = {10, 12, 15, 9};
    float y[4];
    for (int i = 0; i < 4; i++) {
        y[i] = x[i] + 2.0f;
    }
    for (int i = 0; i < 4; i++) {
        printf("%.1f\n", y[i]);
    }
    return 0;
}
```

El primer `for` contiene tres decisiones: empezar en cero, seguir mientras el índice sea menor que cuatro y aumentar uno al terminar cada vuelta. La condición es `i < 4`, no `i <= 4`. La casilla 4 no pertenece a un array de cuatro posiciones. Leerla o escribirla constituye un acceso fuera de límites.

Las llaves delimitan bloques. El punto y coma termina muchas instrucciones. La letra `f` en `2.0f` indica una constante de tipo `float`. `return 0` devuelve al sistema operativo un código de finalización satisfactoria. Estos detalles son sintaxis: permiten que el compilador interprete inequívocamente nuestro texto.

## Compilar y ejecutar no son la misma acción

Guarda el programa como `temperaturas.c`. Compilar significa convertir ese texto en otro artefacto ejecutable. En un entorno Linux con un compilador C instalado:

```bash
cc -O0 temperaturas.c -o temperaturas
./temperaturas
```

La primera orden produce el archivo ejecutable. La segunda lo ejecuta. Si modificas el archivo C pero no vuelves a compilar, el ejecutable anterior no se actualiza por intuición. Si una compilación falla y después ejecutas un archivo antiguo que todavía existe, puedes estar observando el programa equivocado. Por eso los laboratorios conservan la orden, el código de salida y el hash del artefacto.

`-O0` desactiva gran parte de las optimizaciones del compilador C y facilita comenzar. Más adelante compararemos con `-O3`. Aumentar el nivel de optimización no convierte en correcto un acceso inválido ni vuelve segura una carrera entre hilos.

::: cuidado El primer error útil
Cambia temporalmente `i < 4` por `i <= 4`. El programa puede parecer funcionar, mostrar un valor extraño o fallar. Esa variabilidad no demuestra que el error sea inofensivo: has pedido una escritura que no pertenece al array. La ausencia de un fallo visible no equivale a seguridad de memoria.
:::

## Qué hace un compilador de aprendizaje automático

Un compilador C recibe instrucciones como bucles, llamadas y operaciones escalares. Nuestro compilador de ML recibirá intenciones más altas: sumar tensores, reorganizar sus ejes, multiplicar matrices o calcular una reducción. Su trabajo consiste en decidir qué bucles y accesos implementan esas intenciones.

Para las temperaturas, la intención es «cada salida es su entrada más dos». Esa descripción no obliga a recorrer de izquierda a derecha, usar una única CPU ni materializar resultados intermedios. El compilador puede escoger una ejecución siempre que preserve el contrato de entrada y salida.

```diagram
Intención: y = x + 2 | Compilador de tensores | Bucle C y ejecutable
```

No llamaremos compilador a cualquier función que invoque una biblioteca. Las llamadas a bibliotecas pueden formar parte legítima de un compilador, pero el trabajo de representación, análisis, transformación y generación debe quedar visible. En Lumbre construiremos esas etapas en lugar de ocultarlas detrás de `torch.compile`.

## Un generador de texto ya es un primer paso

Este programa Python escribe un fragmento de C. Todavía no optimiza ni valida nada; sirve para entender que los programas también son datos.

```python
def generar_suma(n: int, incremento: float) -> str:
    if n < 0:
        raise ValueError("n no puede ser negativo")
    return (
        f"for (int i = 0; i < {n}; i++) {{\n"
        f"    y[i] = x[i] + {incremento:.9g}f;\n"
        "}\n"
    )

print(generar_suma(4, 2.5))
```

El resultado es texto, no el array de salida. Ese texto puede colocarse dentro de una función C y compilarse. Hay ahora dos momentos distintos: el generador se ejecuta para construir código; el código construido se ejecuta después para procesar datos. Mezclarlos es una de las confusiones más frecuentes del curso.

La pequeña función tampoco es todavía un renderer robusto de constantes: para un valor como 2 podría imprimir `2f`, que no es la forma C que buscamos. Esta limitación es deliberada y local al ejemplo introductorio. El compilador completo utiliza literales hexadecimales de coma flotante y conserva un contrato de tipos. Aprenderemos por qué no basta con convertir cualquier objeto a texto.

## Un problema, tres tiempos

El tiempo de construir el grafo, el tiempo de compilar el código y el tiempo de ejecutarlo responden a preguntas distintas. Supón que generar y compilar cuesta 0,3 segundos, mientras ejecutar tarda 0,001 segundos. Una sola llamada cuesta aproximadamente 0,301 segundos. Mil llamadas que reutilizan el ejecutable cuestan aproximadamente 1,3 segundos, sin contar otros gastos. No es correcto anunciar 0,001 segundos como tiempo de la primera llamada.

Estas cifras son un ejemplo aritmético inventado, no un benchmark. Su utilidad es mostrar la **amortización**: un coste inicial se reparte entre varias ejecuciones. La caché de compilación intenta evitar pagar ese coste otra vez para programas equivalentes bajo la misma configuración.

## Ejercicios con respuesta explicada

::: practica Predecir antes de ejecutar
Cambia la regla por «multiplica por tres y resta uno». Calcula la salida para 2, 0 y -1. Después escribe la expresión Python y el cuerpo del bucle C. ¿Necesitas que la entrada tenga exactamente tres elementos?
:::

**Solución.** Las salidas son 5, -1 y -4. La expresión es `3*x - 1`. El cuerpo C puede ser `y[i] = 3.0f*x[i] - 1.0f;`. El número de posiciones es independiente de la regla aritmética: el bucle utiliza una longitud `n`. El resultado de una posición no depende de las otras.

::: practica Reconocer el artefacto
Un programa Python devuelve la cadena `"y[0] = x[0] + 2;"`. ¿Ha calculado ya la salida? Un compilador produce un ejecutable sin errores. ¿Ha demostrado que la fórmula corresponde a la tarea pedida?
:::

**Solución.** En el primer caso ha construido una instrucción, no la ha aplicado a datos. En el segundo ha aceptado el programa bajo las reglas del lenguaje y sus propias comprobaciones; no conoce necesariamente la intención humana. Una fórmula equivocada puede ser un programa perfectamente compilable.

::: comprueba Antes de continuar
Debes poder distinguir índice y valor, texto C y ejecutable, compilación y ejecución, una prueba concreta y una garantía general. Explica también por qué cuatro tareas independientes pueden repartirse sin cambiar su significado.
:::

# Preparar el taller y aprender a leer un fallo {#ch:taller}

## Una carpeta es parte del experimento

Antes de aprender instrucciones complejas, necesitamos un taller reproducible. Un resultado sin código, configuración y datos identificables es difícil de estudiar. La organización mínima separa fuente, pruebas, ejemplos e informes. No necesitas una plataforma empresarial para conseguirlo.

```text
lumbre-curso/
    lumbre/          # Compilador y representación intermedia
    examples/        # Programas que usan el compilador
    tests/           # Comprobaciones automáticas
    kernels/         # Laboratorios de bajo nivel
    reports/         # Resultados de ejecuciones
```

El archivo fuente es la receta; una prueba es una pregunta que debe responder; un informe registra lo que realmente ocurrió. Los nombres tienen una función: permiten que otra persona ejecute la misma tarea sin adivinar cuál de diez archivos llamados `final_nuevo.py` era el importante.

## Terminal sin conocimientos previos

Una terminal recibe órdenes de texto. La carpeta de trabajo determina cómo se interpretan rutas relativas. `pwd` muestra dónde estás en Linux; `ls` enumera archivos; `cd` cambia de carpeta. Una ruta absoluta empieza desde la raíz del sistema; una relativa se interpreta desde la carpeta actual.

```bash
pwd
ls
cd lumbre-curso
python3 --version
cc --version
```

No copies los símbolos de un prompt como si fueran parte de la orden. En el libro los bloques contienen únicamente lo que debes escribir, salvo que se indique explícitamente una salida. Una línea con `#` en un script de shell suele ser un comentario. Una ruta con espacios debe ir entre comillas.

Para Windows se recomienda realizar los laboratorios POSIX en una distribución Linux dentro de WSL2 o en una máquina Linux separada. Esta recomendación responde a que el runtime de esta edición genera bibliotecas `.so` y utiliza opciones de compilación POSIX. El código Python no transforma automáticamente ese runtime en uno nativo de Windows. Un puerto nativo tendría que gestionar DLL, compilador, ABI y carga dinámica.

La documentación de NVIDIA diferencia el controlador Windows que ofrece CUDA a WSL y los paquetes de desarrollo dentro de la distribución. No debes instalar por inercia un controlador de pantalla Linux dentro de WSL siguiendo instrucciones destinadas a un Linux instalado directamente. Confirma siempre las instrucciones oficiales para la combinación usada. [@cuda-wsl]

## Entorno Python aislado

Un entorno virtual separa dependencias de este proyecto de las de otros. Desde la carpeta del paquete:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=. python -m pytest -q
```

`python -m pip` asegura que se utiliza el instalador asociado al intérprete seleccionado. `PYTHONPATH=.` hace visible el paquete de la carpeta actual en estos ejemplos. La versión de NumPy registrada en las pruebas de esta edición es 2.3.5 y el intérprete es Python 3.13.5. No se deduce que versiones distintas fallen; significa que esa fue la combinación comprobada.

Para un entorno compartido o de producción conviene fijar versiones y hashes mediante un sistema de bloqueo de dependencias. El libro no convierte «instala siempre la versión más reciente» en un procedimiento reproducible. El archivo de requisitos del paquete da una base; el registro de ejecución conserva las versiones observadas.

## Funciones: una tarea con entradas y salida

Una función agrupa instrucciones y recibe datos. Este ejemplo devuelve una lista nueva sin modificar la original:

```python
def desplazar(valores: list[float], delta: float) -> list[float]:
    return [x + delta for x in valores]

assert desplazar([1.0, 4.0], 2.0) == [3.0, 6.0]
```

Las anotaciones `list[float]` describen el tipo esperado. Por sí solas no hacen que Python compruebe todos los valores en ejecución. En la API del compilador escribiremos validaciones explícitas para formas, tipos, dimensiones y parámetros obligatorios.

Una función que cambia un objeto existente tiene un **efecto**. Una que produce un valor sin alterar estado visible puede tratarse como una función pura bajo su contrato. Esta distinción reaparecerá cuando intentemos reordenar operaciones. Reordenar dos sumas puras es diferente de reordenar una lectura y una escritura sobre la misma memoria.

## Excepciones y códigos de salida

Un error útil debe decir qué contrato se ha violado. «No funciona» no identifica si falta un compilador, si las dimensiones son incompatibles o si una comparación numérica excede la tolerancia. En Python, una excepción interrumpe el flujo normal y conserva una traza de llamadas.

```python
def longitud_valida(n: int) -> int:
    if not isinstance(n, int):
        raise TypeError("la longitud debe ser entera")
    if n < 0:
        raise ValueError("la longitud no puede ser negativa")
    return n
```

Una traza se lee desde el error final hacia la llamada que lo provocó. No es necesario comprender toda una biblioteca para reparar una forma incompatible: primero identifica el objeto y sus dimensiones. El compilador no debe ocultar la orden fallida del compilador C ni reemplazar todos los errores por una salida cero que parece correcta.

## La primera prueba automatizada

Un test contiene preparación, ejecución y comparación. En `pytest` una función cuyo nombre comienza por `test_` puede ser descubierta automáticamente. No necesita imprimir «correcto» para pasar; termina sin que falle una aserción.

```python
def desplazar(xs, d):
    return [x + d for x in xs]

def test_desplazamiento():
    entrada = [10, 12, 15, 9]
    salida = desplazar(entrada, 2)
    assert salida == [12, 14, 17, 11]
    assert entrada == [10, 12, 15, 9]
```

La segunda aserción comprueba que la entrada no se ha modificado. Es una propiedad diferente del resultado. Un algoritmo podría devolver la salida correcta y destruir por accidente unos datos que otra parte del programa necesita.

Para coma flotante usaremos comparaciones con tolerancia. No todas las diferencias son errores, pero tampoco toda diferencia pequeña es aceptable. Una tolerancia debe relacionarse con el tipo, el número de operaciones y la escala de los datos. El capítulo numérico prepara esa discusión.

## Tres clases de prueba

Una prueba unitaria examina una pieza pequeña, por ejemplo el cálculo de strides. Una prueba de integración conecta piezas, como generar C, compilarlo, cargarlo y ejecutar una matriz. Una prueba de extremo a extremo verifica una tarea completa, como entrenar un modelo y reanudar su checkpoint.

Ninguna sustituye a las otras. Si únicamente probamos el entrenamiento completo, localizar una transposición errónea puede ser muy costoso. Si únicamente comprobamos sumas aisladas, podemos no detectar que el plan de memoria reutiliza un buffer antes de su último uso.

::: ejemplo Una prueba que no basta
Para comprobar una multiplicación de matrices, llenar todo con unos es cómodo: el resultado de cada casilla suele ser el tamaño de la reducción. Sin embargo, ciertas transposiciones incorrectas siguen dando lo mismo. Añade matrices no cuadradas y valores distintos por fila y columna. El objetivo no es acumular ejemplos fáciles, sino hacer visibles decisiones diferentes.
:::

## Leer el código generado

Lumbre guarda el código fuente generado en su caché. El informe del programa incluye la ruta. Es importante abrirlo: un compilador educativo no debería convertirse en una caja negra justo cuando empieza a funcionar.

```python
from lumbre import param, Program

x = param("x", (4,))
y = x + 2.0
with Program([y]) as programa:
    print(programa.report())
```

Esta construcción compila un programa pero todavía no ha inicializado `x`. Intentar ejecutarlo sin datos produce un error explícito. Inicializar un parámetro, compilar un grafo y ejecutar sus kernels son acciones distintas.

## Seguridad básica del JIT

Un compilador JIT convierte datos estructurados en código ejecutable. Si acepta nombres, opciones o texto arbitrario de usuarios no confiables, aparece una superficie de ataque. En los ejercicios locales no ejecutamos fuentes descargadas sin inspección ni utilizamos `shell=True` para concatenar órdenes.

La práctica segura es pasar la orden al sistema como una lista de argumentos, limitar rutas, comprobar retornos, fijar un tiempo máximo y mantener una caché privada al usuario. Estas medidas no convierten el JIT en una sandbox. Ejecutar código nativo no confiable requiere aislamiento del sistema operativo y una política específica.

## Git como cuaderno de decisiones

Un commit conserva una versión identificable. Antes de optimizar, guarda una versión correcta. Después de cambiar un pase, registra qué propiedad intentas mejorar. Si un resultado deja de coincidir, podrás comparar dos versiones pequeñas en lugar de reconstruir de memoria una semana de cambios.

```bash
git init
git add lumbre tests examples
git commit -m "Compilador de referencia y primeras pruebas"
```

No incluyas claves, credenciales, datasets privados ni enormes cachés binarios por defecto. El hash de un commit identifica código, pero no identifica automáticamente el dataset, el driver ni las opciones de compilación. El pasaporte experimental del final del libro une esas piezas.

::: comprueba Antes de continuar
Localiza la carpeta actual, activa el entorno, ejecuta una prueba y abre un archivo generado. Explica la diferencia entre un fallo de instalación y un fallo numérico. Ninguna de esas tareas requiere todavía saber qué es un tensor core.
:::
