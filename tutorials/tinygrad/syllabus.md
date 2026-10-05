# Leer el syllabus a través de un compilador existente {#tiny:syllabus}

## Observar después de formular una hipótesis

Compilers for Machine Learning pide construir un compilador desde cero, con lenguaje libre y sin código inicial. tinygrad sirve como objeto de estudio y comparación. Consultar cómo resuelve un problema permite contrastar decisiones; copiar sus módulos no cumple por sí solo la construcción propia.

Esta ampliación del 5 de octubre de 2026 conserva el commit fijado y las prácticas del tutorial. Los experimentos siguientes son propuestas de lectura y extensión, no nuevas ejecuciones ni descripción de otra versión. Antes de inspeccionar el código, escribe qué esperas y qué observación podría refutarlo.

## Tres vocabularios relacionados

| Syllabus | Pregunta para tu compilador | Contraste en el tutorial |
|---|---|---|
| UOp y hash-consing | ¿Qué hace que dos nodos sean el mismo? | Expresión repetida |
| Movimientos y rangeify | ¿Cómo se convierte una forma en índices? | Reshape, traspuesta y broadcasting |
| STAGE y fusión | ¿Cuándo guardar un intermedio? | Materialización y plan |
| Autodiff puro | ¿Dónde se acumulan contribuciones? | Pérdida y gradientes analíticos |
| Runtime y kernels | ¿Qué se compila y reutiliza? | Realización y entradas nuevas |

Syllabus, Lumbre y tinygrad no comparten necesariamente operaciones, campos o serialización. Un nombre parecido no demuestra equivalencia. El libro principal desarrolla el contrato conceptual y marca sus convenciones docentes. Aquí, la comparación externa se refiere a la revisión fijada en las fuentes del cuaderno.

## Semana 1: repetición y especialización

Estudia `z=(x+2)*(x+2)` con x no constante. Dibuja una suma compartida que alimenta dos entradas de la multiplicación. Después distingue repetición en el código del usuario, identidad de nodos y repetición de instrucciones finales: no tienen que coincidir uno a uno.

El compilador propio necesita una prueba de hash-consing independiente de tinygrad. Construye literales bool e i32 con igual valor aparente y comprueba que no colisionan. Define también si el cero FP32 con signo forma parte del contrato. No se requiere una API interna nueva: el estudiante elige cómo observar su representación.

**Ejercicio resuelto.** Para x=3, la suma vale 5 y z vale 25. Sustituir el programa por 25 solo es válido si x es constante del programa, no si 3 fue un dato de una ejecución. Una segunda llamada con x=4 debe producir 36. La pareja detecta especialización indebida a datos.

## Semanas 2–4: forma, vista y almacenamiento

Una matriz con filas `[1,2,3]` y `[4,5,6]` tiene traspuesta con filas `[1,4]`, `[2,5]` y `[3,6]`. Un reshape directo a `(3,2)` produce `[1,2]`, `[3,4]` y `[5,6]`. Las formas finales coinciden, pero los valores por posición difieren.

Escribe los dos mapas de índices en tu compilador y compáralos con un evaluador independiente. En tinygrad, usa las operaciones ya presentadas en el tutorial y observa después el programa ejecutado. Una vista puede materializarse por necesidades de un consumidor; una sola expresión no prueba que «trasponer nunca copia» ni que «reshape siempre genera un kernel».

## Semana 4: GEMM rectangular

Usa A de forma `(2,3)` con filas `[1,2,3]` y `[4,5,6]`, y B de forma `(3,2)` con filas `[7,8]`, `[9,10]` y `[11,12]`. AB es `[[58,64],[139,154]]`. Compara producto matricial con expansión, multiplicación y reducción.

El fragmento `uop v1` del syllabus usa matrices cuadradas y no detalla todos los ejes. En la convención docente de reducción inicial, lleva A a `(K,M,1)` mediante permutación y reshape, y B a `(K,1,N)`. El ejemplo rectangular obliga a justificar esa permutación. Nombra las coordenadas antes de traducirlas a una API.

## Semana 5: compartir forward y acumular backward

Sea `s=sum(x*x)` con salidas `s+1` y `s+2`. Un plan puede calcular s una vez o recomponerlo para cada consumidor. Registra el plan antes y después de forzar un intermedio y cuenta recorridos de x además de kernels.

La realización explícita es una intervención experimental en tinygrad. No la identifiques automáticamente con `STAGE` del syllabus ni supongas que produce el mismo plan en cualquier revisión. En el compilador propio, define productor, consumidores y vida del stage.

Para `L=sum((x+b)^2)`, con x de forma `(2,3)` y b de forma `(3,)`, el gradiente de b acumula filas. Si x contiene `[1,2,3]` y `[4,5,6]` y b es cero, L vale 91; el gradiente de b es `[10,14,18]` y el de x es `[[2,4,6],[8,10,12]]`. Esta cuenta proporciona un oráculo independiente de los frameworks.

## La frontera de autodiff

El syllabus diferencia grafos puros con formas y excluye `STORE`, `CALL`, `AFTER`, `RANGE` y `END`. Conserva la representación diferenciable hasta construir los gradientes; después planifica y baja ambos grafos. No intentes derivar escrituras como si fueran sumas puras.

Las prácticas de gradientes del tutorial sirven para comparar funciones y resultados. No exigen que ambos compiladores diferencien en el mismo momento interno ni utilicen las mismas representaciones. El contrato común es la matemática y su convención de derivación.

# De MNIST al modelo de lenguaje: plan de experimentos {#tiny:entrenamiento-syllabus}

## MNIST en la semana 5

La regresión de este cuaderno enseña actualización y guardado de pesos; no equivale a completar el hito MNIST. Para esa ampliación, especifica imágenes, etiquetas, normalización, particiones y arquitectura antes de entrenar.

Un modelo mínimo usa entradas `(B,784)`, pesos `(784,10)` y sesgo `(10,)`. La entropía cruzada media exige GEMM, broadcasting, operaciones elementales, selección de clases y reducción. Implementa la misma función en el compilador y en su referencia. Una exactitud elevada no compensa usar normalizaciones distintas en ambos gradientes.

Primero calcula logits y pérdida de un lote artificial con una referencia directa. Comprueba gradientes en un modelo más pequeño. Después intenta sobreajustar unas pocas muestras y, finalmente, entrena con evaluación separada. Guarda configuración e informe aunque la calidad inicial sea baja. No hay resultados MNIST nuevos implícitos en este capítulo.

## Comparar una actualización antes que una trayectoria

La actualización necesita el gradiente correspondiente a los pesos anteriores. Si el runtime sobrescribe un peso mientras otro cálculo lo necesita, el paso deja de representar la pérdida definida. La planificación de efectos y memoria debe preservar esa dependencia.

Compara pesos iniciales, lote, pérdida, gradientes y pesos tras un paso. Cuando coincidan, compara trayectorias. Cambiar simultáneamente optimizador, inicialización y precisión dificulta localizar el primer desacuerdo.

## Semanas 6–8: dispositivos y rendimiento

Repite el mismo cálculo en CPU y GPU con entradas y contrato iguales. Las prácticas existentes ofrecen referencias para ambos; sus tiempos pertenecen a sus configuraciones registradas. Al investigar upcasting o kernels nuevos, mide preparación y ejecución caliente por separado.

El objetivo CPU aparece antes de GPU. Conserva una referencia CPU comprensible para revisar el backend nuevo. Las comparaciones con PyTorch y la validación HIP requieren campañas propias. Que tinygrad tenga un backend no demuestra que tu compilador implemente sus operaciones.

## Semanas 9–10: causalidad y formas simbólicas

Antes del LLM completo, modifica un token futuro de la atención pequeña del tutorial y comprueba que no altera salidas anteriores. Compara además con la fórmula densa. Igualdad en un ejemplo y causalidad comprueban propiedades diferentes.

Una dimensión simbólica necesita restricciones. Si la dimensión del modelo debe ser divisible entre el número de cabezas, demuestra o comprueba esa condición antes del kernel. Cuando cambia la longitud, declara si recompilas, reutilizas una variante general o rechazas el caso.

Repetir mediante JIT una forma fija no demuestra soporte simbólico general. Los ejercicios existentes cambian datos manteniendo compatibilidad con la captura. Ampliar formas exige revisar el contrato de la revisión fijada y el del runtime propio.

## Guardar pesos y reanudar entrenamiento

El NPZ de la regresión guarda pesos. Una reanudación con estado necesita también momentos del optimizador cuando existan, contador y estados necesarios de aleatoriedad y datos. Compara diez pasos continuos con seis pasos, guardado, carga y cuatro pasos más.

Define determinismo y tolerancia según el dispositivo. Una pérdida final similar no acredita restaurar todo el estado: compara pesos y estados que determinan el siguiente paso. El libro principal desarrolla una prueba de reanudación que puede servir como modelo experimental.

## Proyecto desde la semana 11

Una pregunta acotada sería: «¿Materializar esta reducción compartida mejora el tiempo del paso sin cambiar sus gradientes?». Fija formas, dos planes, referencia y presupuesto. La intervención debe poder desactivarse para compararla.

Otra opción estudia cómo una dimensión simbólica modifica la caché. Diseña una secuencia de tamaños y registra compilaciones, tiempo frío, tiempo caliente y memoria de artefactos. Mantén casos que perjudiquen a cada política.

| Entrega | Contenido |
|---|---|
| Especificación | Función, tipos, formas, dominio y efectos |
| Corrección | Casos manuales, bordes y referencia independiente |
| Integración | Paso con valores y gradientes comprobados |
| Rendimiento | Frontera temporal, repeticiones y casos desfavorables |
| Reproducción | Revisión, entorno, órdenes, datos y hashes |

El resultado académico es una decisión del compilador defendida con evidencia. La meta de entrenar LLM modernos orienta la integración; la escala y calidad alcanzadas se acreditan con el experimento realmente ejecutado.
