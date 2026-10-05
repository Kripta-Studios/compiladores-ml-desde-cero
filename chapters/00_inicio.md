#* Empieza aquí

Imagina que una persona te entrega cien temperaturas y te pide sumar dos grados a cada una. Puedes resolverlo con una calculadora, número a número. Puedes escribir un programa que repita esa operación. Y puedes escribir otro programa que construya automáticamente el programa anterior, escogiendo cómo recorrer los datos y cómo aprovechar la máquina. Ese último trabajo es el que aprenderás a hacer en este libro.

No empezaremos por una GPU, una letra griega ni una arquitectura de red neuronal. Primero habrá una lista de cuatro números, un bucle y una pregunta muy concreta: ¿qué debe producir? Después construiremos un lenguaje pequeño para describir el cálculo. Cuando ese lenguaje funcione, separaremos lo que se calcula de cómo se ejecuta. La separación permite conservar una misma intención y generar código distinto para CPU, CUDA o HIP.

El proyecto que conecta todo el curso se llama **Lumbre**. No es un nombre nuevo para tinygrad ni un envoltorio alrededor de PyTorch. Es un compilador didáctico original, escrito en Python, que genera C y dispone de rutas CUDA/HIP. Python organiza los grafos y llama al código compilado. Las operaciones de entrenamiento de la demostración se ejecutan en el C generado. NumPy proporciona almacenamiento, generación de datos y referencias de comprobación; no calcula los gradientes del modelo.

::: idea Una regla de lectura
Antes de memorizar una definición, debes poder predecir un ejemplo pequeño. Antes de optimizar un programa, debes poder explicar por qué produce el resultado correcto. Antes de publicar una mejora, debes poder decir exactamente qué has medido y qué has dejado fuera.
:::

Las cajas azules dan nombre a una idea; las naranjas traducen fórmulas; las verdes desarrollan ejemplos; las violetas plantean tareas; las rojas señalan errores; las de color petróleo permiten comprobar comprensión. Un recuadro no sustituye el texto que lo rodea. Los símbolos aparecen después de la situación que intentan describir.

Cada bloque tiene tres niveles. La primera lectura construye intuición. La segunda convierte esa intuición en código o en un argumento preciso. La tercera analiza las condiciones bajo las cuales la técnica deja de funcionar. No necesitas completar los tres niveles en una sola sesión. Volver a una operación con más herramientas es parte del aprendizaje, no un fallo de la primera explicación.

## Qué significa completar el curso

El primer hito consiste en generar C para programas elemento a elemento. El segundo incorpora bucles, índices, movimientos y reducciones. El tercero compara versiones lentas y rápidas con un contrato numérico común. El cuarto añade un runtime GPU y kernels que respetan la sincronización del dispositivo. El quinto construye y entrena un decoder Transformer. El sexto exige una extensión propia defendida con pruebas, medidas y limitaciones.

La frase del temario «cualquier modelo» se interpreta de forma técnicamente precisa: cualquier grafo expresable con las operaciones, tipos, formas y efectos que soporte el compilador, sujeto a memoria y recursos. No significa que unas pocas semanas produzcan compatibilidad automática con toda operación de cualquier framework, cualquier control de flujo y cualquier acelerador. Aprender a identificar una operación no soportada forma parte de construir un compilador correcto.

La referencia a código competitivo y a entrenar LLM de última generación se mantiene como **objetivo de ingeniería y de escalado**, no como un resultado garantizado por terminar una lista de ejercicios. El libro desarrolla GEMM, atención eficiente, precisión mixta, paralelismo, memoria de entrenamiento y la separación entre preentrenamiento y ajuste. Incluye una ruta de entrenamiento real de un modelo pequeño y un diseño de cómo llevar el mismo compilador hacia modelos mayores. No llama «SOTA» a una red de juguete ni atribuye rendimiento de H100 a un portátil.

::: cuidado Lo que se ha ejecutado y lo que no
La entrega conserva la validación original en CPU Linux y el entrenamiento de 75.584 parámetros. La revisión local del 1 de octubre añade ejecución CUDA bajo WSL2 en una RTX 5070 Ti Laptop: ejemplos numéricos, microbenchmarks, entrenamiento y WMMA. HIP sigue pendiente de un dispositivo compatible. El registro del apéndice y `code/reports/validation_20261001/` distinguen los resultados originales de los nuevos.
:::

## Un semestre con punto de partida honesto

El programa de semanas 1–10 presupone trabajo práctico continuado. Como aquí se empieza desde cero, se añade un bloque preparatorio sobre Python, C, terminal, matrices y números de coma flotante. Una planificación orientativa reserva entre veinte y treinta y cinco horas para esa preparación, según la experiencia real del estudiante. Es una estimación docente, no una medida experimental ni una promesa de aprendizaje idéntico para todas las personas.

Durante el semestre, una organización posible combina explicación, lectura de código, laboratorio y una sesión de depuración. El progreso se decide por competencias verificables, no por haber pasado de página. Si todavía no puedes calcular la dirección de un elemento de una matriz de dos filas, conviene repetir el capítulo de índices antes de estudiar una instrucción matricial GPU.

| Etapa | Pregunta de control | Entregable |
|---|---|---|
| Preparación | ¿Puedo ejecutar y comprobar un bucle? | Programa C y prueba manual |
| Semanas 1–2 | ¿Puedo transformar un grafo sin alterar su contrato? | UOps, simplificador, generador C |
| Semanas 3–4 | ¿Puedo convertir tensores en recorridos? | Bucles, movimientos, reducciones y rangeify |
| Semanas 5–6 | ¿Puedo diferenciar, entrenar y mejorar CPU? | Fisión, fusión, autodiff, MNIST y kernels rápidos |
| Semanas 7–8 | ¿Puedo ejecutar sin carreras en una GPU? | Runtime, mapeo de ejes y kernels |
| Semanas 9–10 | ¿Puedo construir y entrenar un decoder? | Formas simbólicas, modelo, optimizador y checkpoint |
| Semanas 11 en adelante | ¿Puedo defender una extensión original? | Proyecto, informe y reproducción |

## Cómo utilizar el código entregado

El libro es autocontenido para leer: el listado completo del proyecto aparece al final. El paquete complementario evita transcribir miles de caracteres. Cada laboratorio indica una carpeta, una orden y una condición de éxito. Los fragmentos cortos del texto pueden ser ejemplos aislados; cuando no constituyen un programa completo se indica. El código del paquete es la referencia ejecutable de esta edición.

No copies todo el compilador el primer día intentando comprenderlo de golpe. Empieza con el primer bucle C. Después utiliza el proyecto para observar la salida del generador y relacionar cada fragmento con un capítulo. Un buen ejercicio consiste en predecir cuántos kernels tendrá una expresión antes de consultar el informe.

## Fuentes, fechas y límites de esta investigación

El corte es el **1 de octubre de 2026**. Las páginas de proyectos pueden cambiar posteriormente. El recorrido central de tinygrad se ancla al commit `c3aec477b99d9bb87c54d91897cf60acd3f17441`, fechado el 30 de septiembre de 2026. Para otras páginas se identifica el recurso consultado y su estado visible, sin inventar un hash de código que no se haya recuperado. La bibliografía distingue artículos científicos, documentación y repositorios. [@tinygrad-pin]

Los artículos de arXiv se leen como trabajos con autores, métodos y un ámbito de evaluación; que estén en arXiv no demuestra revisión por pares. Una tabla de un proveedor es evidencia de lo que declara el proveedor. Una ejecución local es evidencia de una configuración concreta. Un razonamiento matemático tiene hipótesis que deben mantenerse al traducirlo a coma flotante.

La estética y el orden didáctico toman como referencia los apuntes de Teoría de Autómatas facilitados por el estudiante. Su contenido no se utiliza como fuente sobre GPU o aprendizaje automático. El temario de compiladores es el proporcionado en la solicitud; el desarrollo, los ejercicios y los proyectos son propios.
