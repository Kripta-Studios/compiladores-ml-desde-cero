# Tensor cores: una operación colectiva, no una multiplicación mágica {#ch:tensorcores}

::: idea La pregunta de este capítulo
Ya sabemos repartir un producto de matrices entre hilos. Ahora imaginaremos una calculadora que no acepta un número cada vez, sino dos pequeños mosaicos completos. Aprender a alimentarla y recoger su resultado es tan importante como invocar la instrucción.
:::

## De una celda a un mosaico

En el producto de matrices del capítulo de CPU, una celda de salida se obtenía recorriendo una fila de A y una columna de B. Podíamos mantener varias celdas en acumuladores para reutilizar los datos. Un acelerador matricial lleva esa idea al hardware: ejecuta una operación de la forma $D=AB+C$ sobre pequeños bloques, con formatos y disposiciones permitidos por su arquitectura.

Imagina una imprenta que imprime dieciséis tarjetas simultáneamente. No basta con entregarle una tarjeta y pedir que sea dieciséis veces más rápida. Hay que colocar las tarjetas en las posiciones correctas, cargar la tinta y esperar a que el mecanismo termine. Si preparar la bandeja cuesta más que imprimirla, una tirada diminuta no aprovecha la máquina.

El paralelismo matricial tampoco elimina el tráfico de memoria, las conversiones de tipos, las máscaras ni los epílogos. Un GEMM rápido organiza todo ese trabajo alrededor de los bloques que puede consumir la unidad especializada.

::: definicion Tres niveles que conviene distinguir
Una operación matemática define qué resultado queremos. Una instrucción matricial define un bloque y una cooperación concreta del hardware. Un kernel define cómo muchas instrucciones, cargas y escrituras producen una operación completa de cualquier tamaño admitido.
:::

## Fragmentos: cada participante guarda solo parte

La interfaz WMMA de CUDA permite describir fragmentos de matrices y ejecutar una multiplicación cooperativa. No hay que interpretar un fragmento como un array ordinario cuya posición cero corresponde siempre a la primera celda matemática. La distribución interna pertenece al contrato de la interfaz y puede depender de la arquitectura. La programación correcta utiliza sus operaciones de carga, multiplicación y almacenamiento. [@cuda;@ptx]

Nuestro archivo `kernels/wmma_demo.cu` usa exactamente un warp y un bloque $16\times16\times16$: entradas FP16, acumulación FP32 y salida FP32. No es todavía un GEMM general. Es una lupa que permite verificar el primer uso de la operación colectiva.

El núcleo relevante es el siguiente; el archivo completo añade asignación, transferencia, comprobación de errores y referencia numérica.

```cuda
using namespace nvcuda;
wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a;
wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b;
wmma::fragment<wmma::accumulator,16,16,16,float> acc;
wmma::fill_fragment(acc,0.0f);
wmma::load_matrix_sync(a,A,16);
wmma::load_matrix_sync(b,B,16);
wmma::mma_sync(acc,a,b,acc);
wmma::store_matrix_sync(C,acc,16,wmma::mem_row_major);
```

Lee `16` en la carga como dimensión principal de esta disposición, no como número de hilos. Todas las operaciones del fragmento se ejecutan cooperativamente. No se coloca la multiplicación dentro de un `if` que solo toma una parte del warp.

La inicialización de los datos utiliza valores distintos y representables de manera sencilla. Una matriz de unos es útil como primera prueba, pero puede esconder una transposición: muchas permutaciones de unos siguen siendo unos. Los patrones por fila y columna revelan qué orientación se ha interpretado.

## Las cuatro preguntas antes de compilar

Primera: ¿qué GPU tenemos realmente? Segunda: ¿qué instrucciones y tipos soporta ese destino? Tercera: ¿el compilador instalado conoce el destino? Cuarta: ¿el controlador puede cargar el código producido?

Una marca comercial no resuelve esas preguntas. La tabla oficial distingue, por ejemplo, capacidades de cómputo diferentes entre Blackwell de centro de datos y GeForce RTX 50. Un kernel que depende de características específicas de `sm_100a` no se convierte en compatible con una GeForce por compartir el nombre Blackwell. [@cuda-gpus;@cutlass]

Para compilar la demostración se debe sustituir `sm_120` por el destino real si tu dispositivo es diferente y comprobar que el toolkit lo conoce. El comando siguiente usa el destino de la GPU validada, sin detectarlo automáticamente:

```bash
nvcc -O2 -std=c++17 -arch=sm_120 kernels/wmma_demo.cu -o wmma_demo
./wmma_demo
```

::: cuidado Estado de esta práctica
La revisión del 1 de octubre compiló y ejecutó este archivo en una RTX 5070 Ti Laptop con CUDA 12.9 y destino `sm_120`. El error absoluto máximo frente a su referencia fue cero; el registro está en `code/reports/validation_20261001/wmma_run.log`. Ajusta el destino al dispositivo real del laboratorio. Este resultado comprueba la operación, sin demostrar rendimiento competitivo.
:::

## De la demostración a un GEMM completo

Para multiplicar matrices mayores, divide la salida en tiles. Cada bloque de salida recorre K por fragmentos. Acumula contribuciones y escribe el resultado al terminar. Esta descripción se parece al GEMM de memoria compartida porque resuelve la misma necesidad de reutilización, pero cambia la unidad de cómputo y su contrato de layout.

Considera M=20, N=18, K=17. Un tamaño de tile de 16 deja bordes en las tres dimensiones. Si el hardware exige un fragmento completo, no podemos leer fuera de los arrays para fingir que esos elementos existen. Una solución didáctica consiste en preparar bloques con ceros para las posiciones no válidas y almacenar únicamente las celdas de salida válidas.

El relleno en K debe aportar cero a la suma. El relleno en M y N no debe producir escrituras fuera de rango. Son dos obligaciones diferentes: una protege el valor, la otra protege la memoria.

Una ruta más sofisticada puede separar los bloques interiores de los bordes y elegir otro kernel para tamaños pequeños. El compilador conserva una alternativa genérica para los casos que no cumplen alineamiento, tipos o tamaños de la ruta rápida. Eso es una especialización con guardas, no una excusa para rechazar silenciosamente entradas.

## Carga asíncrona: iniciar no significa terminar

Imagina dos bandejas. Mientras se calcula con la primera, se prepara la segunda. En la siguiente iteración intercambiamos sus papeles. El beneficio potencial es solapar traslado y cómputo; el peligro es reutilizar una bandeja que todavía está siendo leída.

En una implementación real hacen falta estados para cada fase: espacio disponible, copia iniciada, copia completada, consumidores terminados. Una barrera genérica puesta al azar no sustituye el protocolo específico de la operación asíncrona. Las instrucciones modernas tienen ámbitos y mecanismos de finalización diferentes; el renderer debe elegirlos según el destino. [@ptx]

El siguiente dibujo es un modelo temporal propio. Cada fila es una bandeja, no una garantía de que una GPU concreta ejecute exactamente estos ciclos.

```tikz
\begin{center}
\begin{tikzpicture}[x=1.25cm,y=1cm]
\node[anchor=east] at (0,1) {Bandeja A};
\node[anchor=east] at (0,0) {Bandeja B};
\draw[fill=azul!10] (0,0.7) rectangle (2,1.3);
\node at (1,1) {cargar 0};
\draw[fill=verde!10] (2,0.7) rectangle (5,1.3);
\node at (3.5,1) {calcular 0};
\draw[fill=azul!10] (2,-.3) rectangle (4,.3);
\node at (3,0) {cargar 1};
\draw[fill=verde!10] (5,-.3) rectangle (8,.3);
\node at (6.5,0) {calcular 1};
\draw[flow] (0,-.8) -- (8.2,-.8) node[right] {tiempo};
\draw[dashed] (5,-.45) -- (5,1.5);
\node[font=\small] at (5,1.8) {intercambio tras completar};
\end{tikzpicture}
\end{center}
```

Antes de añadir doble buffering, mide un kernel de una sola etapa. Después compara registros, memoria compartida y tiempo. Doblar el número de buffers puede reducir la ocupación o complicar el epílogo. Solapar más trabajo no implica automáticamente terminar antes.

## El epílogo también tiene arquitectura

La salida del acelerador está distribuida entre participantes y puede necesitar una reorganización antes de escribir. Además, queremos quizá sumar un sesgo, escalar, aplicar una activación o cuantizar. Ese tramo es el epílogo.

Fusionarlo evita escribir una matriz intermedia y volver a leerla. Sin embargo, una reducción por filas en el epílogo exige comunicación entre quienes poseen diferentes columnas. Una función escalar por elemento y una normalización de toda la fila no requieren el mismo mecanismo.

Un buen diseño de IR conserva la información suficiente para decidir si la fusión es legal y rentable. No basta con reconocer el patrón textual `matmul + bias`; también hay que conocer sus formas, el eje del sesgo, el tipo de acumulación y los consumidores posteriores.

## CuTe, CUTLASS y cuTile no son tres nombres de lo mismo

CUTLASS ofrece componentes para construir operaciones de álgebra lineal. CuTe representa layouts y particiones de datos e hilos; su DSL permite expresar esas ideas desde Python. cuTile es otro proyecto y otra interfaz. Que dos herramientas hablen de tiles no hace que compartan sintaxis, modelo de memoria o nivel de control. [@cutlass;@cutile]

La lección para Lumbre es arquitectónica: separar la decisión matemática, la disposición de datos y el mecanismo de cómputo. Una transformación de layout debería poder justificarse sin mezclarla con la lógica del optimizador de entrenamiento.

El README de CUTLASS consultado presenta la edición 4.8.0 de septiembre de 2026 y soporte preliminar de Rubin. Advierte además que ejecutar esos kernels exige un controlador futuro respecto de la entrega preliminar citada. En este libro eso es una observación de código y documentación, no una práctica disponible universalmente el día del corte. [@cutlass]

## Una prueba oral que detecta comprensión

::: practica ¿Por qué una sola instrucción puede necesitar treinta y dos hilos?
Explica por qué no podemos lanzar un hilo, ejecutar `mma_sync` y esperar una matriz completa. Después explica por qué treinta y dos hilos no bastan para garantizar que todo sea correcto.
:::

La primera respuesta es que la interfaz representa una operación cooperativa: los fragmentos están repartidos y las llamadas tienen requisitos colectivos. La segunda es que siguen importando la participación uniforme, el tipo y layout de cada fragmento, las direcciones, el alineamiento, las dimensiones principales, el destino compilado y el almacenamiento de la salida. Contar participantes solo comprueba una parte del contrato.

# AMD: portar el significado antes de perseguir instrucciones {#ch:amd}

## Una traducción que parece demasiado fácil

Muchos ejemplos de suma de vectores cambian `cudaMalloc` por `hipMalloc` y conservan casi todo el kernel. Eso enseña una correspondencia útil entre interfaces. Pero un kernel que dependía de una disposición concreta de registros, un tamaño de warp o una instrucción matricial no se vuelve portátil mediante una sustitución de nombres.

Pensemos en una clase que organiza a sus estudiantes en grupos. Cambiar el nombre de la escuela no nos dice cuántas personas hay en cada grupo ni cómo se reparten los ejercicios. Si el algoritmo suponía que el estudiante número 31 era siempre el último, necesitamos revisar esa suposición.

HIP documenta propiedades y extensiones para el modelo de ejecución. La ruta correcta consulta y especializa las características del destino; no convierte una suposición de NVIDIA en una constante universal. [@hip]

## Tres capas del port

La capa semántica define `sum`, `matmul`, `where`, los tipos y los ejes. No debería cambiar porque cambiemos de fabricante. La capa de planificación decide tiles, grupos, memoria local y sincronización. Puede necesitar decisiones diferentes. La capa de emisión produce las instrucciones e interfaces del destino.

En Lumbre, el backend HIP básico sigue la misma matemática y genera kernels escalares por elemento o un GEMM bloqueado de memoria compartida. No contiene una implementación optimizada de MFMA. Tener backend HIP, por tanto, significa que existe una ruta de emisión y runtime; no que hayamos igualado las bibliotecas especializadas de AMD.

::: traduccion Portabilidad semántica frente a portabilidad de rendimiento
La primera pregunta es «¿calcula lo mismo dentro del contrato numérico?». La segunda es «¿usa bien esta máquina?». Resolver la primera es obligatorio. La segunda suele exigir especialización y mediciones nuevas.
:::

## Un ejemplo de reducción dependiente del grupo

Supón que una reducción parcial usa intercambios entre lanes con distancias 16, 8, 4, 2 y 1. Ese patrón presupone una agrupación concreta. En otro tamaño de grupo, parte de los valores podría no participar o podríamos comunicar posiciones no válidas.

Una implementación portable empieza por una reducción de memoria compartida cuya corrección puede explicarse con el número de elementos activos. Después añade una versión especializada con operaciones de subgrupo para los destinos que cumplen sus guardas.

No basta con reemplazar 32 por una variable si el resto del código aún codifica máscaras de 32 bits, fragmentos fijos o layouts de almacenamiento. Las dependencias de una suposición pueden estar repartidas por varios módulos. Por eso las pruebas incluyen tamaños menores, mayores y no múltiplos del subgrupo.

## LDS y memoria compartida: parecido no significa idéntico coste

En el razonamiento del compilador podemos llamar memoria local del bloque al espacio cooperativo cercano a los hilos. En la documentación de AMD aparece LDS. La abstracción ayuda a expresar un tile reutilizable, pero el rendimiento depende de bancos, acceso, capacidad y recursos del destino.

Si cada lane accede a una columna de una matriz local con un stride desfavorable, puede aparecer contención. Una reorganización o un padding puede mejorar el patrón. Hay que revisar tanto quién escribe como quién lee: arreglar la carga inicial no garantiza que el consumo matricial quede bien distribuido.

Un experimento educativo cambia únicamente el layout local manteniendo las mismas operaciones aritméticas. Se registra error, tiempo, memoria reservada y recursos. Si cambian también el tamaño del tile, el número de hilos y la precisión, ya no podemos atribuir la mejora a una única causa.

## Dónde mirar en el ecosistema

El repositorio histórico Composable Kernel y el de hipBLASLt remiten al monorepo `ROCm/rocm-libraries`. AITER reúne implementaciones orientadas a operaciones de IA, mientras que el fork LLVM de ROCm pertenece a la infraestructura de compilación. Son piezas de niveles diferentes. [@ck;@hipblaslt;@rocm-libraries;@aiter;@rocm-llvm]

La lectura recomendada no es abrir todos los archivos. Escoge un GEMM y reconstruye su contrato de entrada, selección de implementación, parámetros de tile, epílogo y pruebas. Después sigue una configuración concreta hasta el código que ejecutaría. El producto de la lectura es un diagrama y una tabla de decisiones, no una colección de nombres.

| Pieza | Pregunta que ayuda a responder | Lo que no debemos inferir |
|---|---|---|
| HIP | ¿Cómo lanzo y sincronizo trabajo? | Que todos los kernels son óptimos. |
| hipBLASLt | ¿Cómo selecciono un GEMM especializado? | Que es un compilador general de modelos. |
| Composable Kernel | ¿Cómo se componen operaciones y layouts? | Que copiar una configuración sirve en cualquier GPU. |
| AITER | ¿Qué kernels de IA puedo estudiar o contrastar? | Que cada ruta soporta todos los modelos y tipos. |
| LLVM de ROCm | ¿Cómo se baja código hacia el destino? | Que un frontend de tensores viene incluido en nuestra aplicación. |

## Laboratorio de port con presupuesto acotado

Empieza por el ejemplo de elementwise de Lumbre. Conserva los mismos datos guardados en un archivo y la misma referencia CPU. Ejecuta la ruta HIP solo después de verificar instalación, dispositivo y compilador. Anota expresamente la arquitectura detectada y la versión de la cadena.

Después prueba el GEMM compartido con tamaños 31, 47 y 65. Esos números fuerzan bordes. Si falla solo el tamaño irregular, sospecha primero de máscaras o índices; si falla también el bloque exacto, revisa layout, dimensiones principales y barreras.

Para rendimiento, compara kernels con operandos residentes. No atribuyas a la multiplicación el coste de construir un contexto en la primera llamada. Tampoco escondas ese coste: preséntalo por separado porque puede importar en una aplicación de vida corta.

El siguiente paso es incorporar una instrucción matricial o una biblioteca como ruta externa declarada. Eso cambia la afirmación que podemos hacer: «Lumbre despacha a esta implementación» no equivale a «Lumbre genera sus instrucciones desde cero». Ambas opciones son útiles si se documentan con precisión.

::: comprueba Antes de pasar a otro acelerador
Debes poder señalar qué parte del compilador conserva la matemática, qué parte cambia el reparto del trabajo y qué parte depende de la API del fabricante. También debes poder explicar por qué un test numérico idéntico no garantiza un perfil de rendimiento idéntico.
:::

# Huawei Ascend y DeepSeek: separar cálculo, comunicación y compilación {#ch:ascend}

## La palabra kernel estaba escondiendo dos cosas

Un kernel de cálculo es un programa que realiza una operación, como un producto matricial, en un acelerador. El kernel de Linux es el núcleo del sistema operativo. Los controladores pueden vivir parcialmente dentro de este último y permitir que el primero se ejecute, pero no son el mismo programa.

Por eso «el kernel de Huawei» resulta insuficiente como descripción técnica. En este curso distinguimos la familia de aceleradores Ascend, su software de programación, las bibliotecas de kernels de DeepSeek para Ascend y las capas de sistema que gestionan dispositivos.

La precisión en los nombres evita un error de diseño: intentar encontrar un algoritmo de GEMM en un controlador de memoria, o esperar que una biblioteca de comunicación implemente la multiplicación del modelo.

## Una fábrica con estaciones especializadas

Imagina una fábrica con una estación para transportar cajas, otra para operaciones vectoriales y otra para multiplicar bloques de matrices. El desafío no es mantener ocupada una sola estación, sino organizar el flujo completo sin que una consuma una caja antes de que la anterior la haya preparado.

La documentación y ejemplos de Ascend permiten estudiar esa división de responsabilidades y sus espacios de memoria. Aquí usamos la idea de cálculo matricial, cálculo vectorial y transferencia como modelo pedagógico. Las capacidades, nombres de instrucciones y límites concretos deben obtenerse del chip y de la versión de CANN utilizados. [@ascend-samples]

No trasladamos automáticamente un bloque CUDA a una unidad Ascend. Reutilizamos la matemática y volvemos a diseñar el reparto del trabajo, el layout y el protocolo entre etapas.

## Un GEMM con epílogo como primer port

La operación objetivo es sencilla de escribir:

```math
Y_{ij}=\max\left(0,\sum_{k=0}^{K-1}A_{ik}B_{kj}+b_j\right).
```

La multiplicación puede aprovechar una estación matricial. La suma de sesgo y la activación requieren trabajo adicional. Las entradas y resultados parciales deben estar en un formato que ambas partes interpreten correctamente.

Un primer prototipo separa las etapas y comprueba cada frontera. Se guarda el resultado del GEMM antes de aplicar el epílogo. Después se fusiona la ruta y se vuelve a comparar. Si la salida fusionada falla, esa descomposición permite localizar si el error viene de la multiplicación, de la interpretación del tile o del epílogo.

La validación incluye una matriz que distingue filas y columnas, K no múltiplo del bloque y valores positivos y negativos para que ReLU no oculte todos los errores. Si todos los resultados se recortan a cero, una referencia aparentemente perfecta puede estar comprobando muy poco.

## Qué publica cada repositorio de DeepSeek

DeepGEMM se centra en cómputo matricial. DeepEP se centra en comunicación útil para repartir y reunir trabajo, especialmente en mezclas de expertos. DeepJIT aporta infraestructura de compilación, caché, carga y lanzamiento. Las variantes Ascend adaptan partes de esa pila a otro acelerador. [@deepgemm;@deepep;@deepjit;@deepgemm-ascend;@deepep-ascend]

Esta separación es una lección de ingeniería. Si la red está ociosa porque el empaquetado de tokens tarda demasiado, acelerar solo el GEMM quizá no ayude. Si un kernel tarda milisegundos pero recompilarlo tarda segundos en cada llamada, el problema está en la vida del programa compilado.

| Nivel | Entrada conceptual | Salida conceptual |
|---|---|---|
| Kernel matricial | Tiles y escalas numéricas | Acumuladores o resultados. |
| Comunicación de expertos | Tokens, destinos y metadatos | Tokens redistribuidos o combinados. |
| JIT | Fuente, configuración y destino | Programa cargable y reutilizable. |
| Runtime | Programa y buffers | Trabajo ejecutado y finalización observable. |

El compilador Lumbre conserva también esta separación, aunque sus implementaciones sean mucho más pequeñas: construir un grafo, elegir kernels, generar una unidad compilable y ejecutarla son pasos identificables.

## La lectura del README también es parte del laboratorio

Al corte de esta edición, DeepEP-Ascend describe validación sobre hardware PoC Ascend 950DT y una combinación concreta de CANN y dependencias. El mismo repositorio sitúa la disponibilidad de un HDK comercial alrededor del 15 de octubre, una fecha posterior al 1 de octubre. También enumera funciones pendientes o restringidas. No sería correcto transformar esa publicación en «cualquiera puede reproducir todos los resultados hoy». [@deepep-ascend]

DeepJIT se presenta como infraestructura C++20 y documenta rutas CUDA y Ascend. Algunas funciones de su interfaz Python y calentamiento de caché figuran como trabajo en curso. La existencia del repositorio no autoriza a escribir comandos para una API todavía no disponible. [@deepjit]

::: cuidado Disponibilidad documental, binaria y física
Podemos leer código público sin disponer de un paquete instalable compatible. Podemos instalar un paquete sin tener el acelerador. Podemos tener el acelerador y un controlador que no soporte esa ruta. Cada nivel necesita evidencia propia.
:::

## Cuantización por bloques sin perder la escala

Imagina que guardamos cada número con pocas cifras y, junto a un grupo, guardamos una escala. El número aproximado se reconstruye multiplicando la cifra almacenada por esa escala. El formato del grupo y el eje de la escala forman parte del dato.

Para un ejemplo propio, supón A aproximada como $s_A\widehat A$ y B como $s_B\widehat B$, con escalas constantes en el producto considerado. Entonces $AB$ se aproxima por $s_As_B(\widehat A\widehat B)$. Si la escala cambia dentro de K, ya no podemos sacar un único producto de escalas fuera de toda la suma. Hay que agrupar contribuciones compatibles o aplicar las escalas donde corresponde.

Ese detalle explica por qué copiar únicamente el array de valores comprimidos no porta una operación cuantizada. También hay que portar las escalas, su layout, el redondeo, las saturaciones y el tipo del acumulador.

Un test útil construye dos bloques con magnitudes muy diferentes. Si ambos usan accidentalmente la primera escala, el bloque pequeño o el grande revelará el fallo. Un tensor aleatorio de rango estrecho podría disimularlo.

## Planificar una tubería con estados explícitos

Representaremos cada buffer con tres estados: libre, preparado y en uso. La estación de copia solo escribe en uno libre; al terminar lo marca preparado. La estación de cómputo solo consume uno preparado y, al finalizar, lo devuelve a libre.

La implementación real utilizará eventos, colas o primitivas del dispositivo, no necesariamente estas cadenas de texto. Pero el modelo permite formular invariantes: ningún buffer tiene dos escritores simultáneos; ningún consumidor usa datos incompletos; ningún productor sobrescribe datos aún necesarios.

```text
libre --iniciar copia--> copia pendiente
copia pendiente --finalizacion--> preparado
preparado --iniciar computo--> en uso
en uso --ultimo consumidor termina--> libre
```

Esta máquina de estados puede comprobarse en una simulación CPU del scheduler antes de disponer del acelerador. No valida la instrucción concreta ni el rendimiento, pero sí descubre errores lógicos de reutilización.

El proyecto de port del final del libro exige precisamente dos evidencias distintas: validación del protocolo abstracto y validación del backend real. Cuando falta acceso al hardware, la segunda queda pendiente; no se convierte en un resultado mediante una captura de una simulación.

## Qué conservar de esta comparación

Las máquinas cambian; las preguntas fundamentales se repiten. ¿Quién posee cada dato? ¿En qué formato está? ¿Qué etapa puede leerlo? ¿Quién sabe que la operación ha terminado? ¿Cuánto trabajo útil se hace por byte trasladado?

La originalidad de un port serio no está en renombrar llamadas. Está en reconocer qué invariantes sobreviven y qué decisiones eran específicas del hardware anterior. Ese es el conocimiento transferible que buscamos al estudiar repositorios de fabricantes distintos.

# Linux y los controladores: el suelo sobre el que corre el compilador {#ch:linux}

## Una aplicación no posee físicamente toda la GPU

Cuando un programa reserva memoria, recibe una forma de referirse a recursos que el sistema administra. El controlador participa en la creación de contextos, gestión de memoria y envío de trabajo. El sistema debe coordinar aplicaciones, proteger recursos y observar cuándo terminan operaciones.

Es útil imaginar un edificio de talleres. El compilador escribe las instrucciones de fabricación; el runtime entrega el pedido; el controlador y el sistema administran el acceso al taller. Confundirlos dificulta depurar: un cálculo erróneo, un fallo al cargar un módulo y un dispositivo ausente no tienen por qué compartir causa.

La documentación DRM y AMDGPU describe interfaces y mecanismos de gestión de memoria y sincronización. Los módulos abiertos de NVIDIA son otra fuente para estudiar una parte de la pila. Que un módulo del kernel sea abierto no implica que todo el compilador o toda la biblioteca matemática del fabricante lo sean. [@linux-mm;@linux-amdgpu;@nvidia-driver]

## Dirección virtual, residencia y transferencia

Una dirección es un identificador dentro de un espacio de direcciones, no una explicación completa de dónde están los bytes. La memoria puede requerir mapeos, migraciones o transferencias. El tiempo de una operación aparentemente pequeña puede estar dominado por preparar sus datos.

Nuestro runtime didáctico adopta una decisión sencilla: mantiene una arena del dispositivo y realiza copias explícitas. Eso hace visible el coste y reduce la cantidad de políticas que el estudiante debe comprender a la vez. No pretende modelar todos los mecanismos de memoria unificada o virtualización.

Cuando una prueba tarda mucho solo la primera vez, una hipótesis posible es compilación o carga. Otra es creación de contexto. Otra es asignación o movimiento de páginas. Separamos fases antes de culpar al kernel de cálculo.

## Fences: una promesa de finalización

Una fence representa una condición de finalización que otros participantes pueden esperar. La idea es más general que un `sleep`: esperar diez milisegundos no demuestra que el trabajo haya terminado; observar la señal de finalización correcta sí establece una dependencia según su contrato.

Supón que un productor genera un buffer y un consumidor lo reutiliza. La dependencia debe seguir al buffer y al trabajo que lo produce, no a una suposición sobre la velocidad habitual de la máquina. Una GPU menos cargada puede ocultar durante semanas un error que aparece al introducir otra aplicación.

En una arena con reutilización, el último consumidor del grafo no siempre coincide con el último consumidor físicamente terminado. El planificador estático conoce un orden lógico; el runtime asíncrono debe convertirlo en dependencias efectivas. Si varias colas se solapan, los intervalos de vida necesitan esa información adicional.

## Fallo de compilación, fallo de ejecución y reinicio del dispositivo

Un error de sintaxis pertenece a la generación o compilación. Una arquitectura no admitida pertenece a compatibilidad de destino. Una lectura fuera de rango pertenece al programa o sus contratos de memoria. Un error de dispositivo puede tener causas de software, controlador, alimentación o hardware.

No debemos diagnosticar un fallo físico a partir de una excepción genérica de la biblioteca. Tampoco debemos atribuir todo al controlador para evitar revisar un índice. La investigación empieza por un caso mínimo y por conservar el mensaje completo.

Una ficha de incidencia útil incluye la operación, forma, tipo, configuración, fuente generada, versiones, momento de aparición y si se reproduce tras reiniciar el proceso. Se separa lo observado de la causa propuesta.

::: ejemplo Una incidencia bien formulada
«El primer GEMM 31×47×65 falla al sincronizar; la compilación termina y la suma vectorial funciona. Con 32×48×64 no falla. La primera hipótesis es un borde mal protegido». Eso permite diseñar una prueba. «HIP está roto» no delimita ninguna.
:::

## Observación segura para principiantes

Leer información del sistema, consultar versiones y revisar logs autorizados es suficiente para el laboratorio inicial. No se pide descargar firmware experimental, desactivar protecciones de memoria, cambiar parámetros de IOMMU ni cargar un módulo propio como administrador.

Los repositorios de firmware o control de dispositivos son objetos de lectura avanzada. Su existencia en una organización que también desarrolla un compilador no convierte su instalación en requisito del curso. Una modificación de bajo nivel puede dejar el equipo inestable y no ayuda a aprender la primera reescritura de UOps.

El ejercicio consiste en dibujar el camino de una suma: programa Python, IR, C o código GPU, compilador del destino, biblioteca cargada, runtime, controlador y dispositivo. Añade al lado de cada frontera una evidencia que podrías recoger sin modificar el sistema.

## El límite útil del semestre

Un estudiante puede entender la separación de capas sin implementar un controlador. Nuestro proyecto construye un compilador y un runtime pequeño que usan interfaces existentes. Escribir un controlador sería otro proyecto, con un modelo de seguridad, pruebas y acceso al hardware mucho más exigentes.

La ambición del curso no se mide por cuántas capas reescribimos. Se mide por si sabemos explicar cuáles controlamos, cuáles delegamos y qué evidencia respalda cada resultado.
