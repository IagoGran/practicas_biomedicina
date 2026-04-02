# Decisiones técnicas para la Práctica 2

## Objetivo del documento

Este archivo resume las decisiones técnicas previas a la implementación. No pretende ser la memoria final de la práctica, sino una guía para reducir ambigüedad antes de desarrollar la solución.

La práctica pide:

- elegir un dataset tabular público, idealmente médico;
- entrenar una estrategia de generación de datos sintéticos;
- validar fidelidad, utilidad y privacidad;
- y entregar código + PDF justificando metodología y resultados.

Además, el material de clase deja claro que la validación no debe quedarse en "el dataset se parece visualmente", sino que debe cubrir:

- **Fidelidad**: similitud estadística con los datos reales, evitando colapso de moda, pérdida de diversidad y artefactos espurios;
- **Utilidad**: comprobar si los datos sintéticos sirven para análisis o modelado, especialmente en el esquema **TR-TR / TS-TR**;
- **Privacidad**: vigilar fuga de información, cercanía excesiva a registros reales y riesgo de atribución.

## 1. Exploración de datasets fáciles de utilizar

Se prioriza un dataset que sea:

- médico o sanitario;
- claramente tabular;
- fácil de cargar en una sola tabla;
- con una tarea sencilla de modelado;
- y con poco trabajo de limpieza, para dedicar más tiempo a evaluar fidelidad, utilidad y privacidad.

### Comparativa rápida

| Dataset | Dominio y tarea | Tamaño | Tipo de variables | Limpieza esperada | Ventajas para la práctica | Riesgos |
| --- | --- | --- | --- | --- | --- | --- |
| [Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | Diagnóstico de cáncer de mama, clasificación binaria | 569 instancias, 30 variables | Principalmente numéricas | Baja | Muy fácil de cargar, variables limpias, objetivo binario claro, ideal para TR-TR / TS-TR | Menos "clínico" que otros datasets hospitalarios |
| [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease) | Predicción de enfermedad cardiaca, clasificación | 303 instancias, 13 variables | Mixtas: numéricas y categóricas | Media | Más interpretable clínicamente, tamaño manejable | Tiene valores faltantes y más heterogeneidad en el preprocesado |
| [Diabetes 130-US hospitals](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) | Registros hospitalarios de diabetes, clasificación/clustering | 101766 instancias, 47 variables | Categóricas e integer | Alta | Muy realista y claramente médico | Volumen alto, más complejidad de limpieza, codificación y validación |

### Recomendación principal

El dataset recomendado para la primera implementación es **Breast Cancer Wisconsin Diagnostic**.

#### Justificación

- Es un dataset médico tabular y público.
- La tarea es de **clasificación binaria**, lo que facilita una evaluación clara de utilidad con **TR-TR / TS-TR**.
- Tiene una estructura muy limpia y no exige una fase compleja de imputación o normalización ad hoc.
- La mayor parte de sus variables son numéricas, lo que simplifica:
  - la generación sintética;
  - las comparativas estadísticas;
  - y las métricas de privacidad basadas en distancia.
- Reduce el riesgo de perder tiempo en limpieza de datos y deja más margen para justificar correctamente fidelidad, utilidad y privacidad.

### Segunda opción razonable

Si se quiere un dataset más "clínico" o más cercano a variables hospitalarias reales, la segunda opción sería **Heart Disease**.

Aun así, queda por detrás porque:

- requiere más atención a valores ausentes;
- mezcla mejor distintos tipos de variables;
- y aumenta el riesgo de que parte del trabajo se vaya en preparar el dataset en vez de evaluar bien los datos sintéticos.

### Opción descartada para esta práctica

**Diabetes 130-US hospitals** se descarta para la primera versión.

No se descarta por calidad, sino por coste técnico:

- tamaño muy superior;
- mayor heterogeneidad en variables y codificación;
- más pasos de limpieza;
- y más complejidad para explicar el pipeline completo dentro de una práctica corta.

## 2. Librerías candidatas y alineación con lo que pide la práctica

La práctica sugiere tres familias razonables: `Synthetic Data Vault (SDV)`, `Synthcity` y `Be Great`.

### Comparativa de librerías

| Librería | Enfoque | Dificultad de uso | Adecuación al caso | Evaluación disponible | Encaje con esta práctica |
| --- | --- | --- | --- | --- | --- |
| [SDV](https://docs.sdv.dev/sdv) | Ecosistema específico para datos sintéticos tabulares y relacionales | Baja-media | Muy alta para single-table | Muy buena con [SDMetrics](https://docs.sdv.dev/sdmetrics/) | La opción más defendible y directa |
| [Synthcity](https://pypi.org/project/synthcity/) | Framework de investigación con múltiples generadores y métricas | Media-alta | Alta, pero más compleja | Incluye evaluación de calidad y privacidad | Potente, pero más ambiciosa para una primera entrega |
| [Be Great](https://pypi.org/project/be-great/) | Generación tabular con modelos de lenguaje/transformers | Media-alta | Interesante para experimentar | No está tan orientada a una validación clásica cerrada como SDV+SDMetrics | Más pesada y menos directa para justificar resultados académicos |

### Recomendación principal

La combinación recomendada es **SDV + SDMetrics**.

#### Motivos

- `SDV` está diseñada específicamente para generación de datos sintéticos tabulares.
- Tiene un flujo simple para **single-table**, que es justo el caso de esta práctica.
- `SDMetrics` encaja de forma natural con la evaluación de:
  - **fidelidad**, mediante reportes de calidad y similitud;
  - **privacidad**, mediante métricas de cercanía o sobreajuste a registros reales.
- Esta combinación permite defender mejor el trabajo en el PDF porque usa métricas y reportes bien documentados.

### Sintetizador recomendado

El modelo por defecto será [`GaussianCopulaSynthesizer`](https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer).

Se elige como punto de partida porque:

- es sencillo de entrenar;
- funciona bien como baseline robusto;
- maneja tablas de una sola entidad de forma natural;
- y es suficientemente serio para una práctica donde importa más justificar evaluación que comparar muchos generadores.

### Posición de las alternativas

#### Synthcity

Es una opción fuerte si el objetivo fuera hacer una comparativa más de investigación. Tiene muchas familias de generadores y métricas de corrección/privacidad, pero para esta práctica añade más complejidad conceptual y operativa de la necesaria.

#### Be Great

Es interesante por su enfoque basado en modelos de lenguaje para datos tabulares, pero introduce más peso computacional y menos linealidad metodológica. Para una práctica donde hay que defender **fidelidad, utilidad y privacidad** con métricas clásicas, es menos directo que SDV.

### Alineación explícita con el material de clase

La elección de `SDV + SDMetrics` está alineada con lo que se pide en clase:

- **Fidelidad**: comprobar si los datos sintéticos son estadísticamente parecidos a los reales.
- **Utilidad**: comprobar si sirven para entrenar modelos útiles en un test real.
- **Privacidad**: comprobar si hay riesgo de memorization, fuga o atribución excesiva.

También encaja con la advertencia metodológica del tema: los datos sintéticos no arreglan mágicamente todos los problemas del dato real. Pueden ayudar, pero siguen dependiendo de la calidad y sesgos del conjunto de partida.

## 3. Propuesta de implementación

Esta sección define el pipeline recomendado, pero sin entrar todavía en código.

### Pipeline general

1. Cargar el dataset real en una sola tabla.
2. Separar la variable objetivo del resto de predictores.
3. Hacer un `train/test split` estratificado sobre los datos reales.
4. Entrenar el generador sintético **solo** con `real_train`.
5. Generar un conjunto `synthetic_train` del mismo tamaño que `real_train`.
6. Evaluar **fidelidad** comparando `real_train` frente a `synthetic_train`.
7. Evaluar **utilidad** con el esquema **TR-TR / TS-TR**.
8. Evaluar **privacidad** comparando `synthetic_train` frente a `real_train`.

### Restricciones metodológicas

- El conjunto de test debe seguir siendo **real**.
- El generador no debe ver nunca `real_test`.
- La utilidad debe medirse sobre capacidad de generalizar a datos reales, no sobre rendimiento en datos sintéticos.
- La privacidad no debe limitarse a "anonimización" genérica: hay que estudiar cercanía a registros reales.

## 4. Validación propuesta

### 4.1 Fidelidad

La pregunta aquí es: **¿son los datos sintéticos suficientemente similares a los reales desde el punto de vista estadístico?**

#### Qué comparar

- estadísticas descriptivas por variable:
  - media;
  - desviación típica;
  - mínimo y máximo;
  - cuartiles;
  - mediana.
- distribuciones por variable:
  - histogramas;
  - KDE o boxplots en algunas variables representativas.
- relaciones entre variables:
  - matriz de correlaciones;
  - comparación visual de correlaciones real vs sintético.

#### Qué métricas/test usar

- **Kolmogorov-Smirnov** para variables numéricas;
- comparación de correlaciones entre variables;
- y un resumen global con [`QualityReport`](https://docs.sdv.dev/sdmetrics/data-metrics/quality/quality-report) de SDMetrics.

Este enfoque encaja con lo comentado en clase:

- similitud estadística;
- detección indirecta de pérdida de diversidad;
- y vigilancia de posibles patrones espurios.

### 4.2 Utilidad

La pregunta aquí es: **¿sirven los datos sintéticos para el análisis o para entrenar modelos?**

Se seguirá exactamente el esquema visto en clase:

- **TR-TR**: entrenar con datos reales y evaluar con `test real`;
- **TS-TR**: entrenar con datos sintéticos y evaluar con `test real`.

#### Propuesta mínima

- Modelo principal: `LogisticRegression`.
- Entrenamiento 1: `real_train -> real_test`.
- Entrenamiento 2: `synthetic_train -> real_test`.

#### Métricas a comparar

- `accuracy`
- `precision`
- `recall`
- `F1`
- `ROC-AUC`

#### Extensión opcional

Si sobra tiempo, añadir `RandomForest` como segundo modelo para comprobar si la conclusión se mantiene con un clasificador no lineal.

La interpretación esperada es simple:

- si `TS-TR` se acerca razonablemente a `TR-TR`, los datos sintéticos conservan utilidad;
- si la caída es muy grande, la utilidad práctica es limitada aunque la fidelidad visual parezca buena.

### 4.3 Privacidad

La pregunta aquí es: **¿hay riesgo de que la generación sintética esté demasiado cerca de los datos reales o filtre información sensible?**

#### Métricas principales

- **DCR (Distance to Closest Record)**:
  - distancia mínima entre un registro sintético y el registro real más cercano.
  - si las distancias son demasiado pequeñas de forma sistemática, aumenta la sospecha de memorization.
- **NNDR (Nearest Neighbor Distance Ratio)**:
  - relación entre la distancia al vecino real más cercano y la distancia al segundo/tercer vecino.
  - ayuda a detectar si algunos sintéticos están "pegados" a registros reales concretos.

#### Comprobación adicional recomendada

- porcentaje de filas sintéticas idénticas o casi idénticas a filas reales;
- inspección de outliers sintéticos muy cercanos a ejemplos reales.

#### Qué dejar como marco conceptual, no como mínimo obligatorio

- **CAP (Correct Attribution Probability)**:
  - está alineada con el material de clase y es útil para explicar riesgo de atribución;
  - pero se deja fuera de la implementación base por complejidad.
- **k-anonymity**, **l-diversity** y **t-closeness**:
  - deben aparecer en la discusión teórica del PDF;
  - pero no serán el eje principal del experimento, porque el foco aquí es evaluar una estrategia de generación sintética, no rediseñar el dataset como tabla anonimizada clásica.

## 5. Nota metodológica importante

El material de clase insiste en algo clave: **garbage in, garbage out**.

Esto significa que:

- la calidad del dato fuente limita directamente la calidad del dato sintético;
- los datos sintéticos no eliminan por sí solos sesgos del dominio;
- y en datos médicos hay que reconocer posibles sesgos de selección, desbalanceo entre clases, diferencias poblacionales o problemas de captura.

Aunque se elija un dataset relativamente limpio, la memoria final debería mencionar explícitamente:

- posibles sesgos del conjunto original;
- limitaciones de representatividad;
- y el compromiso inevitable entre **privacidad**, **fidelidad** y **utilidad**.

## 6. Decisión final

La propuesta base para implementar es la siguiente:

- **Dataset**: Breast Cancer Wisconsin Diagnostic.
- **Librería principal**: SDV.
- **Evaluación**: SDMetrics + métricas estadísticas y de modelado propias.
- **Sintetizador**: GaussianCopulaSynthesizer.
- **Utilidad**: comparación TR-TR vs TS-TR con Logistic Regression.
- **Privacidad**: DCR + NNDR + comprobación de filas casi idénticas.

## Referencias oficiales

- UCI Breast Cancer Wisconsin Diagnostic: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>
- UCI Heart Disease: <https://archive.ics.uci.edu/dataset/45/heart+disease>
- UCI Diabetes 130-US hospitals: <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008>
- SDV: <https://docs.sdv.dev/sdv>
- GaussianCopulaSynthesizer: <https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer>
- SDMetrics: <https://docs.sdv.dev/sdmetrics/>
- SDMetrics Quality Report: <https://docs.sdv.dev/sdmetrics/data-metrics/quality/quality-report>
- Synthcity: <https://pypi.org/project/synthcity/>
- Be Great: <https://pypi.org/project/be-great/>
