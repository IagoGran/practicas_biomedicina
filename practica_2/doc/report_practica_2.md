# Memoria de la práctica 2

## Miembros del equipo

Pendiente de completar por el equipo.

- Nombre y apellidos: `Iago Grandal del Río`
- Nombre y apellidos: `Claudia Vidal Otero`

## Dataset público empleado

- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Enlace oficial: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>

## Estrategia de generación de datos sintéticos

Se empleó `GaussianCopulaSynthesizer` de la librería `SDV` para modelar la tabla de entrenamiento real y generar un conjunto sintético del mismo tamaño. La práctica utiliza una única tabla con 30 variables numéricas y una variable objetivo binaria (`diagnosis`), lo que encaja bien con un sintetizador probabilístico tabular de tipo copula.

- Librería SDV: <https://docs.sdv.dev/sdv>
- Sintetizador usado: <https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer>

Se trabajó con el dataset local incluido en `data/wdbc.data`, de forma que la práctica puede ejecutarse desde la carpeta entregada sin depender de descargar el dataset durante la corrección.

## Fidelidad

Para validar la fidelidad se comparó `real_train` frente a `synthetic_train` con `QualityReport` y `DiagnosticReport` de `SDMetrics`.

Las métricas obtenidas fueron:

- `overall_score`: `0.9099`
- `column_shapes_score`: `0.8883`
- `column_pair_trends_score`: `0.9314`
- `data_validity_score`: `1.0000`
- `data_structure_score`: `1.0000`

Discusión de resultados:

- La puntuación global de calidad (`0.9099`) indica una similitud alta entre los datos reales y los sintéticos.
- La componente de `Column Shapes` (`0.8883`) sugiere que las distribuciones marginales de las variables se conservan razonablemente bien, aunque no de forma perfecta.
- La componente de `Column Pair Trends` (`0.9314`) es todavía mejor, lo que apunta a que el sintetizador mantiene bastante bien las relaciones entre variables.
- El diagnóstico perfecto en validez y estructura (`1.0`) indica que los datos generados respetan el esquema tabular esperado y no introducen valores fuera de dominio o inconsistencias estructurales.

En conjunto, la fidelidad puede considerarse buena. Para esta práctica, `GaussianCopulaSynthesizer` reproduce con bastante precisión la forma estadística de un dataset médico tabular relativamente limpio.

## Utilidad

La utilidad se validó siguiendo el esquema visto en clase:

- `TR-TR`: entrenar con datos reales y evaluar con test real.
- `TS-TR`: entrenar con datos sintéticos y evaluar con el mismo test real.

El modelo empleado fue una `LogisticRegression` con `StandardScaler`.

Resultados `TR-TR`:

- `accuracy`: `0.9649`
- `precision`: `0.9750`
- `recall`: `0.9286`
- `f1`: `0.9512`
- `roc_auc`: `0.9960`

Resultados `TS-TR`:

- `accuracy`: `0.8772`
- `precision`: `1.0000`
- `recall`: `0.6667`
- `f1`: `0.8000`
- `roc_auc`: `0.9656`

Diferencia `TS-TR - TR-TR`:

- `accuracy`: `-0.0877`
- `precision`: `+0.0250`
- `recall`: `-0.2619`
- `f1`: `-0.1512`
- `roc_auc`: `-0.0304`

Discusión de resultados:

- Los datos sintéticos conservan utilidad, porque un clasificador entrenado con ellos sigue funcionando razonablemente bien sobre datos reales.
- La caída en `ROC-AUC` es pequeña (`-0.0304`), lo que sugiere que la separación global entre clases sigue bastante bien preservada.
- Sin embargo, la bajada en `recall` es importante (`-0.2619`). Eso indica que el modelo entrenado con sintéticos pierde sensibilidad para detectar la clase maligna.
- La `precision` sube hasta `1.0`, pero esto no implica mejor utilidad general: en este caso parece reflejar un clasificador más conservador, que comete menos falsos positivos a costa de dejar escapar más positivos reales.

La conclusión es que los datos sintéticos sí son útiles como aproximación para análisis y modelado, pero no sustituyen por completo al dato real si se necesita máxima sensibilidad clínica.

## Privacidad

La privacidad se evaluó con dos aproximaciones:

- `DCROverfittingProtection`, para comprobar si los registros sintéticos quedan demasiado cerca de registros reales de entrenamiento.
- `DisclosureProtectionEstimate`, para estimar el riesgo de inferencia de atributos sensibles a partir de algunos atributos conocidos.

También se comprobó de forma adicional si existían coincidencias exactas entre filas reales y sintéticas.

Resultados de privacidad:

- `dcr_score`: `0.0088`
- `% closer_to_training`: `0.9956`
- `% closer_to_holdout`: `0.0044`
- `disclosure_score`: `0.7030`
- `cap_protection`: `0.3515`
- `baseline_protection`: `0.5000`
- Coincidencias exactas con `real_train`: `0`
- Coincidencias exactas con `real_test`: `0`

Discusión de resultados:

- No se detectaron copias exactas de registros reales, lo cual es una señal positiva.
- Aun así, el resultado de DCR merece atención: casi todos los registros sintéticos están más cerca de ejemplos de entrenamiento que del conjunto holdout real.
- Ese comportamiento sugiere que, aunque no haya duplicación literal, sí existe una cercanía elevada al conjunto de entrenamiento y por tanto una privacidad mejorable.
- La estimación de disclosure (`0.7030`) no apunta a una fuga extrema, pero tampoco es suficiente para afirmar que el riesgo sea bajo sin matices.

Por tanto, la principal conclusión en privacidad es que el sintetizador funciona bien en fidelidad y utilidad moderada, pero presenta señales de proximidad al dato real que obligan a ser prudentes. En un contexto biomédico real, esta estrategia debería complementarse con controles adicionales de privacidad antes de compartir los datos sintéticos.
