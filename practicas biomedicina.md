# Práctica 1 (Claudia)

Empleando las técnicas de forzado de gramática, implementad por parejas una utilidad para extraer información relevante de alguna de las fuentes de datos clínicos presentes en el dataset ClinText-SP.

(ClinText-SP:)[https://huggingface.co/datasets/IIC/ClinText-SP/viewer]

Estos ficheros están en formato parquet, un formato diseñado para BigData. 

Ejemplo de carga de ficheros .parquet (filtrando por columna de fuente de datos):

```python
import pandas as pd
df = pd.read_parquet("data.parquet") 
filtered = df[df["source"] == "wikidisease"]
print(filtered)
Necesitaréis tener instalados:
```

```python
pandas
fastparquet
!pip install pandas
!pip install fastparquet
```

# Práctica 2 (Iago)

Por parejas, encontrad un dataset tabular público (idealmente del dominio médico)

Entrenad una estrategia de generación de datos sintéticos.
Podéis emplear librerías como:
- Synthetic Data Vault
- Synthcity
- Be Great

Valida su Fidelidad, Utilidad y Privacidad
Entrega un fichero .zip en el que haya:

Carpeta con el código desarrollado.
PDF que tenga:
- Nombre y apellidos de los miembros del equipo.
- Enlace al dataset público
- Estrategia de generación de datos sintéticos empleada (y enlace a la librería)

3 secciones:
- Fidelidad
- Utilidad
- Privacidad

Dentro de cada sección, explicar qué se empleó para validar cada uno de esos factores y discutir los resultados obtenidos.