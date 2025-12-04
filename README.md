Taller: Clasificación con Diabetes Health Indicators

Fundación Universitaria Los Libertadores – Noviembre 2025
Autora: Camila Andrea Hernández González

Proyecto: Predicción de Diabetes con Machine Learning

Este proyecto implementa un clasificador reproducible para predecir la presencia de diabetes usando el dataset Diabetes Health Indicators.
Se desarrolla siguiendo el ciclo CRISP-DM, integrando buenas prácticas: no leakage, validación cruzada, técnicas de balanceo y uso apropiado de métricas.

1. Objetivo general

Desarrollar, evaluar e interpretar un clasificador reproducible para predecir diabetes usando el dataset Diabetes Health Indicators, aplicando CRISP-DM extremo a extremo y seleccionando una métrica prioritaria basada en costo-error.

2. Descripción del dataset

El conjunto de datos contiene 100 000 registros y 31 variables, incluyendo:

Datos demográficos

Hábitos de vida

Indicadores clínicos (glucosa, colesterol, presión arterial, etc.)

Antecedentes familiares y personales

Variable objetivo: diagnosed_diabetes (0 = No, 1 = Sí)

3. Estadística descriptiva

Distribución del target:

Clases balanceadas moderadamente:

59.9% con diabetes

40.0% sin diabetes

No hay valores faltantes en el dataset.
Las variables numéricas presentan rangos adecuados para escalamiento (standardization).

4. Preprocesamiento (Pipeline reproducible)

Eliminación de columnas con leakage:

diabetes_stage

diabetes_risk_score

Separación Train/Test con stratify=y para conservar la proporción 60/40.

Pipeline scikit-learn:

Escalado (StandardScaler) para numéricas

One-Hot Encoding para categóricas

Modelo: Regresión Logística con class_weight="balanced"

Este setup se asegura de evitar data leakage y permite reproducibilidad total.

5. Validación cruzada (5-fold)

Promedios obtenidos:

Recall: 0.877

Precision: 0.929

F1-score: 0.902

ROC-AUC: 0.934

PR-AUC: 0.967

Balanced Accuracy: 0.888

Métrica prioritaria:
Se selecciona PR-AUC, ya que en este tipo de problema es más importante reducir falsos negativos y evaluar rendimiento sobre la clase positiva.

6. Resultados en Test

Accuracy: 0.89

ROC-AUC: 0.934

PR-AUC: 0.967

La regresión logística muestra excelente capacidad discriminativa y buen equilibrio entre sensibilidad (recall) y precisión.

Matriz de confusión, curva ROC y curva Precision-Recall están incluidas en la carpeta del proyecto como archivos PNG.

7. Conclusiones

El modelo basado en Regresión Logística es altamente efectivo para este dataset.

La métrica prioritaria PR-AUC muestra excelente desempeño (0.967), lo que indica muy buen manejo de falsos positivos y negativos.

El pipeline asegura cero leakage y total reproducibilidad.

Se gestionó el desbalance mediante class_weight="balanced" dentro del proceso de validación cruzada, lo cual mejora la calidad del modelo.

La metodología CRISP-DM se ejecutó de forma completa: exploración, preparación, modelado, evaluación y documentación.

8. Cómo ejecutar el proyecto
1. Crear entorno virtual (opcional)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

2. Instalar dependencias
pip install -r requirements.txt

3. Ejecutar el script principal
python main.py

4. Abrir el notebook

Desde cualquier entorno:

jupyter notebook

9. Estructura del repositorio
│── main.py
│── README.md
│── requirements.txt
│── Taller_Diabetes_Clasificacion_CAH.ipynb
│── diabetes_dataset.csv
│── grafico_bmi_diabetes.png
│── grafico_diabetes_genero.png
│── curva_roc_logreg.png
│── curva_pr_logreg.png
│── matriz_confusion_logreg.png

10. Licencia

Uso académico

## Resultados y discusión

El conjunto de datos contiene 100 000 registros y 31 variables. No se encontraron valores faltantes. La variable objetivo `diagnosed_diabetes` presenta una ligera mayor proporción de casos positivos: aproximadamente 60 % de las observaciones corresponden a personas con diagnóstico de diabetes y 40 % a personas sin diagnóstico. Este desbalance motivó el uso de `class_weight="balanced"` en la regresión logística.

En la estadística descriptiva se observan valores medios de IMC (BMI) alrededor de 25–26, con una distribución más cargada hacia valores altos en el grupo con diabetes. El histograma de BMI por diagnóstico muestra que, aunque los rangos se solapan, las personas con diabetes tienden a concentrarse más en IMC altos, lo que es consistente con la literatura sobre sobrepeso y riesgo metabólico. La gráfica de proporción de diabetes por género indica prevalencias muy similares entre hombres y mujeres, con ligeras variaciones que no parecen dominantes frente a otros factores de riesgo.

El modelo principal fue una regresión logística dentro de un pipeline de `scikit-learn` que incluye escalado de variables numéricas y codificación one-hot de las categóricas. Para evitar leakage se excluyeron explícitamente las variables `diabetes_stage` y `diabetes_risk_score` antes de separar la base en entrenamiento y prueba. La partición se hizo con `train_test_split` estratificado, manteniendo el 60/40 de la variable objetivo en ambos subconjuntos.

En la validación cruzada estratificada de 5 pliegues, la regresión logística obtuvo valores estables: recall medio cercano a 0.88, precisión alrededor de 0.93, F1 aproximadamente 0.90, ROC-AUC en torno a 0.93, PR-AUC alrededor de 0.97 y balanced accuracy cercana a 0.89. Estos resultados indican que el modelo es capaz de distinguir bien entre personas con y sin diabetes, especialmente cuando se evalúa con métricas sensibles al desbalance como PR-AUC y balanced accuracy.

En el conjunto de prueba (20 % de los datos), el comportamiento fue muy similar al observado en la validación cruzada. El reporte de clasificación muestra una precisión alta para la clase positiva (≈0.93) y un recall también alto (≈0.88). La matriz de confusión refleja que el modelo detecta la mayoría de los casos de diabetes, aunque todavía existen falsos negativos (personas con diabetes clasificadas como sanas) que conviene minimizar porque tienen mayor costo en términos de salud pública. La ROC-AUC (~0.93) y la PR-AUC (~0.97) confirman que, para una amplia gama de umbrales, el modelo mantiene una buena capacidad discriminativa.

Las curvas ROC y Precision–Recall muestran el típico compromiso entre recall y precisión. Desde la perspectiva del problema, es preferible priorizar un alto recall para la clase positiva, aceptando algunos falsos positivos siempre que el modelo sirva como herramienta de tamizaje o priorización y no como diagnóstico definitivo. En ese contexto, la métrica prioritaria elegida es la PR-AUC, complementada por el recall de la clase positiva, porque reflejan mejor el costo de dejar pasar casos de diabetes que deberían ser detectados de forma temprana.

En resumen, el clasificador basado en regresión logística, con un pipeline correctamente configurado y sin leakage, logra un desempeño alto y estable tanto en validación cruzada como en el conjunto de prueba. El modelo es sencillo de interpretar, reproducible y adecuado como primer filtro automatizado para apoyar la detección temprana de posibles casos de diabetes en poblaciones grandes.
