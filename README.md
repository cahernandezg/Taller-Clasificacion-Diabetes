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