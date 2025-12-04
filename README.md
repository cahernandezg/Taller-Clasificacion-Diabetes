Taller: Clasificación con Diabetes Health Indicators
Fundación Universitaria Los Libertadores – noviembre de 2025
Autora: Camila Andrea Hernández González
1. Objetivo general

Desarrollar, evaluar e interpretar un clasificador reproducible para predecir diabetes usando el conjunto de datos Diabetes Health Indicators, siguiendo el ciclo CRISP-DM de extremo a extremo y aplicando técnicas de preprocesamiento, balanceo, validación cruzada y selección de métricas.

2. Resultados de aprendizaje abordados

Ejecución completa del ciclo CRISP-DM.

Diseño de pipelines evitando data leakage.

Diagnóstico del desbalance y uso de class_weight.

Selección argumentada de métricas priorizadas (Recall / PR-AUC).

Entrenamiento, evaluación y análisis de un clasificador base y su desempeño en test.

Elaboración de gráficos, reporte descriptivo y trazabilidad en repositorio.

3. Descripción del dataset

El dataset contiene 100 000 registros y 31 variables, incluyendo características demográficas, hábitos de vida, indicadores clínicos y dos columnas asociadas directamente al diagnóstico:

diabetes_stage

diabetes_risk_score

Estas columnas no pueden usarse como predictores porque generan leakage.

La variable objetivo es:

diagnosed_diabetes (0 = No, 1 = Sí)

El dataset presenta un desbalance moderado:

Clase 1: 59.9 %

Clase 0: 40.0 %

4. Estadística descriptiva

Se realizaron las siguientes tareas:

Revisión de estructura del dataset (shape)

Tipos de datos por columna

Ausencia de valores faltantes

Distribución del target

Resumen estadístico de variables numéricas

Identificación de variables categóricas y numéricas

Se generaron gráficos en PNG:

grafico_bmi_diabetes.png – Distribución de BMI por diagnóstico

grafico_diabetes_genero.png – Proporción de diabetes por género

5. Preparación de datos
5.1 Evitar leakage

Se eliminaron las columnas:

diabetes_stage

diabetes_risk_score

5.2 División Train/Test

Se usó train_test_split con:

80 % entrenamiento

20 % prueba

Estratificación para mantener la proporción 60/40

5.3 Identificación de tipos:

Variables categóricas:
gender, ethnicity, education_level, income_level, employment_status, smoking_status

Variables numéricas:
22 variables clínicas y de comportamiento.

6. Modelado
6.1 Pipeline de preprocesamiento

Incluye:

Estandarización de variables numéricas (StandardScaler)

Codificación One-Hot para variables categóricas (OneHotEncoder)

6.2 Modelo principal

Se entrenó Regresión Logística con:

max_iter = 1000

class_weight = balanced para corregir el desbalance del dataset.

6.3 Validación cruzada (5 folds)

Métricas evaluadas:

Recall

Precision

F1

ROC-AUC

PR-AUC

Balanced Accuracy

Resultados promedio:

Métrica	Desempeño promedio
Recall	0.877
Precision	0.929
F1	0.902
ROC-AUC	0.934
PR-AUC	0.967
Balanced Accuracy	0.888
7. Evaluación en test

Después del entrenamiento final:

Reporte de clasificación:

Accuracy test: 0.89

Recall clase 1: 0.88

ROC-AUC test: 0.934

PR-AUC test: 0.967

Archivos generados:

matriz_confusion_logreg.png

curva_roc_logreg.png

curva_pr_logreg.png

8. Interpretación de resultados

El modelo tiene un muy buen desempeño global (ROC-AUC y PR-AUC altos).

La métrica prioritaria es Recall, por ser un problema clínico donde es peor no detectar personas con diabetes.

El modelo mantiene un equilibrio adecuado entre Recall y Precision.

No hay evidencia de sobreajuste según la comparación Train vs Test.

9. Estructura del repositorio
Taller-Clasificacion-DiabetesCAHG/
│── diabetes_dataset.csv
│── main.py
│── README.md
│── requirements.txt
│── grafico_bmi_diabetes.png
│── grafico_diabetes_genero.png
│── matriz_confusion_logreg.png
│── curva_roc_logreg.png
│── curva_pr_logreg.png
│── notebook_taller.ipynb
└── .gitignore

10. Cómo ejecutar el proyecto

Crear y activar un entorno virtual

Instalar dependencias:

pip install -r requirements.txt


Ejecutar:

python main.py


Abrir el notebook opcional:

jupyter notebook notebook_taller.ipynb

11. Limitaciones

El dataset es sintético; los resultados pueden no trasladarse a escenarios clínicos reales.

Solo se probó un modelo base; versiones más complejas (RandomForest, XGBoost) podrían mejorar resultados.

No se realizó calibración de probabilidades (pendiente para etapa futura).

12. Próximos pasos

Explorar modelos avanzados

Calibración con Platt o Isotónica

SHAP para interpretabilidad

Pipeline con optimización de hiperparámetros (GridSearchCV o Optuna)