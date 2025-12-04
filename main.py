import pandas as pd  # Manejo de datos en tablas
import matplotlib.pyplot as plt  # Gráficos básicos

from sklearn.model_selection import train_test_split

# ============================================================
# 1. Cargar el dataset
# ============================================================

df = pd.read_csv("diabetes_dataset.csv")

print("======================================")
print("1. ESTADÍSTICA DESCRIPTIVA BÁSICA")
print("======================================\n")

# 1.1 Shape
print("Shape (filas, columnas):", df.shape)

# 1.2 Nombres de columnas
print("\nColumnas:")
print(df.columns)

# 1.3 Primeras filas
print("\nPrimeras filas:")
print(df.head())

# 1.4 Tipos de datos
print("\nTipos de datos:")
print(df.dtypes)

# 1.5 Valores faltantes
print("\nValores faltantes por columna:")
print(df.isna().sum())

# 1.6 Distribución de la variable objetivo
print("\nDistribución de 'diagnosed_diabetes' (frecuencias):")
print(df["diagnosed_diabetes"].value_counts())

print("\nDistribución de 'diagnosed_diabetes' (proporciones):")
print(df["diagnosed_diabetes"].value_counts(normalize=True))


# 1.7 Descriptiva de variables numéricas
print("\nResumen estadístico de variables numéricas:")
print(df.describe().T)


# ============================================================
# 2. SEPARAR VARIABLE OBJETIVO Y EVITAR LEAKAGE
# ============================================================

print("\n======================================")
print("2. DEFINICIÓN DE TARGET Y PREDICTORES")
print("======================================\n")

target = "diagnosed_diabetes"

# Columnas que podrían causar leakage (información muy directa del diagnóstico)
cols_leakage = ["diabetes_stage", "diabetes_risk_score"]

# X: todas menos target y columnas de leakage
X = df.drop(columns=[target] + cols_leakage)
y = df[target]

print("Shape de X (features):", X.shape)
print("Shape de y (target):", y.shape)

# ============================================================
# 3. TRAIN-TEST SPLIT CON ESTRATIFICACIÓN
# ============================================================

print("\n======================================")
print("3. TRAIN-TEST SPLIT")
print("======================================\n")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,        # 20% para prueba
    random_state=42,      # para reproducibilidad
    stratify=y            # mantiene el 60/40 en train y test
)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)

print("\nDistribución en y_train:")
print(y_train.value_counts(normalize=True))

print("\nDistribución en y_test:")
print(y_test.value_counts(normalize=True))


# ============================================================
# 4. IDENTIFICAR VARIABLES CATEGÓRICAS Y NUMÉRICAS
# ============================================================

print("\n======================================")
print("4. TIPOS DE VARIABLES (CATEGÓRICAS / NUMÉRICAS)")
print("======================================\n")

cat_features = X.select_dtypes(include=['object']).columns.tolist()
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("Variables categóricas:", cat_features)
print("\nVariables numéricas:", num_features)

# ============================================================
# 5. GRÁFICOS DESCRIPTIVOS
# ============================================================

print("\n======================================")
print("5. GRÁFICOS DESCRIPTIVOS (SE GUARDAN COMO PNG)")
print("======================================\n")

# Histograma de BMI separado por diagnóstico
plt.figure()
df[df["diagnosed_diabetes"] == 0]["bmi"].hist(alpha=0.6)
df[df["diagnosed_diabetes"] == 1]["bmi"].hist(alpha=0.6)
plt.xlabel("BMI")
plt.ylabel("Frecuencia")
plt.title("Distribución de BMI según diagnóstico de diabetes")
plt.legend(["No Diabetes (0)", "Diabetes (1)"])
plt.tight_layout()
plt.savefig("grafico_bmi_diabetes.png")
plt.close()

# Distribución de diabetes por género
plt.figure()
df.groupby("gender")["diagnosed_diabetes"].mean().plot(kind="bar")
plt.ylabel("Proporción con diabetes")
plt.title("Proporción de diabetes por género")
plt.tight_layout()
plt.savefig("grafico_diabetes_genero.png")
plt.close()

print("Se guardaron los archivos:")
print("- grafico_bmi_diabetes.png")
print("- grafico_diabetes_genero.png")

# ============================================================
# 6. PIPELINE DE PREPROCESAMIENTO + MODELO (Regresión Logística)
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

print("\n======================================")
print("6. PIPELINE + REGRESIÓN LOGÍSTICA")
print("======================================\n")

# 6.1 Definir el preprocesamiento:
# - Escalar numéricas
# - One-hot encoding para categóricas

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ]
)

# 6.2 Definir el modelo base
# class_weight='balanced' para manejar el desbalance (60% / 40%)
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# 6.3 Armar el pipeline completo
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", log_reg),
    ]
)

# 6.4 Validación cruzada con varias métricas
scoring = {
    "recall": "recall",                       # prioridad: encontrar diabéticos
    "precision": "precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",           # área bajo curva Precision-Recall
    "bal_acc": "balanced_accuracy",
}

scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

print("Resultados de validación cruzada (5-fold):\n")
for name in scoring.keys():
    valores = scores[f"test_{name}"]
    print(f"- {name}: {valores.mean():.3f} ± {valores.std():.3f}")
# ============================================================
# 7. AJUSTE FINAL EN TRAIN Y EVALUACIÓN EN TEST
# ============================================================

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score,
)

print("\n======================================")
print("7. EVALUACIÓN EN CONJUNTO DE PRUEBA (TEST)")
print("======================================\n")

# 7.1 Entrenar el pipeline completo con todos los datos de entrenamiento
pipeline.fit(X_train, y_train)

# 7.2 Predicciones en test
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# 7.3 Métricas en test
print("Reporte de clasificación (test):\n")
print(classification_report(y_test, y_pred))

roc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print(f"ROC-AUC en test: {roc:.3f}")
print(f"PR-AUC (Precision-Recall AUC) en test: {pr_auc:.3f}")

# 7.4 Matriz de confusión (imagen)
plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Matriz de confusión - Regresión Logística")
plt.tight_layout()
plt.savefig("matriz_confusion_logreg.png")
plt.close()

print("Se guardó la matriz de confusión como 'matriz_confusion_logreg.png'")
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Curva ROC
plt.figure()
RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("Curva ROC - Regresión Logística")
plt.tight_layout()
plt.savefig("curva_roc_logreg.png")
plt.close()

# Curva Precision-Recall
plt.figure()
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Curva Precision-Recall - Regresión Logística")
plt.tight_layout()
plt.savefig("curva_pr_logreg.png")
plt.close()

print("También se guardaron:")
print("- curva_roc_logreg.png")
print("- curva_pr_logreg.png")
# ============================================================
# 6. PIPELINE: PREPROCESAMIENTO + MODELO (REGRESIÓN LOGÍSTICA)
# ============================================================

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
# Preprocesador: escala numéricas y aplica One-Hot a categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ],
    remainder="drop",  # solo usamos estas columnas
)
log_reg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",  # para compensar el 60/40
    solver="lbfgs"
)

pipeline_logreg = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", log_reg),
    ]
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "roc_auc": "roc_auc",
    "balanced_accuracy": "balanced_accuracy",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "pr_auc": "average_precision",  # PR-AUC
}

scores = cross_validate(
    pipeline_logreg,
    X_train,
    y_train,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False,
)

print("\n======================================")
print("6. VALIDACIÓN CRUZADA - REGRESIÓN LOGÍSTICA")
print("======================================\n")

for metric_name in scoring.keys():
    mean_score = scores[f"test_{metric_name}"].mean()
    std_score = scores[f"test_{metric_name}"].std()
    print(f"{metric_name}: {mean_score:.3f} (+/- {std_score:.3f})")
# ============================================================
# 7. AJUSTE FINAL EN TRAIN Y EVALUACIÓN EN TEST
# ============================================================

pipeline_logreg.fit(X_train, y_train)

y_pred = pipeline_logreg.predict(X_test)
y_proba = pipeline_logreg.predict_proba(X_test)[:, 1]

roc = roc_auc_score(y_test, y_proba)
pr = average_precision_score(y_test, y_proba)

print("\nROC-AUC en test:", round(roc, 3))
print("PR-AUC en test:", round(pr, 3))

print("\nReporte de clasificación en test (umbral 0.5):")
print(classification_report(y_test, y_pred, digits=3))
# ============================================================
# 6. PREPROCESAMIENTO (PIPELINE SIN LEAKAGE)
# ============================================================

print("\n======================================")
print("6. PREPROCESAMIENTO CON PIPELINE")
print("======================================\n")

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Preprocesamiento para variables categóricas
cat_transformer = OneHotEncoder(handle_unknown="ignore")

# Preprocesamiento para variables numéricas
num_transformer = StandardScaler()

# Ensamblar el preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)
    ]
)

print("Preprocesador configurado correctamente.")
# ============================================================
# 7. MODELO BASE: Regresión Logística + Cross-Validation
# ============================================================

print("\n======================================")
print("7. MODELO BASE CON VALIDACIÓN CRUZADA")
print("======================================\n")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

# Modelo
model = LogisticRegression(max_iter=2000)

# Pipeline completo: preprocesamiento + modelo
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# Métricas a evaluar: recall y ROC-AUC
scoring = {
    "recall": "recall",
    "roc_auc": "roc_auc"
}

scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring=scoring,
    n_jobs=-1,
    return_train_score=False
)

print("Recall promedio:", scores["test_recall"].mean())
print("ROC-AUC promedio:", scores["test_roc_auc"].mean())

