
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. CARGAR LOS DATOS

training = pd.read_parquet("Training.parquet")
testing = pd.read_parquet("Testing.parquet")

# 2. PREPARAR LAS VARIABLES

columnas_eliminar = [
    "url",
    "status",
    "nb_or",
    "ratio_nullHyperlinks",
    "ratio_intRedirection",
    "ratio_intErrors",
    "submit_email",
    "sfh"
]

X_train = training.drop(columns=columnas_eliminar)
X_test = testing.drop(columns=columnas_eliminar)

y_train = training["status"].map({
    "legitimate": 0,
    "phishing": 1
})

y_test = testing["status"].map({
    "legitimate": 0,
    "phishing": 1
})

# Garantizar el mismo orden de columnas
X_test = X_test[X_train.columns]

# 3. CONSTRUIR EL MEJOR SVM

modelo = Pipeline([
    ("scaler", StandardScaler()),

    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ))
])

# 4. ENTRENAR

print("Entrenando el SVM final...")

modelo.fit(X_train, y_train)

print("Entrenamiento finalizado.")

# 5. EVALUAR CON TESTING

y_pred = modelo.predict(X_test)

print("\nRESULTADOS FINALES SVM")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))

print("\nREPORTE DE CLASIFICACIÓN")

print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Phishing"]
))

# 6. MATRIZ DE CONFUSIÓN

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["Legitimate", "Phishing"],
    cmap="Blues"
)

plt.title("Matriz de confusión - SVM")
plt.show()

