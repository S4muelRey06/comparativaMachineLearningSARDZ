
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. CARGAR LOS DATOS
# ==========================================

training = pd.read_parquet("Training.parquet")
testing = pd.read_parquet("Testing.parquet")

# ==========================================
# 2. PREPARAR LOS DATOS
# ==========================================

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

# Verificar que las características coincidan
assert list(X_train.columns) == list(X_test.columns)

# ==========================================
# 3. CONSTRUIR LA MLP DEFINITIVA
# ==========================================

modelo = Pipeline([
    ("scaler", StandardScaler()),

    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42
    ))
])

# ==========================================
# 4. ENTRENAMIENTO FINAL
# ==========================================

print("Entrenando modelo definitivo...")

modelo.fit(X_train, y_train)

print("Entrenamiento finalizado.")

# ==========================================
# 5. EVALUACIÓN CON TESTING
# ==========================================

predicciones = modelo.predict(X_test)

print("\nRESULTADOS FINALES - TESTING")

print("Accuracy:",
      accuracy_score(y_test, predicciones))

print("Precision:",
      precision_score(y_test, predicciones))

print("Recall:",
      recall_score(y_test, predicciones))

print("F1-score:",
      f1_score(y_test, predicciones))

print("\nREPORTE DE CLASIFICACIÓN")

print(classification_report(
    y_test,
    predicciones,
    target_names=["Legitimate", "Phishing"]
))

# ==========================================
# 6. MATRIZ DE CONFUSIÓN
# ==========================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predicciones,
    display_labels=["Legitimate", "Phishing"],
    cmap="Blues"
)

plt.title("Matriz de confusión - MLP definitiva")
plt.tight_layout()
plt.savefig("matriz_confusion_mlp.png", dpi=300)
plt.show()

print("\nMatriz guardada correctamente.")
