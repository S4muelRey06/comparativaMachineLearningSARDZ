
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
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

# ==========================================
# 2. PREPARAR LAS VARIABLES
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

X = training.drop(columns=columnas_eliminar)

# Codificar la variable objetivo
y = training["status"].map({
    "legitimate": 0,
    "phishing": 1
})

print("Cantidad de características:", X.shape[1])

# ==========================================
# 3. DIVIDIR ENTRENAMIENTO Y VALIDACIÓN
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Registros de entrenamiento:", X_train.shape[0])
print("Registros de validación:", X_val.shape[0])

# ==========================================
# 4. CONSTRUIR LA RED NEURONAL
# ==========================================

modelo = Pipeline([
    ("scaler", StandardScaler()),

    ("mlp", MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=False,
        random_state=42
    ))
])

# ==========================================
# 5. ENTRENAR
# ==========================================

print("\nEntrenando la red neuronal...")

modelo.fit(X_train, y_train)

print("Entrenamiento finalizado.")

# ==========================================
# 6. EVALUAR CON VALIDACIÓN
# ==========================================

y_pred = modelo.predict(X_val)

print("\nRESULTADOS DE VALIDACIÓN")

print("Accuracy:",
      accuracy_score(y_val, y_pred))

print("Precision:",
      precision_score(y_val, y_pred))

print("Recall:",
      recall_score(y_val, y_pred))

print("F1-score:",
      f1_score(y_val, y_pred))

print("\nREPORTE DE CLASIFICACIÓN")

print(classification_report(
    y_val,
    y_pred,
    target_names=["Legitimate", "Phishing"]
))

# ==========================================
# 7. MATRIZ DE CONFUSIÓN
# ==========================================

ConfusionMatrixDisplay.from_predictions(
    y_val,
    y_pred,
    display_labels=["Legitimate", "Phishing"],
    cmap="Blues"
)

plt.title("Matriz de confusión - MLP")
plt.show()

# ==========================================
# 8. CURVA DE APRENDIZAJE
# ==========================================

plt.plot(modelo.named_steps["mlp"].loss_curve_)

plt.title("Curva de pérdida - MLP")
plt.xlabel("Iteraciones")
plt.ylabel("Pérdida")
plt.grid()
plt.show()
