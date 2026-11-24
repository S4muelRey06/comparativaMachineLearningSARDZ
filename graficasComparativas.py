from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    roc_curve,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Mantener las mismas variables y partición final de los evaluadores existentes.
columnas_eliminar = [
    "url",
    "status",
    "nb_or",
    "ratio_nullHyperlinks",
    "ratio_intRedirection",
    "ratio_intErrors",
    "submit_email",
    "sfh",
]
codificacion_clases = {"legitimate": 0, "phishing": 1}

training = pd.read_parquet("Training.parquet")
testing = pd.read_parquet("Testing.parquet")

X_train = training.drop(columns=columnas_eliminar)
X_test = testing.drop(columns=columnas_eliminar)
X_test = X_test[X_train.columns]
y_train = training["status"].map(codificacion_clases)
y_test = testing["status"].map(codificacion_clases)

modelos = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ]),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
        )),
    ]),
}

predicciones = {}
scores_roc = {}

for nombre, modelo in modelos.items():
    print(f"Entrenando {nombre} con Training.parquet...")
    modelo.fit(X_train, y_train)
    predicciones[nombre] = modelo.predict(X_test)

    if nombre == "SVM":
        scores_roc[nombre] = modelo.decision_function(X_test)
    else:
        scores_roc[nombre] = modelo.predict_proba(X_test)[:, 1]

# Proyectar los registros de prueba a dos dimensiones usando PCA entrenado
# únicamente con los registros de entrenamiento estandarizados.
scaler_pca = StandardScaler().fit(X_train)
X_train_scaled = scaler_pca.transform(X_train)
X_test_scaled = scaler_pca.transform(X_test)
pca = PCA(n_components=2, random_state=42)
pca.fit(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
varianza = pca.explained_variance_ratio_ * 100

carpeta_graficas = Path("graficas")
carpeta_graficas.mkdir(exist_ok=True)

# Distribución real y clases predichas en la misma proyección 2D.
fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharex=True, sharey=True)
vistas = [
    ("Clase real", y_test.to_numpy()),
    ("Predicción SVM", predicciones["SVM"]),
    ("Predicción MLP", predicciones["MLP"]),
]
colores = {0: "#2878B5", 1: "#D9534F"}
etiquetas = {0: "Legitimate", 1: "Phishing"}

for ax, (titulo, clases) in zip(axes, vistas):
    for clase in (0, 1):
        mascara = clases == clase
        ax.scatter(
            X_test_pca[mascara, 0],
            X_test_pca[mascara, 1],
            s=24,
            alpha=0.65,
            c=colores[clase],
            label=etiquetas[clase],
            edgecolors="none",
        )
    ax.set_title(titulo)
    ax.set_xlabel(f"PC1 ({varianza[0]:.1f}% varianza)")
    ax.grid(alpha=0.2)
    ax.legend()

axes[0].set_ylabel(f"PC2 ({varianza[1]:.1f}% varianza)")
fig.suptitle("Clases reales y predichas proyectadas con PCA", fontsize=14)
fig.tight_layout()
fig.savefig(carpeta_graficas / "comparacion_pca.png", dpi=180)
plt.close(fig)

# Curvas ROC para ambos modelos sobre los mismos registros de prueba.
fig, ax = plt.subplots(figsize=(7, 6))
for nombre, scores in scores_roc.items():
    fpr, tpr, _ = roc_curve(y_test, scores)
    area = auc(fpr, tpr)
    ax.plot(fpr, tpr, linewidth=2, label=f"{nombre} (AUC = {area:.3f})")

ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Clasificador aleatorio")
ax.set_title("Curvas ROC en el conjunto de prueba")
ax.set_xlabel("Tasa de falsos positivos")
ax.set_ylabel("Tasa de verdaderos positivos")
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(carpeta_graficas / "comparacion_roc.png", dpi=180)
plt.close(fig)

# Matrices de confusión comparables lado a lado.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, nombre in zip(axes, modelos):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predicciones[nombre],
        display_labels=["Legitimate", "Phishing"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(f"Matriz de confusión - {nombre}")

fig.tight_layout()
fig.savefig(carpeta_graficas / "comparacion_matrices_confusion.png", dpi=180)
plt.close(fig)

print("Gráficas comparativas guardadas en la carpeta 'graficas'.")
