import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. CARGAR LOS DATOS
# ==========================================

print("Cargando dataset...")
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

y = training["status"].map({
    "legitimate": 0,
    "phishing": 1
})

print("Cantidad de características:", X.shape[1])


# ==========================================
# 3. DIVIDIR LOS DATOS
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDIVISIÓN DE LOS DATOS")
print("Entrenamiento:", X_train.shape[0])
print("Validación:", X_val.shape[0])


# ==========================================
# 4. DEFINIR ARQUITECTURAS
# ==========================================

# Cada tupla representa la cantidad de neuronas
# en cada capa oculta.
arquitecturas = [
    (32,),
    (64, 32),
    (128, 64),
    (128, 64, 32)
]

resultados = []


# ==========================================
# 5. ENTRENAR Y EVALUAR CADA ARQUITECTURA
# ==========================================

for arquitectura in arquitecturas:

    print("\n" + "=" * 50)
    print("Entrenando arquitectura:", arquitectura)
    print("=" * 50)

    modelo = Pipeline([
        ("scaler", StandardScaler()),

        ("mlp", MLPClassifier(
            hidden_layer_sizes=arquitectura,
            activation="relu",
            solver="adam",
            alpha=0.0001,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        ))
    ])

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicción sobre validación
    predicciones = modelo.predict(X_val)

    # Métricas
    accuracy = accuracy_score(y_val, predicciones)
    precision = precision_score(
        y_val, predicciones, zero_division=0
    )
    recall = recall_score(
        y_val, predicciones, zero_division=0
    )
    f1 = f1_score(
        y_val, predicciones, zero_division=0
    )

    iteraciones = modelo.named_steps["mlp"].n_iter_
    loss = modelo.named_steps["mlp"].loss_

    resultados.append({
        "Arquitectura": str(arquitectura),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Iteraciones": iteraciones,
        "Loss final": loss
    })

    print("Accuracy:", round(accuracy, 5))
    print("Precision:", round(precision, 5))
    print("Recall:", round(recall, 5))
    print("F1-score:", round(f1, 5))
    print("Iteraciones:", iteraciones)
    print("Loss final:", round(loss, 6))


# ==========================================
# 6. COMPARAR RESULTADOS
# ==========================================

tabla = pd.DataFrame(resultados)

print("\n" + "=" * 75)
print("COMPARACIÓN DE ARQUITECTURAS MLP")
print("=" * 75)
print(tabla.to_string(index=False))

# Seleccionar la arquitectura con mayor F1-score
mejor_indice = tabla["F1-score"].idxmax()
mejor_resultado = tabla.loc[mejor_indice]

print("\nMEJOR ARQUITECTURA")
print("Arquitectura:", mejor_resultado["Arquitectura"])
print("F1-score:", round(mejor_resultado["F1-score"], 5))


# ==========================================
# 7. EXPORTAR RESULTADOS
# ==========================================

tabla.to_csv("resultados_arquitecturas_mlp.csv", index=False)
print("\nResultados guardados en resultados_arquitecturas_mlp.csv")


# ==========================================
# 8. GRAFICAR RESULTADOS
# ==========================================

carpeta_graficas = Path("graficas")
carpeta_graficas.mkdir(exist_ok=True)

metricas = ["Accuracy", "Precision", "Recall", "F1-score"]

fig, ax = plt.subplots(figsize=(10, 5))

for metrica in metricas:
    ax.plot(
        tabla["Arquitectura"],
        tabla[metrica],
        marker="o",
        label=metrica
    )

ax.set_ylim(0, 1.05)
ax.set_title("Comparación de arquitecturas MLP")
ax.set_xlabel("Arquitectura (capas ocultas)")
ax.set_ylabel("Puntuación")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(
    carpeta_graficas / "comparacion_arquitecturas_mlp.png",
    dpi=150
)
plt.close(fig)

print("Gráfica guardada en graficas/comparacion_arquitecturas_mlp.png")
