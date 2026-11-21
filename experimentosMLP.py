
import pandas as pd

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
# 4. DEFINIR EXPERIMENTOS
# ==========================================

arquitectura = (128, 64)

valores_alpha = [
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1
]

resultados = []

# ==========================================
# 5. ENTRENAR Y EVALUAR
# ==========================================

for alpha in valores_alpha:

    print("\n" + "=" * 45)
    print("Entrenando con alpha:", alpha)
    print("=" * 45)

    modelo = Pipeline([
        ("scaler", StandardScaler()),

        ("mlp", MLPClassifier(
            hidden_layer_sizes=arquitectura,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        ))
    ])

    # Entrenamiento
    modelo.fit(X_train, y_train)

    # Predicción
    predicciones = modelo.predict(X_val)

    # Métricas
    accuracy = accuracy_score(y_val, predicciones)

    precision = precision_score(
        y_val,
        predicciones,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predicciones,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predicciones,
        zero_division=0
    )

    iteraciones = modelo.named_steps["mlp"].n_iter_

    loss = modelo.named_steps["mlp"].loss_

    # Guardar resultados
    resultados.append({
        "Arquitectura": str(arquitectura),
        "Alpha": alpha,
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

# ==========================================
# 6. COMPARAR RESULTADOS
# ==========================================

tabla = pd.DataFrame(resultados)

print("\n")
print("=" * 65)
print("COMPARACIÓN DE REGULARIZACIÓN L2")
print("=" * 65)

print(tabla.to_string(index=False))

# ==========================================
# 7. IDENTIFICAR MEJOR CONFIGURACIÓN
# ==========================================

mejor_indice = tabla["F1-score"].idxmax()

mejor_resultado = tabla.loc[mejor_indice]

print("\nMEJOR CONFIGURACIÓN")

print("Arquitectura:", mejor_resultado["Arquitectura"])
print("Alpha:", mejor_resultado["Alpha"])
print("Accuracy:", mejor_resultado["Accuracy"])
print("Precision:", mejor_resultado["Precision"])
print("Recall:", mejor_resultado["Recall"])
print("F1-score:", mejor_resultado["F1-score"])

# ==========================================
# 8. EXPORTAR RESULTADOS
# ==========================================

tabla.to_csv(
    "resultados_regularizacion.csv",
    index=False
)

print("\nResultados guardados en resultados_regularizacion.csv")
