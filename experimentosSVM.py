
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# 1. CARGAR LOS DATOS

training = pd.read_parquet("Training.parquet")

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

X = training.drop(columns=columnas_eliminar)

y = training["status"].map({
    "legitimate": 0,
    "phishing": 1
})

# 3. DIVIDIR LOS DATOS

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. DEFINIR EXPERIMENTOS CON C

valores_C = [0.01, 0.1, 1, 10, 100]

resultados_C = []

# 5. ENTRENAR Y EVALUAR CON C

for C in valores_C:

    print("\nEntrenando SVM con C =", C)

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=C,
            gamma="scale"
        ))
    ])

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_val)

    accuracy = accuracy_score(y_val, predicciones)
    precision = precision_score(y_val, predicciones)
    recall = recall_score(y_val, predicciones)
    f1 = f1_score(y_val, predicciones)

    resultados_C.append({
        "Kernel": "rbf",
        "C": C,
        "Gamma": "scale",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print("Accuracy:", round(accuracy, 5))
    print("Precision:", round(precision, 5))
    print("Recall:", round(recall, 5))
    print("F1-score:", round(f1, 5))

# 6. COMPARAR RESULTADOS DE C

tabla_C = pd.DataFrame(resultados_C)

print("\nCOMPARACIÓN DE CONFIGURACIONES C")
print(tabla_C.to_string(index=False))

# 7. IDENTIFICAR MEJOR C

mejor_indice_C = tabla_C["F1-score"].idxmax()
mejor_resultado_C = tabla_C.loc[mejor_indice_C]

print("\nMEJOR CONFIGURACIÓN C")
print(mejor_resultado_C.to_string())

print("\nResultados de C guardados.")

# 9. DEFINIR EXPERIMENTOS 
valores_gamma = ["scale", 0.001, 0.01, 0.1, 1]

resultados_gamma = []

# 10. ENTRENAR Y EVALUAR 

for gamma in valores_gamma:

    print("\nEntrenando SVM con gamma =", gamma)

    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma=gamma
        ))
    ])

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_val)

    accuracy = accuracy_score(y_val, predicciones)
    precision = precision_score(y_val, predicciones)
    recall = recall_score(y_val, predicciones)
    f1 = f1_score(y_val, predicciones)

    resultados_gamma.append({
        "Kernel": "rbf",
        "C": 1.0,
        "Gamma": gamma,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print("Accuracy:", round(accuracy, 5))
    print("Precision:", round(precision, 5))
    print("Recall:", round(recall, 5))
    print("F1-score:", round(f1, 5))

# 11. COMPARAR RESULTADOS

tabla_gamma = pd.DataFrame(resultados_gamma)

print("\nCOMPARACIÓN DE CONFIGURACIONES GAMMA")
print(tabla_gamma.to_string(index=False))

# 12. IDENTIFICAR MEJOR 

mejor_indice_gamma = tabla_gamma["F1-score"].idxmax()
mejor_resultado_gamma = tabla_gamma.loc[mejor_indice_gamma]

print("\nMEJOR CONFIGURACIÓN GAMMA")
print(mejor_resultado_gamma.to_string())

# 13. EXPORTAR RESULTADOS 

tabla_C["Experimento"] = "C"
tabla_gamma["Experimento"] = "Gamma"

tabla_final = pd.concat(
    [tabla_C, tabla_gamma],
    ignore_index=True
)

tabla_final.to_csv(
    "resultados_experimentosSVM.csv",
    index=False
)
