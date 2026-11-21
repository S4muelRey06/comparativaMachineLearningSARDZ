import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training = pd.read_parquet("Training.parquet")
testing = pd.read_parquet("Testing.parquet")

# 1. Distribución de las clases
print("DISTRIBUCIÓN DE CLASES")
print(training["status"].value_counts())
print(training["status"].value_counts(normalize=True) * 100)

# Gráfica
sns.countplot(data=training, x="status")
plt.title("Distribución de clases - Training")
plt.xlabel("Tipo de URL")
plt.ylabel("Cantidad")
plt.show()

# 2. Registros duplicados
print("\nREGISTROS DUPLICADOS")
print("Training:", training.duplicated().sum())
print("Testing:", testing.duplicated().sum())

# 3. Estadísticas descriptivas
print("\nESTADÍSTICAS DESCRIPTIVAS")
print(training.drop(columns=["url", "status"]).describe().T)


# 4. Buscar valores negativos
X = training.drop(columns=["url", "status"])

negativos = X.columns[(X < 0).any()]

print("\nVARIABLES CON VALORES NEGATIVOS")
for columna in negativos:
    print(columna, ":", X[columna].min())

# 5. Buscar variables constantes
constantes = X.columns[X.nunique() <= 1]

print("\nVARIABLES CONSTANTES")
print(constantes.tolist())

# 6. Comprobar posibles coincidencias entre Training y Testing
coincidencias = set(training["url"]) & set(testing["url"])

print("\nURL REPETIDAS ENTRE CONJUNTOS")
print(len(coincidencias))

# 7. Distribución de clases en Testing
print("\nDISTRIBUCIÓN TESTING")
print(testing["status"].value_counts())
print(testing["status"].value_counts(normalize=True) * 100)


# Revisar valores negativos y su frecuencia
for columna in ["domain_registration_length", "domain_age"]:
    print(f"\n{columna}")
    print(training.loc[training[columna] < 0, columna].value_counts().sort_index())

    print("\nDistribución según la clase:")
    print(
        training[training[columna] < 0]
        .groupby(["status", columna])
        .size()
    )

# Verificar las seis variables constantes en Testing
variables_constantes = [
    "nb_or",
    "ratio_nullHyperlinks",
    "ratio_intRedirection",
    "ratio_intErrors",
    "submit_email",
    "sfh"
]

print("\nVARIACIÓN EN TESTING")
for columna in variables_constantes:
    print(
        columna,
        "Valores diferentes:",
        testing[columna].nunique(),
        "Valores:",
        testing[columna].unique()
    )
