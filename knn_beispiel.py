import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Iris-Datensatz laden (150 Blumen, 4 Merkmale, 3 Klassen)
iris = load_iris()
X = iris.data    # Merkmale: Kelch-/Bluetenblattlaenge und -breite
y = iris.target  # Klassen: 0=Setosa, 1=Versicolor, 2=Virginica

# Aufteilen in Training (80%) und Test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN-Modell erstellen und trainieren (k=5 Nachbarn)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Vorhersagen auf Testdaten
y_pred = knn.predict(X_test)

# Genauigkeit auswerten
print(f"Genauigkeit: {accuracy_score(y_test, y_pred) * 100:.1f}%")
print()
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Neue, unbekannte Blume klassifizieren
neue_blume = np.array([[5.1, 3.5, 1.4, 0.2]])  # typische Setosa-Werte
vorhersage = knn.predict(neue_blume)
print(f"Neue Blume -> vorhergesagte Klasse: {iris.target_names[vorhersage[0]]}")

# Verschiedene k-Werte vergleichen
print("\nVergleich verschiedener k-Werte:")
for k in [1, 3, 5, 7, 11]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  k={k:2d} -> Genauigkeit: {acc * 100:.1f}%")
