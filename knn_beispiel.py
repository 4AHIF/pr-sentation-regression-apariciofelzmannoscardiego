import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data      # Merkmale 
y = iris.target    # Klassen 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# KNN-Modell erstellen und trainieren 
knn = KNeighborsClassifier(n_neighbors=90)
knn.fit(X_train, y_train)

# Vorhersagen 
y_pred = knn.predict(X_test)

print(f"Genauigkeit: {accuracy_score(y_test, y_pred) * 100}%")

# Neue Blume klassifizieren
neue_blume = np.array([[5.1, 3.5, 1.4, 0.2]])
vorhersage = knn.predict(neue_blume)
print(f"Neue Blume -> {iris.target_names[vorhersage[0]]}")