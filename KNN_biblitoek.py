import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.neighbors import KNeighborsClassifier

# CSV-Dateien laden
train_data = pd.read_csv(r'train.csv')
valid_data = pd.read_csv(r'valid.csv')

# Labels und Features extrahieren
X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]
X_valid, y_valid = valid_data.iloc[:, 1:], valid_data.iloc[:, 0]

# Labels in numerische Werte umwandeln, falls notwendig
y_train = pd.Categorical(y_train).codes
y_valid = pd.Categorical(y_valid).codes

# Optimierung des besten k-Werts basierend auf den Validierungsdaten
best_k = 1
best_score = 0

for k in range(1, 21):  # Teste k-Werte von 1 bis 20
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    valid_score = knn.score(X_valid, y_valid)
    if valid_score > best_score:
        best_k = k
        best_score = valid_score

print(f"Bester k-Wert: {best_k}, Validierungsgenauigkeit: {best_score:.2f}")

# Trainieren des finalen Modells mit dem besten k-Wert
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Vorhersagen für die Testdaten
y_pred = knn.predict(X_valid)

# Metriken berechnen
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred, average='weighted')
recall = recall_score(y_valid, y_pred, average='weighted')
f1 = f1_score(y_valid, y_pred, average='weighted')

# Confusion-Matrix erstellen und anzeigen
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png', dpi = 500)  # Confusion Matrix speichern
plt.show()

# Metriken speichern
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}

# Metriken in eine Textdatei speichern
with open('metrics.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value:.4f}\n")

print("Modell und Metriken erfolgreich gespeichert.")
