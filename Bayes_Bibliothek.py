import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

# CSV-Dateien laden
train_data = pd.read_csv(r'C:\Daten_JD\HKA\VDKI\projekt\Versuch_Knn\train.csv')

test_data = pd.read_csv(r'C:\Daten_JD\HKA\VDKI\projekt\Versuch_Knn\test.csv')

# Labels und Features extrahieren
X_train, y_train = train_data.iloc[:, 1:], train_data.iloc[:, 0]

X_test, y_test = test_data.iloc[:, 1:], test_data.iloc[:, 0]

# Labels in numerische Werte umwandeln, falls notwendig
y_train = pd.Categorical(y_train).codes

y_test = pd.Categorical(y_test).codes



from sklearn.naive_bayes import GaussianNB  # importiere Bayes Klassifikator

bayes_clf = GaussianNB() # zuweisen zu Klassifikatorvariable/Objekt
bayes_clf.fit(X_train,y_train) # Klassifikator anwenden auf Trainingsdaten

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

# Vorhersagen für die Testdaten
y_pred = bayes_clf.predict(X_test)

# Metriken berechnen
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Confusion-Matrix erstellen und anzeigen
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix_Bayes_Bibliothek.png', dpi = 500)  # Confusion Matrix speichern
plt.show()

# Metriken speichern
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1}

# Metriken in eine Textdatei speichern
with open('metrics_Bayes_Bibliothek.txt', 'w') as f:
    for key, value in metrics.items():
        f.write(f"{key}: {value:.4f}\n")

print("Modell und Metriken erfolgreich gespeichert.")



