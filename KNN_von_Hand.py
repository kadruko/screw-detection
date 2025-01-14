from csv import reader
from math import sqrt

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


# CSV einlesen mit flexiblem Delimiter
# Erste Zeile (Header) wird übersprungen
def laden_csv(filename):
    dataset = []
    delimiters = [',', ';']
    for delimiter in delimiters:
        try:
            with open(filename, 'r') as file:
                csv_reader = reader(file, delimiter=delimiter)
                next(csv_reader)  # Skip header row
                for row in csv_reader:
                    if not row:
                        continue
                    dataset.append(row)
            if dataset:
                break
        except Exception as e:
            continue
    if not dataset:
        raise ValueError("Keine gültige CSV gefunden oder leerer Inhalt.")
    return dataset

# Prozente berechnen
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Euklidische Distanz berechnen
def euklDist(row1, row2):
    distance = 0.0
    for i in range(1, len(row1)):
        distance += (float(row1[i]) - float(row2[i]))**2
    return sqrt(distance)

# Ähnlichsten Nachbarn feststellen
def Nachbar_fest(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euklDist(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = [distances[i][0] for i in range(num_neighbors)]
    return neighbors

# Vorhersage mit den Nachbarn treffen
def KlassenVorSage(train, test_row, num_neighbors):
    neighbors = Nachbar_fest(train, test_row, num_neighbors)
    output_values = [row[0] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

# kNN Algorithmus
def kNN(train, test, num_neighbors):
    predictions = []
    for row in test:
        output = KlassenVorSage(train, row, num_neighbors)
        predictions.append(output)
    return predictions


train_data = laden_csv('cleaned_train.csv')
valid_data = laden_csv('cleaned_valid.csv')

# Daten prüfen und Labels (erste Spalte) als String belassen
num_neighbors = 3

predicted = kNN(train_data, valid_data, num_neighbors)
actual = [row[0] for row in valid_data]

# Zusätzliche Metriken berechnen
accuracy = accuracy_score(actual, predicted)
precision = precision_score(actual, predicted, average='weighted', zero_division=0)
recall = recall_score(actual, predicted, average='weighted', zero_division=0)
f1 = f1_score(actual, predicted, average='weighted', zero_division=0)
confusion = confusion_matrix(actual, predicted)

print("Zusätzliche Metriken:")
print("Accuracy: %.3f" % accuracy)
print("Precision: %.3f" % precision)
print("Recall: %.3f" % recall)
print("F1-Score: %.3f" % f1)
print("Confusion Matrix:")
print(confusion)

# Metriken in Textdatei speichern
with open('metrics_KNN_von_Hand.txt', 'w') as file:
    file.write(f"Accuracy: {accuracy:.3f}\n")
    file.write(f"Precision: {precision:.3f}\n")
    file.write(f"Recall: {recall:.3f}\n")
    file.write(f"F1-Score: {f1:.3f}\n")
    file.write(f"Confusion Matrix:\n{confusion}\n")

# Confusion-Matrix als PNG speichern
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.savefig("confusion_matrix_KNN_von_Hand.png", dpi=500)
plt.close()
