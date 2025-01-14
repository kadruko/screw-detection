from random import randrange
from csv import reader
from math import sqrt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# Datenset in 3 Teile für die Kreuzvalidierung aufteilen
def split_dataset(dataset):
    dataset_copy = list(dataset)
    split = [[], [], []]
    while dataset_copy:
        for part in split:
            if dataset_copy:
                index = randrange(len(dataset_copy))
                part.append(dataset_copy.pop(index))
    return split

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

# Kreuzvalidierung

def Genauigkeit(dataset, algorithm, num_neighbors):
    splits = split_dataset(dataset)
    scores = []
    all_actual = []
    all_predicted = []
    for i in range(len(splits)):
        test_set = splits[i]
        train_set = [item for s in splits if s != splits[i] for item in s]
        predicted = algorithm(train_set, test_set, num_neighbors)
        actual = [row[0] for row in test_set]
        scores.append(accuracy_metric(actual, predicted))
        all_actual.extend(actual)
        all_predicted.extend(predicted)
    return scores, all_actual, all_predicted


filename = (r'C:\Daten_JD\HKA\VDKI\projekt\Versuch_Knn\train.csv')
dataset = laden_csv(filename)

# Daten prüfen und Labels (erste Spalte) als String belassen
num_neighbors = 3

scores, all_actual, all_predicted = Genauigkeit(dataset, kNN, num_neighbors)

print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

# Zusätzliche Metriken berechnen
accuracy = accuracy_score(all_actual, all_predicted)
precision = precision_score(all_actual, all_predicted, average='weighted', zero_division=0)
recall = recall_score(all_actual, all_predicted, average='weighted', zero_division=0)
f1 = f1_score(all_actual, all_predicted, average='weighted', zero_division=0)
confusion = confusion_matrix(all_actual, all_predicted)

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
