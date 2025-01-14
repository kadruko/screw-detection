import csv
from math import exp, pi, sqrt

import matplotlib.pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)


# Funktion zum Laden der CSV-Datei
def load_csv(file_path, delimiter=',', has_headers=True):
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file, delimiter=delimiter)
        if has_headers:
            next(csv_reader)  # Überspringt die Header-Zeile
        dataset = []
        for row in csv_reader:
            dataset.append(row)
    return dataset

# Konvertiere Labels zu numerischen Werten
def encode_labels(dataset):
    return {label: idx for idx, label in enumerate(set(row[0] for row in dataset))}

# Apply label mapping
def apply_label_mapping(dataset, label_mapping):
    for row in dataset:
        row[0] = label_mapping[row[0]]
    return dataset

# Konvertiere Merkmalswerte zu float
def convert_features_to_float(dataset):
    for row in dataset:
        for i in range(1, len(row)):  # Überspringe die Label-Spalte
            row[i] = float(row[i].replace('.', '.'))  # Konvertiere ',' zu '.' für Dezimalzahlen
    return dataset

# Trenne das Dataset nach Klassen
def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        label = row[0]
        if label not in separated:
            separated[label] = []
        separated[label].append(row[1:])  # Ohne Label-Spalte
    return separated

# Berechne Mittelwert und Standardabweichung
def mean(numbers):
    return sum(numbers) / len(numbers)

def stdev(numbers):
    avg = mean(numbers)
    variance = sum((x - avg) ** 2 for x in numbers) / (len(numbers) - 1)
    return sqrt(variance)

# Zusammenfassen nach Klassen
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = {}
    for class_value, rows in separated.items():
        summaries[class_value] = [(mean(column), stdev(column)) for column in zip(*rows)]
    return summaries

# Berechne Wahrscheinlichkeit
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# Klassenwahrscheinlichkeiten berechnen
def calculate_class_probabilities(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            probabilities[class_value] *= calculate_probability(input_vector[i], mean, stdev)
    return probabilities

# Vorhersage treffen
def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_label = class_value
            best_prob = probability
    return best_label

# Vorhersagen für das Testset
def get_predictions(summaries, test_set):
    predictions = []
    for row in test_set:
        input_vector = row[1:]  # Ohne Label
        result = predict(summaries, input_vector)
        predictions.append(result)
    return predictions

# Berechne Metriken
def calculate_metrics(true_labels, predicted_labels):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=1)
    confusion = confusion_matrix(true_labels, predicted_labels)
    return accuracy, precision, recall, f1, confusion

# Hauptfunktion für Training und Test
def naive_bayes():
    # Lade und verarbeite die Daten
    train_data = load_csv('cleaned_train.csv', delimiter=',')
    train_data = convert_features_to_float(train_data)
    
    valid_data = load_csv('cleaned_valid.csv', delimiter=',')
    valid_data = convert_features_to_float(valid_data)

    label_mapping = encode_labels(train_data)
    
    train_data = apply_label_mapping(train_data, label_mapping)
    valid_data = apply_label_mapping(valid_data, label_mapping)

    # Zusammenfassen der Trainingsdaten
    summaries = summarize_by_class(train_data)

    # Vorhersagen
    predictions = get_predictions(summaries, valid_data)

    # Labels extrahieren
    true_labels = [row[0] for row in valid_data]

    # Metriken berechnen
    accuracy, precision, recall, f1, confusion = calculate_metrics(true_labels, predictions)

    # Ergebnisse anzeigen
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(confusion)
    print("Label Mapping:", label_mapping)
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig('confusion_matrix_Bayes_von_Hand.png', dpi = 500)  # Confusion Matrix speichern
    plt.show()

    # Metriken speichern
    metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1}

    # Metriken in eine Textdatei speichern
    with open('metrics_Bayes_von_Hand.txt', 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print("Modell und Metriken erfolgreich gespeichert.")

# Beispielnutzung
naive_bayes()


