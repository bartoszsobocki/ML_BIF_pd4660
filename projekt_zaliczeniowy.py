#pd4660
#RNAseq projekt zaliczeniowy
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#1. Wczytuję
df = pd.read_csv("dane_projekt.csv")
#pierwsze 5 wierszy - patrzę
print(df.head())
#podstawowe rozmiary macierzy
print("\nInformacje o danych:")
print(df.info())
#NAs
print("\nBraki danych w kolumnach:")
print(df.isna().sum())
#klasy genów
print("\nRozkład klas w kolumnie Gene_Function:")
print(df["Gene_Function"].value_counts())
#2. Eksploruję dane i przetwarzanie
feature_cols = [c for c in df.columns if c not in ["Gene_ID", "Gene_Function"]]
print("\nStatystyki (log1p cech):")
print(np.log1p(df[feature_cols]).describe().T.round(3).head(10))
#widzę, że dane są zasadniczo zbilansowane, poza tym są to wartości TPM według instrukcji, więc nie ma potrzeby logarytimzacji ani z-score standaryzacji
#sam format danych też mi odpowiada, więc nic zasadniczo nie przekształcam
#żeby nie było zbyt ubogo mimo wszystko, to zrobię graph ile klas genów jest w naszej grupie
target_col = "Gene_Function"
class_counts = df[target_col].value_counts()
print("\nRozkład klas:")
print(class_counts)

plt.figure(figsize=(5,3))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Rozkład klas")
plt.ylabel("Liczebność")
plt.xlabel("Klasa")
plt.tight_layout()
plt.show()
#przygotowuję strukturę danych
X = df.drop(columns=["Gene_ID", "Gene_Function"])
y = df["Gene_Function"]
#3. Podział na zbiory treningowy i testowy, wybieram klasycznie 0.2 test (80% trening)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
#Budowa modelu ML
mlp = MLPClassifier(
    hidden_layer_sizes=(50, 25, 10),   
    activation='relu',
    max_iter=5000,
    random_state=42
)
mlp.fit(X_train, y_train)
#Preykcja, ewolucja modelu i interpretacja wyników
y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", round(accuracy, 2))
print("Precision (macro):", round(precision, 2))
print("Recall (macro):", round(recall, 2))
print("F1-score (macro):", round(f1, 2))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix,
            annot=True,
            fmt='d',
            cmap='Purples',
            xticklabels=y.unique(),
            yticklabels=y.unique())
plt.xlabel('Przewidziana klasa')
plt.ylabel('Rzeczywista klasa')
plt.title('Macierz pomyłek - Sieć neuronowa MLP')
plt.tight_layout()
plt.show()
#stworzyłem model do rozpoznawania klas funkcji i zaskoczyło mnie jak zły on jest. Wybrałem MLP z ciekawości, ponieważ na etapie zadań treningowych dawał on aż za idealne rezultaty i chciałem sprawdzić co się stanie jak dam trudniejsze zadanie
#prawdopodobnie wynika to z małej liczby próbek na klasę i nie uda mi się dostać i tak dobrego modelu (raczej będzie overfitting), ale zobaczę co się stanie jak pozmieniam nieco parametry
##to wyniki dla 1000 iteracji, 2  warstwy ukryte (16,8), test size 0.2
#          precision    recall  f1-score   support
#
 #             enzyme       0.00      0.00      0.00         2
  #          receptor       0.33      0.33      0.33         3
  #structural_protein       1.00      0.50      0.67         2
#transcription_factor       0.50      0.67      0.57         3

 #           accuracy                           0.40        10
  #         macro avg       0.46      0.38      0.39        10
  #      weighted avg       0.45      0.40      0.40        10
  #zmieniłem liczbę warstw na 3 (50,25,10) i utrzymałem 1000 iteracji, test size 0.2
#             precision    recall  f1-score   support
#
 #             enzyme       0.33      0.50      0.40         2
  #          receptor       0.00      0.00      0.00         3
  #structural_protein       0.50      0.50      0.50         2
#transcription_factor       0.50      0.67      0.57         3

 #           accuracy                           0.40        10
  #         macro avg       0.33      0.42      0.37        10
   #     weighted avg       0.32      0.40      0.35        10
#nadal źle, więc idzie w ruch liczba iteracji do 5000, co w sumie niewiele dało
# Widzę kilka przyczyn, podstawowym problemem jest mała liczba próbek na klase i zbiór testowy
# Co mogę zrobić, żeby poprawić model:
#Rzecz jasna zwiększyć zbiór danych
# Lepiej znormalizować dane zaczynając od rawcounts, TPM może nie być najlepszym wyborem
# Zmniejszyć liczbę warstw i neuronów, żeby nie przeuczyć modelu
# Dodać walidację krzyżową
# Może prosty RandomForest albo inny model byłby lepszy, dając jednocześnie ważność cech
#Dziękuję za cenną lekcję, czuję się zachęcony żeby dalej rozwijać wiedzę w obrębie ML, spróbuję na swoich danych z RNAseq



