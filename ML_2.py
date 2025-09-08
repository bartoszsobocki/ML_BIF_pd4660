# ML_2.py
#import bibliotek
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
#wrzucam dane
data = {
    'TP53_expr': [2.1, 8.5, 1.8, 6.2, 7.9, 3.1, 9.2, 2.8],
    'BRCA1_expr': [3.4, 7.2, 2.5, 6.1, 6.8, 4.0, 7.9, 3.9],
    'TF_motifs':  [2,   6,   1,   4,   5,   2,   6,   3],
    'KRAS':       [1.2, 7.1, 0.9, 6.8, 1.5, 5.5, 1.0, 6.3],
    'Cancer_status': [0, 1, 0, 1, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

X = df[['TP53_expr', 'BRCA1_expr', 'TF_motifs', 'KRAS']]
y = df['Cancer_status']

# new sample
new_sample_df = pd.DataFrame([[4.0, 5.0, 3, 2.5]], columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

pred = model.predict(new_sample_df)
print("Nowa próbka została zaklasyfikowana jako:", "nowotwór (1)" if pred[0] == 1 else "zdrowy (0)")

# Wykres, tutaj zrobię tak jak w przykładzie, ale wiadomo że model ma 4 cechy, a nie 2
plt.figure(figsize=(8,6))
plt.scatter(
    X_train['TP53_expr'], X_train['BRCA1_expr'],
    c=y_train, cmap='coolwarm', s=100, edgecolors='k', label="Dane treningowe"
)

tp53_new, brca1_new = new_sample_df[['TP53_expr', 'BRCA1_expr']].values[0]
plt.scatter(tp53_new, brca1_new, c='green', marker='X', s=200, label="Nowa próbka")

plt.xlabel("Ekspresja TP53 (TP53_expr)")
plt.ylabel("Ekspresja BRCA1 (BRCA1_expr)")
plt.title("kNN (k=3) – klasyfikacja nowego pacjenta")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("wykres.png", dpi=300)
print("Wykres zapisano jako 'wykres.png'")

# Ocena modelu
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1)
print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

#Output kodu jest właściwie idealny, więć ciężko to poprawić. Wynika to najpewniej z małej liczby próbek, dla wszystkich parametrów mam wartośći 1.00
#Powinienem mieć więcej danych, aby model był lepszy i nie było ryzyka overfittingu
#model poprawie zakwalifikował wszystkie próbki -accuracy, nie ma fałszywych alarmów- precision, wszystkie przypadki są wykryte - recall, co daje idealne f1 score - kompromis między precyzją, a czułośicą