import pandas as pd
from sklearn.model_selection import train_test_split

# Wczytaj plik CSV
df = pd.read_csv("dane_projekt1.csv")

# Wyświetl podstawowe statystyki
print("Statystyki danych:")
print(df.describe())
print("\nRozmiar całego zbioru:", df.shape)

# Przypisujemy cechy X i etykiety y
X = df.drop(columns=["Gene_Function"])
y = df["Gene_Function"]

# Dzielimy dane na 70% treningowe + walidacyjne, 30% testowe
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Dzielimy dane treningowe na 70% trening + 30% walidacja 
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.3, stratify=y_temp, random_state=42)

# Wyświetl rozmiary podzbiorów
print("\nRozmiary zbiorów:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

# Opcjonalnie: wyświetl pierwsze 5 wierszy każdego zbioru
print("\nPrzykład X_train:")
print(X_train.head())
print("\nPrzykład y_train:")
print(y_train.head())