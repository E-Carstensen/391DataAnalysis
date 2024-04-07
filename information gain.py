from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read the data from CSV
data = pd.read_csv("Final.csv")

#data = data[data["HomeTeam"] == "Barcelona"]

# Encode the result (win/loss) as categorical (adjust if needed)
data['Result_Encoded'] = data['Result'].map({'H': 1, 'A': -1, "D": 0})

print(data.head(10))
data = data.head(1000)

if data.empty:
    print("Error: The DataFrame is empty. Please load data before splitting.")
cols = ['HomeShots', 'HomeShotsOnTarget', 'HomeFouls', 'AwayShots', 'AwayShotsOnTarget', 'AwayFouls']
# Separate features (X) and target variable (y)
X = data[['HomeShots', 'HomeShotsOnTarget', 'HomeFouls', 'AwayShots', 'AwayShotsOnTarget', 'AwayFouls']]
y = data['Result_Encoded']

# Calculate Information Gain for each feature
information_gain = mutual_info_classif(X, y)

# Print feature names and Information Gain scores
feature_names = data.columns.tolist()
for i, name in enumerate(cols):
    print(f"Feature: {name}, Information Gain: {information_gain[i]:.4f}")

# Sort features by Information Gain (optional)
sorted_features = sorted(zip(feature_names, information_gain), key=lambda x: x[1], reverse=True)
print("\nFeatures sorted by Information Gain (descending):")
for name, gain in sorted_features:
    print(f"{name}: {gain:.4f}")



