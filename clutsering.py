import pandas as pd
from sklearn.cluster import AgglomerativeClustering


df = pd.read_csv('Final.csv')
#data = data[[ 'HomeFouls_Discrete', 'AwayFouls_Discrete','Result','HomeTeam','AwayTeam', 'HomeShotsOnTarget_Discrete','AwayShotsOnTarget_Discrete', 'Time']]


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np


df = df.replace('NaN', np.nan)

df = df.dropna()

# Define features and target variable
X = df[['HomeShots', 'HomeShotsOnTarget', 'HomeFouls', 'AwayShots', 'AwayShotsOnTarget', 'AwayFouls',"HomeTeam", "AwayTeam"]]  # Features
y = df['Result_Discrete']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Display coefficients
print('Coefficients:', model.coef_)
