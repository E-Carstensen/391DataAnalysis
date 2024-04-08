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
pd.set_option('display.max_columns', None)  # Show all columns
#pd.set_option('display.max_rows', None)
#df_encoded = pd.get_dummies(df, columns=['HomeTeam', 'AwayTeam'])
#X = df.drop(columns=['HomeTeam', 'AwayTeam','Index', 'Div', 'Date', 'Time', 'HomeGoals', 'AwayGoals', 'Result', 'HalfTimeHG', 'HalfTimeAG', 'HalfTimeResult', 'HomeCorners', 'AwayCorners', 'HomeYellows', 'AwayYellows', 'HomeReds', 'AwayReds', 'HomeShotsOnTarget_Discrete', 'AwayShotsOnTarget_Discrete', 'HomeShots_Discrete', 'AwayShots_Discrete', 'HomeFouls_Discrete', 'AwayFouls_Discrete', 'Result_Discrete', 'Result_Encoded'])
# Define features and target variable
X = df[['HomeShotsOnTarget', 'HomeShots', 'HomeFouls', 'HomeCorners', 'HomeYellows', 'HomeReds', 
        'AwayShotsOnTarget', 'AwayShots', 'AwayFouls', 'AwayCorners', 'AwayYellows', 'AwayReds']]
y = df['Result_Discrete']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(['HomeShots', 'HomeShotsOnTarget', 'HomeFouls', 'AwayShots', 'AwayShotsOnTarget', 'AwayFouls','HomeCorners', 'AwayCorners', 'HomeYellows', 'AwayYellows', 'HomeReds', 'AwayReds'])
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


real_madrid_avg = {
    'HomeShotsOnTarget': df[df["HomeTeam"] == "Real Madrid"]["HomeShotsOnTarget"].mean(),
    'HomeShots': df[df["HomeTeam"] == "Real Madrid"]["HomeShots"].mean(),
    'HomeFouls': df[df["HomeTeam"] == "Real Madrid"]["HomeFouls"].mean(),
    'HomeCorners': df[df["HomeTeam"] == "Real Madrid"]["HomeCorners"].mean(),
    'HomeYellows': df[df["HomeTeam"] == "Real Madrid"]["HomeYellows"].mean(),
    'HomeReds': df[df["HomeTeam"] == "Real Madrid"]["HomeReds"].mean(),
    'AwayShotsOnTarget': df[df["AwayTeam"] == "Real Madrid"]["AwayShotsOnTarget"].mean(),
    'AwayShots': df[df["AwayTeam"] == "Real Madrid"]["AwayShots"].mean(),
    'AwayFouls': df[df["AwayTeam"] == "Real Madrid"]["AwayFouls"].mean(),
    'AwayCorners': df[df["AwayTeam"] == "Real Madrid"]["AwayCorners"].mean(),
    'AwayYellows': df[df["AwayTeam"] == "Real Madrid"]["AwayYellows"].mean(),
    'AwayReds': df[df["AwayTeam"] == "Real Madrid"]["AwayReds"].mean()
}

barcelona_avg = {
    'HomeShotsOnTarget': df[df["HomeTeam"] == "Barcelona"]["HomeShotsOnTarget"].mean(),
    'HomeShots': df[df["HomeTeam"] == "Barcelona"]["HomeShots"].mean(),
    'HomeFouls': df[df["HomeTeam"] == "Barcelona"]["HomeFouls"].mean(),
    'HomeCorners': df[df["HomeTeam"] == "Barcelona"]["HomeCorners"].mean(),
    'HomeYellows': df[df["HomeTeam"] == "Barcelona"]["HomeYellows"].mean(),
    'HomeReds': df[df["HomeTeam"] == "Barcelona"]["HomeReds"].mean(),
    'AwayShotsOnTarget': df[df["AwayTeam"] == "Barcelona"]["AwayShotsOnTarget"].mean(),
    'AwayShots': df[df["AwayTeam"] == "Barcelona"]["AwayShots"].mean(),
    'AwayFouls': df[df["AwayTeam"] == "Barcelona"]["AwayFouls"].mean(),
    'AwayCorners': df[df["AwayTeam"] == "Barcelona"]["AwayCorners"].mean(),
    'AwayYellows': df[df["AwayTeam"] == "Barcelona"]["AwayYellows"].mean(),
    'AwayReds': df[df["AwayTeam"] == "Barcelona"]["AwayReds"].mean()
}
average_data = {
    'HomeShotsOnTarget': barcelona_avg["HomeShotsOnTarget"],
    'HomeShots': barcelona_avg["HomeShots"],
    'HomeFouls': barcelona_avg["HomeFouls"],
    'HomeCorners': barcelona_avg["HomeCorners"],
    'HomeYellows': barcelona_avg["HomeYellows"],
    'HomeReds': barcelona_avg["HomeReds"],
    'AwayShotsOnTarget': real_madrid_avg["AwayShotsOnTarget"],
    'AwayShots': real_madrid_avg["AwayShots"],
    'AwayFouls': real_madrid_avg["AwayFouls"],
    'AwayCorners': real_madrid_avg["AwayCorners"],
    'AwayYellows': real_madrid_avg["AwayYellows"],
    'AwayReds': real_madrid_avg["AwayReds"]
}
average_data2 = {
    'HomeShotsOnTarget': real_madrid_avg["HomeShotsOnTarget"],
    'HomeShots': real_madrid_avg["HomeShots"],
    'HomeFouls': real_madrid_avg["HomeFouls"],
    'HomeCorners': real_madrid_avg["HomeCorners"],
    'HomeYellows': real_madrid_avg["HomeYellows"],
    'HomeReds': real_madrid_avg["HomeReds"],
    'AwayShotsOnTarget': barcelona_avg["AwayShotsOnTarget"],
    'AwayShots': barcelona_avg["AwayShots"],
    'AwayFouls': barcelona_avg["AwayFouls"],
    'AwayCorners': barcelona_avg["AwayCorners"],
    'AwayYellows': barcelona_avg["AwayYellows"],
    'AwayReds': barcelona_avg["AwayReds"]
}
average_data_df = pd.DataFrame([average_data])
average_data_df2 = pd.DataFrame([average_data2])
predicted_outcome = model.predict(average_data_df)
predicted_outcome2 = model.predict(average_data_df2)

# Print the predicted outcome
if predicted_outcome == 1:
    print("Predicted outcome: Barcelona wins")
elif predicted_outcome == 0:
    print("Predicted outcome: Draw")
else:
    print("Predicted outcome: Real Madrid wins")


if predicted_outcome2 == 1:
    print("Predicted outcome: Barcelona wins")
elif predicted_outcome2 == 0:
    print("Predicted outcome: Draw")
else:
    print("Predicted outcome: Real Madrid wins")