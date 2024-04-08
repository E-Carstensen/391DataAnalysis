import pandas as pd
import numpy as np

average_data = {
    'HomeShots': 12,
    'HomeShotsOnTarget': 5,
    'HomeFouls': 10,
    'AwayShots': 12,
    'AwayShotsOnTarget': 5,
    'AwayFouls': 10,
    'HomeTeam_Barcelona': 1,  # Assuming 1 indicates Barcelona and 0 otherwise
    'AwayTeam_Real Madrid': 1  # Assuming 1 indicates Real Madrid and 0 otherwise
}
df = pd.read_csv("Final.csv")

print(['HomeShots', 'HomeShotsOnTarget', 'HomeFouls', 'AwayShots', 'AwayShotsOnTarget', 'AwayFouls','HomeCorners', 'AwayCorners', 'HomeYellows', 'AwayYellows', 'HomeReds', 'AwayReds'])

print("Barcelona")


print(df[df["HomeTeam"] == "Barcelona"]["HomeShotsOnTarget"].mean())
print(df[df["HomeTeam"] == "Barcelona"]["HomeShots"].mean())
print(df[df["AwayTeam"] == "Barcelona"]["AwayShotsOnTarget"].mean())
print(df[df["AwayTeam"] == "Barcelona"]["AwayShots"].mean())
print(df[df["HomeTeam"] == "Barcelona"]["HomeReds"].mean())
print(df[df["AwayTeam"] == "Barcelona"]["AwayReds"].mean())



print("Real Madrid")


print(df[df["HomeTeam"] == "Real Madrid"]["HomeShotsOnTarget"].mean())
print(df[df["AwayTeam"] == "Real Madrid"]["AwayShotsOnTarget"].mean())
print(df[df["HomeTeam"] == "Real Madrid"]["HomeReds"].mean())
print(df[df["AwayTeam"] == "Real Madrid"]["AwayReds"].mean())




"""Barcelona
7.257703081232493
5.811797752808989
0.08403361344537816
0.10955056179775281
Real Madrid
7.617977528089888
5.8879551820728295
0.10955056179775281
0.1568627450980392"""

predicted_outcome = model.predict(average_data_df)

# Print the predicted outcome
if predicted_outcome == 1:
    print("Predicted outcome: Barcelona wins")
elif predicted_outcome == 0:
    print("Predicted outcome: Draw")
else:
    print("Predicted outcome: Real Madrid wins")