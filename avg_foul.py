import pandas as pd

data = pd.read_csv("Final.csv")

data['HomeShotsOnTarget_Discrete'] = pd.qcut(data['HomeShotsOnTarget'], q=3, labels=['LowHomeShotsOnTarget', 'MediumHomeShotsOnTarget', 'HighHomeShotsOnTarget'])
data['AwayShotsOnTarget_Discrete'] = pd.qcut(data['AwayShotsOnTarget'], q=3, labels=['LowAwayShotsOnTarget', 'MediumAwayShotsOnTarget', 'HighAwayShotsOnTarget'])

data['HomeShots_Discrete'] = pd.qcut(data['HomeShots'], q=3, labels=['LowHomeShots', 'MediumHomeShots', 'HighHomeShots'])
data['AwayShots_Discrete'] = pd.qcut(data['AwayShots'], q=3, labels=['LowAwayShots', 'MediumAwayShots', 'HighAwayShots'])

data['HomeFouls_Discrete'] = pd.qcut(data['HomeFouls'], q=3, labels=['LowHomeFouls', 'MediumHomeFouls', 'HighHomeFouls'])
data['AwayFouls_Discrete'] = pd.qcut(data['AwayFouls'], q=3, labels=['LowAwayFouls', 'MediumAwayFouls', 'HighAwayFouls'])

data['HomeReds_Discrete'] = pd.qcut(data['HomeReds'], q=3, labels=['LowHomeReds', 'MediumHomeReds', 'HighHomeReds'])
data['AwayReds_Discrete'] = pd.qcut(data['AwayReds'], q=3, labels=['LowAwayReds', 'MediumAwayReds', 'HighAwayReds'])

data['Result_Discrete'] = data['Result'].map({'H': 1, 'A': -1, "D": 0})


data.to_csv("Final.csv")
