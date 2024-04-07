from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

timeSlot = { 
            '11:00' : 'Morning',
            '12:00' : 'Morning',
            '13:00' : 'Afternoon',
            '13:15': 'Afternoon',
            '13:30': 'Afternoon',
            '13:45': 'Afternoon',
            '14:00': 'Afternoon',
            '14:15': 'Afternoon',
            '14:30': 'Afternoon',
            '14:45': 'Afternoon',
            '15:00': 'Afternoon',
            '15:15': 'Afternoon',
            '15:30': 'Afternoon',
            '15:45': 'Afternoon',
            '16:00': 'Afternoon',
            '16:15': 'Afternoon',
            '16:30': 'Afternoon',
            '16:45': 'Afternoon',
            '17:00': 'Evening',
            '17:15': 'Evening',
            '17:30': 'Evening',
            '17:45': 'Evening',
            '18:00': 'Evening',
            '18:15': 'Evening',
            '18:30': 'Evening',
            '18:45': 'Evening',
            '19:00': 'Evening',
            '19:15': 'Evening',
            '19:30': 'Evening',
            '19:45': 'Evening',
            '20:00': 'Night',
            '20:15': 'Night',
            '20:30': 'Night',
            '20:45': 'Night',
            '21:00': 'Night',
            '21:15': 'Night',
            '21:30': 'Night',
            '21:45': 'Night'
            }
default = "UKN"

def set_time(time):
    return timeSlot.get(time, default)

def set_result(home_goals, away_goals):
    if home_goals > away_goals:
        return 'Win >2' if (home_goals - away_goals) > 2 else 'Win <2'
    elif home_goals < away_goals:
        return 'Loss'
    else:
        return 'Draw'

def main():
    matches = pd.read_csv("simplified.csv")

    for index, row in matches.iterrows():
     matches.loc[index, "Time"] = set_time(row["Time"])
     matches.loc[index, "Result_Encoded"] = set_result(int(row["HomeGoals"]), int(row["AwayGoals"]))

    matches.to_csv("Final.csv")

if __name__ == '__main__':
    main()