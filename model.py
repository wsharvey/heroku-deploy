import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

dataset = pd.read_csv('Seasons_Stats.csv')
dataset = dataset.loc[:, ['Age', 'Pos', 'FG%', 'FT%', '3P%', 'PTS']].dropna()

def convert_to_int(word):
    pos = word.split("-")[0]
    word_dict = {'PG': 1, 'SG': 2, 'SF': 3, 'PF': 4, 'C': 5}
    return word_dict[pos]

X = dataset.iloc[:, range(0, 5)]
X['Pos'] = X['Pos'].apply(lambda x: convert_to_int(x))
X['FG%'] = X['FG%'] * 100
X['FT%'] = X['FT%'] * 100
X['3P%'] = X['3P%'] * 100
y = dataset.iloc[:, 5]

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))




