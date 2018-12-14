#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:27:11 2018

@author: Pablo Moreno Vera & Alejandro Bravo Fernández
"""

# Imports.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, validation_curve
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import scatter_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Carga de las bases de datos.
dataTraining = pd.read_csv('~/TGIM/practica/poker-hand-training-true.data',sep = ',')
dataTraining = dataTraining.rename(index=str, columns={"1": "Suit_1", "10": "Rank_1", "1.1": "Suit_2", "11": "Rank_2", "1.2": "Suit_3", "13": "Rank_3", "1.3": "Suit_4", "12": "Rank_4", "1.4": "Suit_5", "1.5": "Rank_5", "9": "Hand"})
dataTest = pd.read_csv('~/TGIM/practica/poker-hand-testing.data', sep = ',')
dataTest = dataTest.rename(index=str, columns={"1": "Suit_1", "1.1": "Rank_1", "1.2": "Suit_2", "13": "Rank_2", "2": "Suit_3", "4": "Rank_3", "2.1": "Suit_4", "3": "Rank_4", "1.3": "Suit_5", "12": "Rank_5", "0": "Hand"})

# Unión de las bases de datos.
dataFrame = pd.concat([dataTraining, dataTest])

dataFrameInit = dataFrame       # Base de datos original.

#%% Adecuación de la base de datos.

# Selección de las clases a utilizar.
dataFrame = dataFrame[dataFrame.Hand.isin([0,1,2,3,4,6])]

# Balanceo de la base de datos.
dataFrameBalanced = dataFrame.groupby('Hand')
dataFrameBalanced = dataFrameBalanced.apply(lambda x: x.sample(dataFrameBalanced.size().min())).reset_index(drop=True)

#%% One-Hot-Encoder
dataFrame['Suit_1'] = dataFrame['Suit_1'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})
dataFrame['Suit_2'] = dataFrame['Suit_2'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})
dataFrame['Suit_3'] = dataFrame['Suit_3'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})
dataFrame['Suit_4'] = dataFrame['Suit_4'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})
dataFrame['Suit_5'] = dataFrame['Suit_5'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})

dataFrame = pd.get_dummies(dataFrame) #variables categóricas ----> variables numéricas

#%% Selección de las características predictivas a utilizar.
dataFrameRank = dataFrameBalanced[['Rank_1', 'Rank_2', 'Rank_3', 'Rank_4', 'Rank_5', 'Hand']]

#%%Resumen de la base de datos.

# Probabilidades a priori de la base de datos original.
dataFrameInit['Hand'].value_counts().div(len(dataFrameInit))
# Probabilidades a priori de la base de datos a utilizar.
dataFrameRank['Hand'].value_counts().div(len(dataFrameRank))

# Descripción de la base de datos a utilizar.
dataFrameRank.describe().to_csv("my_description.csv")

# Visualizaciones de los histogramas
dataFrameRank.hist()

#% Correlaciones de las características originales con la salida.
scatter_matrix(dataFrameInit, alpha = 0.2, figsize = (12,12), diagonal = 'kde')
#%% Partición del dataset.

# Establecimiento de los dataFrames de características predictivas y del atributo objetivo.
X = dataFrameRank[['Rank_1', 'Rank_2', 'Rank_3', 'Rank_4', 'Rank_5']].values
y = dataFrameRank['Hand'].values

# Normalizamos los valores del dataFrame de características predictivas.
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Hacemos la partición del dataFrame de Training y del dataFrame de Test.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#%% KNN
model_KNN = KNeighborsClassifier(n_neighbors=3)
model_KNN.fit(X_train, y_train)

print("KNN-Train accuracy:", model_KNN.score(X_train, y_train))
print("KNN-Test accuracy:", model_KNN.score(X_test, y_test))

#%% Naive Bayes
model_NB = GaussianNB()
model_NB.fit(X_train, y_train)

print("NB-Train accuracy:", model_NB.score(X_train, y_train))
print("NB-Test accuracy:", model_NB.score(X_test, y_test))

#%% Redes Neuronales
model_RN = MLPClassifier(max_iter=10000, alpha=1e-8, activation='relu')
model_RN.fit(X_train, y_train)

print("RN-Train accuracy:", model_RN.score(X_train, y_train))
print("RN-Test accuracy:", model_RN.score(X_test, y_test))

#%% Árboles de decisión
model_DTC = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
model_DTC.fit(X_train, y_train)

print("DTC-Train accuracy:", model_DTC.score(X_train, y_train))
print("DTC-Test accuracy:", model_DTC.score(X_test, y_test))

# %% KNN - Cross-validation
n_neighbors = np.arange(3,61,2)

cv = KFold(n_splits=3, random_state=16)
model_KNNc = KNeighborsClassifier()

train_scores, test_scores = validation_curve(model_KNNc, X_train, y_train, param_name="n_neighbors", param_range=n_neighbors, cv=cv, scoring="accuracy", n_jobs=-1)

# Visualización del accuracy del cross-validation.
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.xlabel("$\gamma$")
plt.ylabel("Score")
lw = 2

plt.plot(n_neighbors, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(n_neighbors, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.plot(n_neighbors, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(n_neighbors, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")

