#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:27:11 2018

@author: Pablo Moreno Vera & Alejandro Bravo Fernández
"""

#Representar gráficas inline
import matplotlib.pyplot as plt
import numpy as np
#Importa el módulo de pandas
import pandas as pd

# Imports para hacer el knn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Lea el documento diabetes.csv. Recuerde que el fichero debe estar en el mismo directorio en el que esté creando el 
#notebook o bien debe indicar el path completo.
dataTraining = pd.read_csv('~/TGIM/practica/poker-hand-training-true.data',sep = ',')
dataTraining = dataTraining.rename(index=str, columns={"1": "Suit_1", "10": "Rank_1", "1.1": "Suit_2", "11": "Rank_2", "1.2": "Suit_3", "13": "Rank_3", "1.3": "Suit_4", "12": "Rank_4", "1.4": "Suit_5", "1.5": "Rank_5", "9": "Hand"})
dataTest = pd.read_csv('~/TGIM/practica/poker-hand-testing.data', sep = ',')
dataTest = dataTest.rename(index=str, columns={"1": "Suit_1", "1.1": "Rank_1", "1.2": "Suit_2", "13": "Rank_2", "2": "Suit_3", "4": "Rank_3", "2.1": "Suit_4", "3": "Rank_4", "1.3": "Suit_5", "12": "Rank_5", "0": "Hand"})
dataFrame = pd.concat([dataTraining, dataTest])
dataFrameInit = dataFrame

#%% Adecuación de la base de datos.
dataFrame = dataFrame[dataFrame.Hand.isin([0,1])]  # Escogemos las observaciones que tengan par o impar.

#%% Escogemos sólo el valor
dataFrameRank = dataFrame[['Rank_1', 'Rank_2', 'Rank_3', 'Rank_4', 'Rank_5', 'Hand']]
#dataFrame = dataFrame[['Rank_1', 'Suit_1', 'Rank_2', 'Suit_2', 'Rank_3', 'Suit_3', 'Rank_4', 'Suit_4', 'Rank_5', 'Suit_5', 'Hand']]

#%% Cambiamos los nombres del palo de la carta.
dataFrame['Suit_1'] = dataFrame['Suit_1'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})
dataFrame['Suit_2'] = dataFrame['Suit_2'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})
dataFrame['Suit_3'] = dataFrame['Suit_3'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})
dataFrame['Suit_4'] = dataFrame['Suit_4'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})
dataFrame['Suit_5'] = dataFrame['Suit_5'].map({1:'Hearts', 2:'Spades', 3:'Diamond', 4:'Clubs'})

dataFrame = pd.get_dummies(dataFrame)   #variables categóricas ----> variables numéricas

#%%Resumen de la etiqueta
dataFrame['Hand'].describe().to_csv("my_description.csv")

# Visualizaciones
dataFrame['Rank_1'].plot.hist(x = 'Age', alpha = 0.5)
plt.figure()
dataFrame['Hand'].plot.hist(x='BMI', alpha=0.5)

#%% Correlaciones
from pandas.plotting import scatter_matrix
scatter_matrix(dataFrame, alpha = 0.2, figsize = (12,12), diagonal = 'kde')

dataFrame.corr()

#%% Partición de train y test
X = dataFrameRank[['Rank_1', 'Rank_2', 'Rank_3', 'Rank_4', 'Rank_5']].values
y = dataFrame['Hand'].values
#X = dataFrame[['Rank_1', 'Suit_1_Clubs', 'Suit_1_Diamond', 'Suit_1_Hearts', 'Suit_1_Spades','Rank_2', 'Suit_2_Clubs', 'Suit_2_Diamond', 'Suit_2_Hearts', 'Suit_2_Spades','Rank_3', 'Suit_3_Clubs', 'Suit_3_Diamond', 'Suit_3_Hearts', 'Suit_3_Spades', 'Rank_4', 'Suit_4_Clubs', 'Suit_4_Diamond', 'Suit_4_Hearts', 'Suit_4_Spades', 'Rank_5', 'Suit_5_Clubs', 'Suit_5_Diamond', 'Suit_5_Hearts', 'Suit_5_Spades']].values

# KNN con el palo
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#%% 
model_KNN = KNeighborsClassifier(n_neighbors=13)
model_KNN.fit(X_train, y_train)
y_hat_KNN = model_KNN.predict(X_test)
acc = np.mean(y_test == y_hat_KNN)
print("acc:", np.mean(y_test == y_hat_KNN))

#%% Naive Bayes
from sklearn.naive_bayes import GaussianNB

model_NB = GaussianNB()

model_NB.fit(X_train, y_train)
print("theta:", model_NB.theta_)
print("sigma", model_NB.sigma_)

y_hat_NB = model_NB.predict(X_test)
print("acc:", np.mean(y_test == y_hat_NB))

#%% Redes Neuronales
from sklearn.neural_network import MLPClassifier

model_RN = MLPClassifier(max_iter=10000, alpha=1e-8, activation='relu', hidden_layer_sizes=(75,2))

model_RN.fit(X_train, y_train)
y_hat_RN = model_RN.predict(X_test)
print("acc:", np.mean(y_test == y_hat_RN))

#%% Árboles de decisión

from sklearn.tree import DecisionTreeClassifier

model_DTC = DecisionTreeClassifier()
model_DTC.fit(X_train, y_train)

y_hat_DTC = model_DTC.predict(X_test)
print("acc:", np.mean(y_test == y_hat_DTC))