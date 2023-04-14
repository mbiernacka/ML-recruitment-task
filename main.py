#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# Loading the dataset
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
                 names=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                        'Vertical_Distance_To_Hydrology',
                        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                        'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2',
                        'Wilderness_Area3',
                        'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
                        'Soil_Type6',
                        'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
                        'Soil_Type13',
                        'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
                        'Soil_Type20',
                        'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
                        'Soil_Type27',
                        'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',
                        'Soil_Type34',
                        'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',
                        'Cover_Type'])


print(df.head())

print(df.shape)

df_classified = df.copy()
mean = df['Elevation'].mean()


# Simple heuristic - based on 'Elevation' attribute
def classify(elevation):
    if elevation <= mean:
        return 1  # Classify as Type 1
    else:
        return 2  # Classify as Type 2


# Apply the heuristic
df_classified['class'] = df_classified['Elevation'].apply(classify)

print(df_classified['class'].value_counts())


# Split original dataframe into anttributes and class
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']


print(X.head())


print(y.head())

# Split data into training and tets sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


