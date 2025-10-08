import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('data.csv')
print("Data loaded successfully.")
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())
data = data.dropna()
print("Missing values handled.")

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 6))
sns.pairplot(data)
plt.show()
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
print("EDA completed.")

# Feature Engineering