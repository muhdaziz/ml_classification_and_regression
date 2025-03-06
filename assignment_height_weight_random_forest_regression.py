import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

data = pd.read_csv("SOCR-HeightWeight.csv")
#print(data.head())

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values
#print(X.head())
#print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
len(y_test)
rfrmodel = RandomForestRegressor(n_estimators=500, random_state=42)

rfrmodel.fit(X_train, y_train)

y_pred = rfrmodel.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)
r_squared = r2_score(y_test, y_pred)

print("mae:", mae)
print("mse:", mse)
print("rmse:", rmse)
print("r^2:", r_squared)

comparison_table = pd.DataFrame({'Actual Weight': y_test, 'Predicted Weight': y_pred})
print(comparison_table.head(20))



