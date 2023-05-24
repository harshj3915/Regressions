import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Polynomial_Regression_Data.csv')
x = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, -1:].values


poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x.reshape(-1, 1))


regressor = LinearRegression()
regressor.fit(x_poly, y)


y_pred = regressor.predict(x_poly)


plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()
