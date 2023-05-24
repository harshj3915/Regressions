import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Logistic_Regression_Data.csv')
x = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1:].values

regressor = LogisticRegression()
regressor.fit(x, y)

y_pred = regressor.predict(x)

plt.scatter(x[:, 0], x[:, 1], c=y)
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = regressor.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.show()
