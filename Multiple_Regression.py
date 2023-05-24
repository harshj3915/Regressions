import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

dataset = pd.read_csv("Multiple_Regression_Data.csv")
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1:].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred=regressor.predict(x_test)

print(f'Score of this train data: {regressor.score(x_train,y_train)}')
plt.scatter(y_test,y_pred, c='blue',s=0.3)





plt.show()

