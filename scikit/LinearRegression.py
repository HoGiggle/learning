# -*- coding: utf-8 -*-
from sklearn import datasets, linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# load data
diabetes = datasets.load_diabetes()

# one feature
x = diabetes.data[:, np.newaxis, 2]
x_train = x[:-20]
x_test = x[-20:]

# label data
y_train = diabetes.target[:-20]
y_test = diabetes.target[-20:]

# create model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print("Coefficients: ", model.coef_)

print("MSE: %.2f"
      % mean_squared_error(y_test, y_predict))
print("Variance score: %.2f" % r2_score(y_test, y_predict))

# plot output
plt.figure(figsize=(20,20), dpi=80)
plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, y_predict, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()