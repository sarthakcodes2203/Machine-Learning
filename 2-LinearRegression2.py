#It is a type of Regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()


diabetes_X=np.array([[1],[2],[3]])                       #using sample data

# print(diabetes_X)

diabetes_X_train=diabetes_X
diabetes_X_test=diabetes_X

diabetes_Y_train=np.array([3,2,4])
diabetes_Y_test=np.array([3,2,4])

model=linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predicted=model.predict(diabetes_X_test)

print("Mean squared error:",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))
print("Weights ",model.coef_)
print("Intercepts ",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show() 

