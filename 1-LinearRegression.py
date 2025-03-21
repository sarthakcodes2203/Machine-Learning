##From CodeWithHarry tutorial>>>


#It is a type of Regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

diabetes=datasets.load_diabetes()

# print(diabetes.keys())
# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.data)


diabetes_X=diabetes.data                        #For using all features
# diabetes_X=diabetes.data[:,np.newaxis,2]      #For taking just one feature

# print(diabetes_X)

diabetes_X_train=diabetes_X[:-30]
diabetes_X_test=diabetes_X[-30:]

diabetes_Y_train=diabetes.target[:-30]
diabetes_Y_test=diabetes.target[-30:]

model=linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_Y_train)
diabetes_Y_predicted=model.predict(diabetes_X_test)

print("Mean squared error:",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))
print("Weights ",model.coef_)
print("Intercepts ",model.intercept_)

# plt.scatter(diabetes_X_test,diabetes_Y_test)
# plt.plot(diabetes_X_test,diabetes_Y_predicted)
# plt.show()                                        #The plots works only with one feature and not for multi feature labels

