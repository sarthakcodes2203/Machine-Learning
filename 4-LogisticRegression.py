#It is a type of Classification

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
iris=datasets.load_iris()

# print(list(iris.keys()))
# print(iris['data'])
# print(iris['target'])

x=iris["data"][:,3:]
y=(iris["target"]==2).astype(np.int_)
# print(x)
# print(y)

# training the classifier
clf=LogisticRegression()
clf.fit(x,y)
# example=clf.predict(([[1.6]]))
example=clf.predict(([[3.6]]))
# print(example)

# Plotting
x_new=np.linspace(0,3,1000).reshape(-1,1)
# print(x_new)
y_prob=clf.predict_proba(x_new)
# print(y_prob)

plt.plot(x_new,y_prob[:,1],"g-",label="Virginica")
plt.show()

