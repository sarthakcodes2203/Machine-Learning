from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
# mnist=fetch_openml('MNIST original')
mnist=fetch_openml('mnist_784')

x,y=mnist['data'],mnist['target']

# print(x)
# print(x)
# print(x.shape)
# print(y.shape)

digit=x[36001]
digitImage=digit.reshape(28,28)

plt.show(digitImage,cmap=plt.cm.binary,interpolation="nearest")


