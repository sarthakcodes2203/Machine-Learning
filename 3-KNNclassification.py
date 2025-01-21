#It is a type of Classification

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier


# loading datasets
iris=datasets.load_iris()
# print(iris.DESCR)

# printing features and labels
features=iris.data
labels=iris.target
# print(features[0],labels[0])

# training the classifier
clf=KNeighborsClassifier()
clf.fit(features,labels)

preds=clf.predict([[1,1,1,1]])
print(preds)


