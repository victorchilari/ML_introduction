import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)
importances = clf.feature_importances_
print(importances)

pr = clf.predict_proba([[3, 3]])
print(pr)

X, y = load_iris(return_X_y=True)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

asd = tree.plot_tree(clf)
print(asd)
