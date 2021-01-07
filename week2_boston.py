import numpy
import wwt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston


def accuracyClassification(X, y, cv, score, azRange=[1, 10, 200]):
    neighbors_accuracy = {}

    a, z, num = azRange
    for p in numpy.linspace(a, z, num):
        neighbor = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)
        importances = cross_val_score(neighbor, X, y, cv=cv, scoring=score)
        neighbors_accuracy[p] = importances.mean()

    max_p = max(neighbors_accuracy, key=neighbors_accuracy.get)
    max_accuracy = max(neighbors_accuracy.values())
    return [max_p, max_accuracy]


# 1
boston = load_boston()
# 2
X = scale(boston.data)
y = boston.target
# 3+4
kf = KFold(shuffle=True, random_state=42, n_splits=5)
max_p, max_accuracy = accuracyClassification(X, y, kf, score='neg_mean_squared_error')

file_name = "Src 5"
wwt.intoFile(file_name, max_p)
