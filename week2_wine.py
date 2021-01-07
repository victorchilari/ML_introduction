import pandas
import wwt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale


def accuracyClassification(X, y, cv, score, azRange=[1, 51]):
    neighbors_accuracy = {}

    a, z = azRange
    for n in range(a, z):
        neighbor = KNeighborsClassifier(n_neighbors=n)
        importances = cross_val_score(neighbor, X, y, cv=cv, scoring=score)
        neighbors_accuracy[n] = importances.mean()

    max_n = max(neighbors_accuracy, key=neighbors_accuracy.get)
    max_accuracy = max(neighbors_accuracy.values())
    return [max_n, max_accuracy]


# 1
data = pandas.read_csv('wine.data', header=None)
# 2
X = data.loc[:, 1:]
y = data[0]
# 3
kf = KFold(shuffle=True, random_state=42, n_splits=5)
# 4
max_n, max_accuracy = accuracyClassification(X, y, kf, score='accuracy')

file_name = "Src 1"
wwt.intoFile(file_name, max_n)
file_name = "Src 2"
wwt.intoFile(file_name, max_accuracy)

# 5+6
X_scaled = scale(X)

max_n, max_accuracy = accuracyClassification(X=X_scaled, y=y, cv=kf, score='accuracy')

file_name = "Src 3"
wwt.intoFile(file_name, max_n)
file_name = "Src 4"
wwt.intoFile(file_name, max_accuracy)
