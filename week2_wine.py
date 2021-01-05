import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale


data = pandas.read_csv('wine.data', header=None)
X = data.loc[:,1:]
y = data[0]

kf = KFold(shuffle=True, random_state=42, n_splits=5)

neighbors_accuracy = {}

for n in range(50):
    neighbor = KNeighborsClassifier(n_neighbors = n+1)
    importances = cross_val_score(neighbor,X,y, cv = kf, scoring='accuracy')
    neighbors_accuracy[n+1] = importances.mean()

max_accuracy = max(neighbors_accuracy.values())
max_n = max(neighbors_accuracy, key=neighbors_accuracy.get)
print(max_n, max_accuracy)

