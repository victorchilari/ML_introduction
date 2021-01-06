import pandas
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def intoFile(name, content, sec_content=""):
    # round float
    content = content if not isinstance(content, float) else round(content, 2)
    sec_content = sec_content if not isinstance(sec_content, float) else round(sec_content, 2)

    f = open(name + ".txt", "w")
    if sec_content == "":
        text = content
    else:
        text = str(content) + " " + str(sec_content)
    f.write(str(text))
    f.close()


def printFile(name):
    f = open(name + ".txt", "r")
    print(name + "\n", f.read())

def classificationAccuracy(X, y, cv, score, azRange=[1,51]):
    neighbors_accuracy = {}

    a, z = azRange
    for n in range(a, z):
        neighbor = KNeighborsClassifier(n_neighbors=n)
        importances = cross_val_score(neighbor, X, y, cv=cv, scoring=score)
        neighbors_accuracy[n] = importances.mean()

    max_n = max(neighbors_accuracy, key=neighbors_accuracy.get)
    max_accuracy = max(neighbors_accuracy.values())
    return  [max_n,max_accuracy]


#1
data = pandas.read_csv('wine.data', header=None)
#2
X = data.loc[:,1:]
y = data[0]
#3
kf = KFold(shuffle=True, random_state=42, n_splits=5)
#4
max_n, max_accuracy = classificationAccuracy(X,y,kf,score='accuracy')
file_name = "Src 1"
intoFile(file_name, max_n)
file_name = "Src 2"
intoFile(file_name, max_accuracy)

#5
X_scaled = sklearn.preprocessing.scale(X)
'''
neighbors_accuracy_scaled = {}
for n in range(1, 51):
    neighbor = KNeighborsClassifier(n_neighbors = n)
    importances = cross_val_score(neighbor, X_scaled, y, cv = kf, scoring='accuracy')
    neighbors_accuracy_scaled[n] = importances.mean()
#6
max_n = max(neighbors_accuracy_scaled, key=neighbors_accuracy_scaled.get)
max_accuracy = max(neighbors_accuracy_scaled.values())
'''
max_n, max_accuracy = classificationAccuracy(X= X_scaled, y= y, cv= kf, score= 'accuracy')

file_name = "Src 3"
intoFile(file_name, max_n)
file_name = "Src 4"
intoFile(file_name, max_accuracy)