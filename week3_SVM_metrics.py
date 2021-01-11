import pandas
from sklearn.svm import SVC

# Опорные объекты

# Какие объекты SVM выберет в качестве опорных?

data1 = pandas.read_csv('svm-data.csv', header=None)
data = pandas.DataFrame(data1)
X = data.loc[:, 1:]
y = data[0]

clf = SVC(C=100000, kernel='linear', random_state=241)
clf.fit(X, y)

vectors = clf.support_vectors_
print(vectors)

rez = clf.support_
rez = map(lambda x: x + 1, rez)
rez = list(rez)
print(rez)

f = open("Src 1.txt", "w")
f.write(str(rez))
f.close()

# The correct answer is without [], with space between
