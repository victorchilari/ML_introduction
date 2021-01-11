import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer

# Опорные объекты

# Какие 10 слов имеют наибольший по модулю вес?


# 1
newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)
# 2
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
feature_map = vectorizer.get_feature_names()
# 3
grid = {'C': np.power(10.0, np.arange(-5, 5))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)
# 4
C = gs.best_params_['C']
clf = SVC(C=C, kernel='linear', random_state=241)
clf.fit(X, y)
# 5
weights = np.absolute(clf.coef_.toarray())

max_weights = sorted(zip(weights[0], feature_map))[-10:]
max_weights.sort(key=lambda x: x[1])
print(max_weights)

f = open("Src 2.txt", "w")
for w, c in max_weights[:-1]:
    f.write(c)
    f.write(' ')
f.write(max_weights[-1][1])
f.close()
