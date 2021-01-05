import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


WORK_COLUMNS = ['Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 'Flavanoids',
                'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity', 'Hue', 'OD280_OD315OfDilutedWines',
                'Proline']

data = pandas.read_table('wine.data', sep=',', names=['Class', *WORK_COLUMNS])

wineClass = pandas.DataFrame(data, columns=['Class'])
signs = pandas.DataFrame(data, columns=WORK_COLUMNS)

X = wineClass
y = signs

kf = KFold(shuffle=True, random_state=42, n_splits=5)
knn = KNeighborsClassifier()
knn.fit(X, y)
# knn.n_neighbors
# importances = cross_val_score(estimator=knn, cv = kf, X = wineClass, y = signs, scoring='accuracy')
# importances = clf.feature_importances_
# print(importances)
# print(kf)

# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

'''accuracy
countSex = data['Sex'].value_counts()

file_name = "Src 1"
intoFile(file_name, countSex.male, countSex.female)
printFile(file_name)

total = data['Survived'].count()
survived = data['Survived'].sum()
survivedAvg = survived / total * 100

file_name = "Src 2"
intoFile(file_name, survivedAvg)
printFile(file_name)

# firstClass1 = data['Pclass']==1 #.sum()/total*100).
firstClass = data[data.Pclass.eq(1)]
avgOfFirstClass = firstClass.shape[0] / total * 100

file_name = "Src 3"
intoFile(file_name, avgOfFirstClass)
printFile(file_name)

avgAge = data['Age'].mean(axis=0)
avgAge_bad = data['Age'].sum() / total  # bad because includes row where we dont know age
median = data['Age'].median(axis=0)

file_name = "Src 4"
intoFile(file_name, avgAge, median)
printFile(file_name)

df = pandas.DataFrame(data, columns=['SibSp', 'Parch'])
corelation = df.corr(method='pearson', min_periods=1)

file_name = "Src 5"
intoFile(file_name, corelation['SibSp']['Parch'])
printFile(file_name)

# data['Sex']='female'
# df_with_female = data[data['Sex'].eq('female')]
# sex is additional, can delete
dirty_first_name = data['Name'].str.extract(r'(Miss\.\s\w+)')
count_name = dirty_first_name.value_counts()
first_name = count_name.index[0][0][6:]
file_name = "Src 6"
intoFile(file_name, first_name)
printFile(file_name)
'''
