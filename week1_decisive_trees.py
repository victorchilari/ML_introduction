import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def mostImportantClf(arrOfI, arrOfD, quantity):
    rez_arr = []
    
    #
    sortI = arrOfI.copy()
    sortI.sort()
    sortI[:] = sortI[::-1]
    
    z=0
    while z < quantity:
        imp_weight = sortI[z]
        #imp_val = arrOfI.index(imp_weight)
        imp_pos = np.where(arrOfI == imp_weight)
        arrOfD = np.array(arrOfD)
        field = arrOfD[imp_pos]
        
        rez_arr.extend(field)
        z += 1
    
    return rez_arr

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

WORK_COLUMNS = ['Pclass', 'Fare', 'Age', 'Sex']

dirty_df = pandas.DataFrame(data, columns=[*WORK_COLUMNS, 'Survived'])
df = dirty_df.replace("female", 0).replace("male", 1)
df = df.dropna()

X = pandas.DataFrame(df, columns=WORK_COLUMNS)
X = np.array(X)

y = df['Survived']

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
importances = clf.feature_importances_
print(importances)

arrMIC = mostImportantClf(importances, WORK_COLUMNS, 2)
print(*arrMIC)

f = open("Src 7.txt", "w")
f.write(arrMIC[0] + " " + arrMIC[1])
f.close()