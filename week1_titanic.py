import pandas
import wwt


data = pandas.read_csv('titanic.csv', index_col='PassengerId')

countSex = data['Sex'].value_counts()

file_name = "Src 1"
wwt.intoFile(file_name, countSex.male, countSex.female)
wwt.printFile(file_name)

total = data['Survived'].count()
survived = data['Survived'].sum()
survivedAvg = survived / total * 100

file_name = "Src 2"
wwt.intoFile(file_name, survivedAvg)
wwt.printFile(file_name)

# firstClass1 = data['Pclass']==1 #.sum()/total*100).
firstClass = data[data.Pclass.eq(1)]
avgOfFirstClass = firstClass.shape[0] / total * 100

file_name = "Src 3"
wwt.intoFile(file_name, avgOfFirstClass)
wwt.printFile(file_name)

avgAge = data['Age'].mean(axis=0)
avgAge_bad = data['Age'].sum() / total  # bad because includes row where we dont know age
median = data['Age'].median(axis=0)

file_name = "Src 4"
wwt.intoFile(file_name, avgAge, median)
wwt.printFile(file_name)

df = pandas.DataFrame(data, columns=['SibSp', 'Parch'])
corelation = df.corr(method='pearson', min_periods=1)

file_name = "Src 5"
wwt.intoFile(file_name, corelation['SibSp']['Parch'])
wwt.printFile(file_name)

# data['Sex']='female'
# df_with_female = data[data['Sex'].eq('female')]
# sex is additional, can delete
dirty_first_name = data['Name'].str.extract(r'(Miss\.\s\w+)')
count_name = dirty_first_name.value_counts()
first_name = count_name.index[0][0][6:]
file_name = "Src 6"
wwt.intoFile(file_name, first_name)
wwt.printFile(file_name)