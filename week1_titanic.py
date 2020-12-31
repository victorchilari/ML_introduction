import pandas


def intoFile(name, content, sec_content=""):
    # round float
    content = content if not isinstance(content, float) else round(content, 2)
    sec_content = sec_content if not isinstance(sec_content, float) else round(sec_content, 2)

    f = open(name + ".txt", "w")
    if sec_content == "":
        text = content
    else:
        text = (content, sec_content)
    f.write(str(text))
    f.close()


def printFile(name):
    f = open(name + ".txt", "r")
    print(name + "\n", f.read())


data = pandas.read_csv('titanic.csv', index_col='PassengerId')

countSex = data['Sex'].value_counts()
file_name = "Src 1"
intoFile(file_name, countSex)
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
corelation = df.corr(method='pearson', min_periods=1).to_string()

file_name = "Src 5"
intoFile(file_name, corelation)
printFile(file_name)

# data['Sex']='female'
df_with_female = data[data['Sex'].eq('female')]
# sex is additional, can delete
first_name = df_with_female['Name'].str.extract(r'(Miss\.\s\w+)')
count_name = first_name.value_counts()

file_name = "Src 6"
intoFile(file_name, count_name)
printFile(file_name)
