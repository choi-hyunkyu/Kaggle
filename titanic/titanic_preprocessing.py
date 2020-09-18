import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set() # setting seaborn default for plots

# train, test 데이터 pands로 불러오기
train = pd.read_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\train.csv')
test = pd.read_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\test.csv')

# train 데이터 상위 5개 출력
train.head()

# train 데이터 feature 별 개수 확인
train.info()

# train 데이터 구성 확인
train.shape # 891명의 승객, 12개의 feature

# train 데이터 결측치 개수 확인
train.isnull().sum() # 각 Age 177개, Cabin 687개, Embarked 2개 씩 결측치 존재

# test 데이터 상위 5개 출력
test.head()

# test 데이터 feature 별 개수 확인
test.info()

# test 데이터 구성 확인
test.shape # 418명의 승객, 11개의 feature, test 데이터를 통해서 생존여부를 밝혀야 하므로 생존여부에 해당하는 1개의 feature가 없음

# test 데이터 결측치 확인
test.isnull().sum() # 각 Age 86개, Fare 1개, Cabin 327개 씩 결측치 존재




# 함수 정의
def bar_chart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
train.info()
train.isnull().sum()
bar_chart('Embarked')
#plt.show()

# Feature Engineering
# combining train and test dataset
train_test_data = [train, test]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

train['Title'].value_counts()
test['Title'].value_counts()

# First name 군집화

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, 
                 "Master": 3, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"Countess": 3,
                 "Ms": 3, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3 }
for dataset in train_test_data:
    dataset['Title'] = dataset['Title'].map(title_mapping)

train.head()
test.head()

bar_chart('Title')
#plt.show()

# 성별 군집화
sex_mapping = {"male" : 0, "female" : 1}
for dataset in train_test_data:
    dataset["Sex"] = dataset["Sex"].map(sex_mapping)

bar_chart('Sex')
#plt.show()

# Age 결측치 제거
# Age feature에서는 결측치가 존재하기 때문에 값을 대체해주어야 합니다.
train.head()
train.isnull().sum() # 177개의 결측치 존재

train["Age"].fillna(train.groupby("Title")["Age"].transform("median"), inplace = True)
test["Age"].fillna(test.groupby("Title")["Age"].transform("median"), inplace = True)

train.head(30)
train.isnull().sum() # 결측치 사라짐
train.groupby("Title")["Age"].transform("median")

# 연령별 생존-사망 그래프 출력
facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()

#plt.show() # 10대 후반 ~ 30대 중반에서 사망자가 가장 많았음

# 10대 후반 ~ 30대 중반 생존-사망 그래프 출력
facet = sns.FacetGrid(train, hue = "Survived", aspect = 4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim = (0, train['Age'].max()))
facet.add_legend()

plt.xlim(15, 38) # 구간별 출력
#plt.show()

# Age 군집화
for dataset in train_test_data:
    dataset.loc[ dataset['Age'] <= 20, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 30), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 50, 'Age'] = 4

train.head()

# Pclass 군집화 및 결측치 제거
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))
#plt.show()

for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

train.head()

embarked_mapping = {"S": 0, "C": 1, "Q": 2}
for dataset in train_test_data:
    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

train.head()

# Fare scaling 및 군집화
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform("median"), inplace = True)
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform("median"), inplace = True)
train.head(30)
test.head(30)

# Fare 분포에 따른 생존-사망 그래프 출력
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Fare',shade= True)
facet.set(xlim=(0, train['Fare'].max()))
facet.add_legend()
 
#plt.show()

for dataset in train_test_data:
    dataset.loc[ dataset['Fare'] <= 20, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 20) & (dataset['Fare'] <= 40), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 40) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 100, 'Fare'] = 3

train.head(30)

# Cabin 결측치 제거 및 군집화
train.Cabin.value_counts()
train.isnull().sum()
train.info()

for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([Pclass1, Pclass2, Pclass3])
df.index = ['1st class','2nd class', '3rd class']
df.plot(kind='bar',stacked=True, figsize=(10,5))

cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)

# FamilySize Feature combining 및 군집화
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade= True)
facet.set(xlim=(0, train['FamilySize'].max()))
facet.add_legend()
plt.xlim(0)
# plt.show()

family_mapping = {1: 0, 2: 0.4, 3: 0.8, 4: 1.2, 5: 1.6, 6: 2, 7: 2.4, 8: 2.8, 9: 3.2, 10: 3.6, 11: 4}
for dataset in train_test_data:
    dataset['FamilySize'] = dataset['FamilySize'].map(family_mapping)

train.head(30)

# 필요없는 데이터 삭제
train.drop('Name', axis = 1, inplace = True)
test.drop('Name', axis = 1, inplace = True)

feature_drop = ['Ticket', 'SibSp', 'Parch']
train = train.drop(feature_drop, axis = 1)
test = test.drop(feature_drop, axis = 1)

train = train.drop(['PassengerId'], axis = 1)
test = test.drop(['PassengerId'], axis = 1)

# 학습을 위한 데이터 분리
train_data = train.drop('Survived', axis = 1)
target = train['Survived']
test_data = test

train_data.shape, target.shape, test_data.shape

train_data.to_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\train_data.csv', index = False)
target.to_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\target.csv', index = False)
test_data.to_csv(r'C:\Users\user\Pytorch\AI\Kaggle\titanic\data\test_data.csv', index = False)