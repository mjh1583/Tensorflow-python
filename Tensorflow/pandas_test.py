import pandas as pd

# 파일들로부터 데이터 읽어오기
path1 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(path1)

path2 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(path2)

path3 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(path3)

# 데이터 모양으로 확인하기
print(lemonade.shape)
print(boston.shape)
print(iris.shape)

# 칼럼 이름 출력
print(lemonade.columns)
print(boston.columns)
print(iris.columns)

indep1 = lemonade[['온도']]
dep1 = lemonade[['판매량']]
print(indep1.shape, dep1.shape)


indep2 = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
dep2 = boston[['medv']]
print(indep2.shape, dep2.shape)

indep3 = iris[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dep3 = iris[['품종']]
print(indep3.shape, dep3.shape)

print(lemonade.head())
print(boston.head())
print(iris.head())
