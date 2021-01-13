import pandas as pd

# 변수(칼럼) 타입 확인 : 데이터.dtypes
# 변수를 범주형으로 변경
#   데이터['칼럼명'].astype('category')
# 변수를 수치형으로 변경
#   데이터['칼럼명'].astype('int')
#   데이터['칼럼명'].astype('float')
# NA 값의 처리
#   NA 갯수 체크 : 데이터.isna().sum()
#   NA 값 채우기 : 데이터['칼럼명'].fillna(특정숫자)

# 파일 읽어 오기
path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris2.csv'
iris = pd.read_csv(path)
print(iris.head())

# 원핫인코딩
encoding = pd.get_dummies(iris)
print(encoding.head())

# 데이터 타입 확인
print(iris.dtypes)

# 품종 타입을 범주형으로 바꾸어 준다.
iris['품종'] = iris['품종'].astype('category')
print(iris.dtypes)

# 원핫인코딩
encoding = pd.get_dummies(iris)
print(encoding.head())

# NA값 체크
print(iris.isna().sum())
print(iris.tail())  # 꽃잎폭 마지막 값 NaN

# NA값에 꽃잎폭 평균값을 넣어주는 방법
mean = iris['꽃잎폭'].mean()
iris['꽃잎폭'] = iris['꽃잎폭'].fillna(mean)
print(iris.tail())
