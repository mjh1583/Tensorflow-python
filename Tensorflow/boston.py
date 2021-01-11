# 라이브러리 사용
import tensorflow as tf
import pandas as pd

# 1. 데이터 불러오기
path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(path)
print(boston.columns)
print(boston.head())

# 2. 독립/종속
indep = boston[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']]
dep = boston[['medv']]
print(indep.shape, dep.shape)

# 3. 모델의 구조
X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 4. 데이터로 모델을 학습합니다.
model.fit(indep, dep, epochs=10000, verbose=0)
model.fit(indep, dep, epochs=10)

# 5. 모델을 이용합니다.
print(model.predict(indep[5:10]))

# 종속변수 확인
print(dep[5:10])

# 6. 모델의 수식 확인
print(model.get_weights())
