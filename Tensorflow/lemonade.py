# 라이브러리 사용
import tensorflow as tf
import pandas as pd

# 1. 데이터 준비
path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(path)
print(lemonade.head())

# 2. 종속변수, 독립변수 분리
indep = lemonade[['온도']]
dep = lemonade[['판매량']]
print(indep.shape, dep.shape)

# 3. 모델을 만듭니다.
X = tf.keras.layers.Input(shape=[1])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

# 4. 모델을 학습합니다.
model.fit(indep, dep, epochs=10000, verbose=0)
model.fit(indep, dep, epochs=10)

# 5. 모델을 이용합니다.
print(model.predict(indep))
print(model.predict([[15]]))
