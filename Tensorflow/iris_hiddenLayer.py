# 라이브러리 사용
import tensorflow as tf
import pandas as pd

# 1. 과거의 데이터를 준비합니다.
path = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(path)
print(iris.head())

# 원핫인코딩 : 범주형 변수를 0,1 의 데이터로 바꿔줌
encoding = pd.get_dummies(iris)
print(encoding.head())
print(encoding.columns)

indep = encoding[['꽃잎길이', '꽃잎폭', '꽃받침길이', '꽃받침폭']]
dep = encoding[['품종_setosa', '품종_versicolor', '품종_virginica']]
print(indep.shape, dep.shape)

# 2. 모델의 구조를 만듦니다.
# 비율을 예측하는데 Softmax 사용
X = tf.keras.layers.Input(shape=[4])
H = tf.keras.layers.Dense(8, activation='swish')(X)
H = tf.keras.layers.Dense(8, activation='swish')(H)
H = tf.keras.layers.Dense(8, activation='swish')(H)
Y = tf.keras.layers.Dense(3, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')  # 분류에 사용하는 loss : categorical_crossentropy

# 3. 데이터로 모델을 학습(Fit)합니다.
model.fit(indep, dep, epochs=1000, verbose=0)
model.fit(indep, dep, epochs=10)

# 4. 모델을 이용합니다.
model.predict(indep[0:5])  # 마지막 5개의 데이터만 예측
print(model.predict(indep[0:5]))
print(dep[0:5])

# 5. 학습한 가중치
model.get_weights()
print(model.get_weights())
