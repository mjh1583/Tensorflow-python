import tensorflow as tf
import pandas as pd

# with reshape() : reshape 사용
# 데이터 불러오기
(indep, dep), _ = tf.keras.datasets.mnist.load_data()
print(indep.shape, dep.shape)

indep = indep.reshape(60000, 784)
dep = pd.get_dummies(dep)
print(indep.shape, dep.shape)

# 모델 만들기
X = tf.keras.layers.Input(shape=[784])
H = tf.keras.layers.Dense(84, activation='swish')(X)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 모델 학습
model.fit(indep, dep, epochs=10)

# 모델 이용
pred = model.predict(indep[0:5])
print(pd.DataFrame(pred).round(2))  # 예측

print(dep[0:5])  # 답

# ======================================================
# with flatten() : flatten 사용
(indep, dep), _ = tf.keras.datasets.mnist.load_data()
print(indep.shape, dep.shape)

# indep = indep.reshape(60000, 784)
dep = pd.get_dummies(dep)
print(indep.shape, dep.shape)

# 모델 만들기
X = tf.keras.layers.Input(shape=[28, 28])
H = tf.keras.layers.Flatten()(X)  # Flatten 사용
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 모델 학습
model.fit(indep, dep, epochs=5)

# 모델 이용
pred = model.predict(indep[0:5])
print(pd.DataFrame(pred).round(2))  # 예측

print(dep[0:5])  # 답
