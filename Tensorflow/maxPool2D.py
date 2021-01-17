import tensorflow as tf
import pandas as pd

# 데이터 준비하기
(indep, dep), _ = tf.keras.datasets.mnist.load_data()
print(indep.shape, dep.shape)

indep = indep.reshape(60000, 28, 28, 1)
dep = pd.get_dummies(dep)  # 원핫인코딩
print(indep.shape, dep.shape)

# 모델의 구조 만들기
X = tf.keras.layers.Input(shape=[28, 28, 1])

H = tf.keras.layers.Conv2D(3, kernel_size=5, activation='swish')(X)  # 3개의 특징맵 =  3채널의 특징맵
H = tf.keras.layers.MaxPool2D()(H)
H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(H)  # 6개의 특징맵 =  6채널의 특징맵
H = tf.keras.layers.MaxPool2D()(H)

H = tf.keras.layers.Flatten()(H)  # 표로 만듦
H = tf.keras.layers.Dense(84, activation='swish')(H)
Y = tf.keras.layers.Dense(10, activation='softmax')(H)
model = tf.keras.models.Model(X, Y)
model.compile(loss='categorical_crossentropy', metrics='accuracy')

# 모델 학습
model.fit(indep, dep, epochs=10)

# 모델 이용
pred = model.predict(indep[0:5])
print(pd.DataFrame(pred).round(2))  # 예측

print(dep[0:5])  # 답

# 모델 알아보기
print(model.summary())
