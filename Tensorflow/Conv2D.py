# 컨볼루션 이해하기
# 컨볼루션(Convolution) : 특정한 패턴의 특징이 어디서 나타나는지를 확인하는 도구
# 필터를 거쳐 특징맵(feature map)이 만들어짐, 필터 하나가 하나의 특징맵을 만듦

# 필터의 이해
# 1. 필터셋은 3차원 형태로 된 가중치의 모음
# 2. 필터셋 하나는 앞선 레이어의 결과인 "특징맵" 전체를 본다.
# 3. 필터셋 갯수 만큼 특징맵을 만든다.

# GPU 이용하지 않으면 연산이 많이 느리므로 Gpu 환경 구축하는것이 좋음

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
H = tf.keras.layers.Conv2D(6, kernel_size=5, activation='swish')(H)  # 6개의 특징맵 =  6채널의 특징맵
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
