from keras import models
from keras import layers


# 建立神经网络模型

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(76,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',  # 还可以通过optimizer = optimizers.RMSprop(lr=0.001)来为优化器指定参数
                  loss='binary_crossentropy',  # 等价于loss = losses.binary_crossentropy
                  metrics=['accuracy'])  # 等价于metrics = [metircs.binary_accuracy]
    return model
