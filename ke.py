import os
import pandas as pd


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from load import load_data
import numpy as np
from build import build_model


train_data, test_data, train_labels, test_labels = load_data('d:/source/cicids2018/fri.csv', 500000)

x_train = train_data
x_test = test_data
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = build_model()
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,  # 在全数据集上迭代20次
                    batch_size=512,  # 每个batch的大小为512
                    validation_data=(x_val, y_val))

history_dict = history.history
history_dict.keys()

print(model.summary())
with pd.ExcelWriter('test.xlsx') as writer:
    data = pd.DataFrame(model.get_weights())
    data.to_excel(writer, sheet_name='page1', float_format='%.6f')

print(model.get_weights())
model.fit(x_train,
          y_train,
          epochs=4,  # 由loss图发现在epochs=4的位置上validation loss最低
          batch_size=512)

results = model.evaluate(x_test, y_test)
print(results)
