from matplotlib.pyplot import figure, plot, title, legend, show
from numpy import arange, vstack, zeros, array
from numpy import random
from pandas import read_csv
from sklearn.metrics import accuracy_score
from tensorflow.python.keras import *
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.utils import np_utils


def data(data, selection):
    info = read_csv(f'C:/Users/gamza/PycharmProjects/machine_learning/CP/support/{selection}/{data}.csv').drop(['id', 'time'], axis=1)
    return info

def generation(data, lookback, delay, min_index, kz, max_index=None, shuffle=False, batch_size=128, step=1):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while i + batch_size <= max_index:
        if shuffle:
            rows = random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = arange(i, min(i + batch_size, max_index))
        i += len(rows)

        print(f"Number of matrices：{rows}")
        print(f"Number of rows：{lookback // step}")
        print(f"Number of columns：{data.shape[-1]}")

        samples = zeros((len(rows), lookback // step, data.shape[-1]))
        targets = zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = list(range(rows[j] - lookback, rows[j], step))
            samples[j] = data[indices]
            targets[j] = kz  # [1]
        yield samples, targets


abc_train = data('abc', 'train')
ab_train = data('ab', 'train')
bc_gr_train = data('bc_gr', 'train')
a_train = data('a', 'train')
c_gr_train = data('c_gr', 'train')

abc_test = data('abc', 'test')
ab_test = data('ab', 'test')
bc_gr_test = data('bc_gr', 'test')
a_test = data('a', 'test')
c_gr_test = data('c_gr', 'test')


# 43800
def program(kz_sample, kz):
    x_kz_sample = [i[0] for i in
                   generation(kz_sample.values, lookback=30, delay=0, min_index=9000, max_index=21000, batch_size=128, step=1, kz=kz)]
    y_kz_sample = [i[1] for i in
                   generation(kz_sample.values, lookback=30, delay=0, min_index=9000, max_index=21000, batch_size=128, step=1, kz=kz)]
    return array(x_kz_sample).reshape(-1, array(x_kz_sample).shape[2], array(x_kz_sample).shape[3]), array(y_kz_sample).reshape(-1, 1)


Xabc_train, Yabc_train = program(abc_train, 0)
Xab_train, Yab_train = program(ab_train, 1)
Xbc_gr_train, Ybc_gr_train = program(bc_gr_train, 2)
Xa_train, Ya_train = program(a_train, 3)
Xc_gr_train, Yc_gr_train = program(c_gr_train, 4)

Xabc_test, Yabc_test = program(abc_test, 0)
Xab_test, Yab_test = program(ab_test, 1)
Xbc_gr_test, Ybc_gr_test = program(bc_gr_test, 2)
Xa_test, Ya_test = program(a_test, 3)
Xc_gr_test, Yc_gr_test = program(c_gr_test, 4)


def change(x_kz0, x_kz1, x_kz2, x_kz3, x_kz4, y_kz0, y_kz1, y_kz2, y_kz3, y_kz4):
    X_sample = vstack((x_kz0, x_kz1, x_kz2, x_kz3, x_kz4))
    Y_sample = vstack((y_kz0, y_kz1, y_kz2, y_kz3, y_kz4))
    Y_sample = np_utils.to_categorical(Y_sample, 5)
    print(X_sample.shape)
    print(Y_sample.shape)
    return X_sample, Y_sample


X_train, y_train = change(Xabc_train, Xab_train, Xbc_gr_train, Xa_train, Xc_gr_train,
                          Yabc_train, Yab_train, Ybc_gr_train, Ya_train, Yc_gr_train)

X_test = vstack((Xabc_test, Xab_test, Xbc_gr_test, Xa_test, Xc_gr_test))
y_test = vstack((Yabc_test, Yab_test, Ybc_gr_test, Ya_test, Yc_gr_test))
print(X_test.shape)
print(y_test.shape)
y_test = y_test.reshape(-1, )

NB_CLASSES = 5
model = Sequential()
model.add(GRU(32, return_sequences=True, input_shape=(None, 3)))
model.add(Dropout(0.1))
model.add(GRU(32, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
model.summary()

EPOCHS = 10
EPOCHS_list = arange(EPOCHS)
history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_train, y_train))
accuracy_list = history.history['accuracy']
min_index = accuracy_list.index(min(accuracy_list))
print(f'Optimal number of training epochs: {EPOCHS_list[min_index]}')
print("accuracy_min: ", accuracy_list[min_index])

epochs = range(len(accuracy_list))

figure()
plot(epochs, 'bo', label='Training loss')
title('Training and validation loss')
legend()
show()

def print_accuracy(X, y_true, y):
    Y_pred = model.predict_classes(X)
    y_pred = []
    for i in range(len(Y_pred)):
        if Y_pred[i] == y:
            y_pred.append(y)
        else:
            y_pred.append(y + 1)
    print(f"Short circuit prediction accuracy -> {y}:", accuracy_score(y_true, array(y_pred)))

print_accuracy(Xabc_test, Yabc_test, 0)
print_accuracy(Xab_test, Yab_test, 1)
print_accuracy(Xbc_gr_test, Ybc_gr_test, 2)
print_accuracy(Xa_test, Ya_test, 3)
print_accuracy(Xc_gr_test, Yc_gr_test, 4)

# # Save the entire model to an HDF5 file
# model.save('my_model.h5')
# # Let's recreate exactly this model including weights and the optimizer.
# model = models.load_model('my_model.h5')
