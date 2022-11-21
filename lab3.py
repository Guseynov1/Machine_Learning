from keras.datasets import mnist
from pandas import *
from matplotlib.pyplot import *
from seaborn import *
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical
import warnings

from additionally.supportive import draw, neuronet, combined_histogram, data, readCSV, boxplots
import tensorflow as tf

warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
range_ = arange(1, 20)
range__ = arange(1, 10)

# Build a histogram of objects based on "workclass"
data['workclass'].hist(figsize=(15, 10))
# Visualize various combined histograms of objects by various attributes
combined_histogram("income", data, "sex", "Set1", "Frequency distribution of income variable wrt sex")
combined_histogram("income", data, "race", "Set1", "Distribution of income variable thoughtout race")
combined_histogram("workclass", data, "income", "Set1", "Distribution of income variable thoughtout workclass")
combined_histogram("workclass", data, "sex", "Set1", "Distribution of sex variable thoughtout workclass")
# Build a histogram of objects based on "age"
subplots(figsize=(10, 8))
histplot(data['age'], bins=10, color='purple').set_title("Distribution of age variable")
show()
# Visualize boxes with moustaches by various signs
boxplots(x=data['age'], y=None, hue=None, data=data, name="Visualize outliers in age variable")
boxplots(x="income", y="age", hue=None, data=data, name="Visualize income wrt age variable")
boxplots(x="income", y="age", hue="sex", data=data, name="Visualize income wrt age and sex variable")
boxplots(x="race", y="age", hue=None, data=data, name="Visualize race wrt age variable")
# Heat map of feature correlation
data.corr(numeric_only=True).style.format("{:.4}").background_gradient(cmap=get_cmap('coolwarm'), axis=1)

print('\nCategorical features in the set')
categorical = [var for var in data.columns if data[var].dtype == 'O']
print(data[categorical].head())

print('\nNumerical features in the set')
numerical = [var for var in data.columns if data[var].dtype == 'int64']
print(data[numerical].head())

X, y = readCSV()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
# To train a model of the decision tree for the classification problem,
# to plot the dependence of the F-measure on the training sample and on the test sample on the depth of the tree.
f1_train, f1_test = ([] for i in range(2))
for depth in range_:
    trees = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X_train, y_train)
    y_pred = trees.predict(X_test)
    y_pred_train = trees.predict(X_train)
    f1_test.append(f1_score(y_test, y_pred))
    f1_train.append(f1_score(y_train, y_pred_train))
draw(range_, f1_train, f1_test, "depth")
optim_depth = range_[f1_test.index(max(f1_test))]
print(f"{optim_depth} - optimal tree depth")
model = DecisionTreeClassifier(max_depth=optim_depth, random_state=0).fit(X_train, y_train)
matrix = confusion_matrix(y_test, model.predict(X_test))
matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix).plot()
title('Confusion matrix of the decision tree model')
show()

# П5. Train a random forest model for a classification task
f1_train, f1_test = ([] for i in range(2))
for trees in range_:
    forest = RandomForestClassifier(n_estimators=trees).fit(X_train, y_train.values.ravel())
    y_pred = forest.predict(X_test)
    y_pred_train = forest.predict(X_train)
    f1_test.append(f1_score(y_test, y_pred))
    f1_train.append(f1_score(y_train, y_pred_train))
draw(range_, f1_train, f1_test, "trees")
optim_num_of_trees = range_[f1_test.index(max(f1_test))]
print(f"{optim_num_of_trees} - optimal number of trees")
model = RandomForestClassifier(n_estimators=optim_num_of_trees).fit(X_train, y_train.values.ravel())
matrix = confusion_matrix(y_test, model.predict(X_test))
matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix).plot()
title('Confusion matrix of the random forest model')
show()

# П6. Train a gradient boosting model for a classification problem
f1_train, f1_test = ([] for i in range(2))
for cat in range_:
    boost = CatBoostClassifier(n_estimators=cat, verbose=False).fit(X_train, y_train)
    y_pred = boost.predict(X_test)
    y_pred_train = boost.predict(X_train)
    f1_test.append(f1_score(y_test, y_pred))
    f1_train.append(f1_score(y_train, y_pred_train))
draw(range_, f1_train, f1_test, "cat")
optim_num_of_cat = range_[f1_test.index(max(f1_test))]
print(f"{optim_num_of_cat} - optimal number of trees (cat)\n"
      f"Best number of trees is {optim_num_of_cat}")
model = CatBoostClassifier(n_estimators=optim_num_of_cat, verbose=False).fit(X_train, y_train)
matrix = confusion_matrix(y_test, model.predict(X_test))
matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix).plot()
title('Confusion matrix of the gradient boosting model')
show()

# П7.1. Training of the multilayer perceptron model, data preparation.
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
y_train = to_categorical(y_train, 2)[:, 0]
y_test = to_categorical(y_test, 2)[:, 0]
# Training of the multilayer perceptron model.
# To train a multilayer perceptron model for a classification problem with optimal parameters.
train, test = ([] for i in range(2))
for epochs in range_:
    y_train = to_categorical(y_train, 2)
    y_test = to_categorical(y_test, 2)
    NB_CLASSES = y_train.shape[1]
    INPUT_SHAPE = (X_train.shape[1],)
    model = Sequential()
    model.add(Dense(32, input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision', 'Recall'])
    history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(X_test, y_test))

    f1_score_list_train, f1_score_list_test = ([] for i in range(2))
    for i in range(epochs):
        f1_score_list_train.append(2 * history.history['precision'][i] * history.history['recall'][i] /
                                   (history.history['precision'][i] + history.history['recall'][i]))
        f1_score_list_test.append(2 * history.history['val_precision'][i] * history.history['val_recall'][i] /
                                  (history.history['val_precision'][i] + history.history['val_recall'][i]))
    test.append(mean(f1_score_list_test))
    train.append(mean(f1_score_list_train))
    X, y = readCSV()
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

y_train = to_categorical(y_train, num_classes=0)
y_test = to_categorical(y_test, num_classes=0)
draw(range_, train, test, 'epochs')
optimal_epochs = range_[test.index(max(test))]
print(f"{optimal_epochs} - best number of epochs")
neuronet(X_train, y_train)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision', 'Recall'])
model.fit(X_train, y_train, batch_size=32, epochs=optimal_epochs, verbose=1, validation_data=(X_test, y_test))
matrix = confusion_matrix(argmax(y_test, axis=1), argmax(model.predict(X_test), axis=-1))
matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix).plot()
title('Confusion matrix of the multilayer perceptron model')
show()

# П8.1 Catboost for MNIST
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
image = X_train_mnist[1]
# plot the sample
fig = figure
imshow(image, cmap='gray')
show()
X_train_mnist = X_train_mnist.reshape(X_train_mnist.shape[0], 28 * 28)
X_test_mnist = X_test_mnist.reshape(X_test_mnist.shape[0], 28 * 28)
f1_train, f1_test = ([] for i in range(2))
for trees_mnist in range_:
    boost = CatBoostClassifier(n_estimators=trees_mnist, verbose=False)
    boost.fit(X_train_mnist, y_train_mnist)
    y_pred = boost.predict(X_test_mnist)
    y_pred_train = boost.predict(X_train_mnist)
    f1_test.append(f1_score(y_test_mnist, boost.predict(X_test_mnist), average='micro'))
    f1_train.append(f1_score(y_train_mnist, boost.predict(X_train_mnist), average='micro'))
draw(range_, f1_train, f1_test, "trees_mnist")
best_num_of_trees = range_[f1_test.index(max(f1_test))]
print(f"{best_num_of_trees} - best number of trees (cat)")
model = CatBoostClassifier(n_estimators=best_num_of_trees)
model.fit(X_train_mnist, y_train_mnist)

DataFrame(data=confusion_matrix(y_test_mnist, model.predict(X_test_mnist)),
          columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
          index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# П8.2 Neural network for MNIST
(X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = mnist.load_data()
X_train_mnist = X_train_mnist.reshape(X_train_mnist.shape[0], 28 * 28)
X_test_mnist = X_test_mnist.reshape(X_test_mnist.shape[0], 28 * 28)
y_train_mnist = to_categorical(y_train_mnist, 10)
y_test_mnist = to_categorical(y_test_mnist, 10)
train, test = ([] for i in range(2))
for epochs_mnist in range__:
    NB_CLASSES = y_train_mnist.shape[1]
    INPUT_SHAPE = (X_train_mnist.shape[1],)
    model = Sequential()
    model.add(Dense(32, input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy', 'Precision', 'Recall'])
    history = model.fit(X_train_mnist, y_train_mnist, batch_size=32, epochs=epochs_mnist, verbose=1,
                        validation_data=(X_test_mnist, y_test_mnist))
    f1_score_list_train, f1_score_list_test = ([] for i in range(2))
    for i in range(epochs_mnist):
        f1_score_list_train.append(2 * history.history['precision'][i] * history.history['recall'][i] /
                                   (history.history['precision'][i] + history.history['recall'][i]))
        f1_score_list_test.append(2 * history.history['val_precision'][i] * history.history['val_recall'][i] /
                                  (history.history['val_precision'][i] + history.history['val_recall'][i]))
    test.append(mean(f1_score_list_test))
    train.append(mean(f1_score_list_train))

draw(range__, train, test, 'epochs')
optimal_epochs = range__[test.index(max(test))]
print(f"{optimal_epochs} - best number of epochs")
neuronet(X_train_mnist, y_train_mnist)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy', 'Precision', 'Recall'])
model.fit(X_train_mnist, y_train_mnist, batch_size=32, epochs=optimal_epochs,
          verbose=1, validation_data=(X_test_mnist, y_test_mnist))

y_pred = model.predict_classes(X_test_mnist)
matrix = confusion_matrix(argmax(y_test_mnist, axis=-1), y_pred)
cm = DataFrame(data=matrix,
               columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
               index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
heatmap(cm, annot=True, fmt="d")
matrix_display = ConfusionMatrixDisplay(confusion_matrix=matrix)
matrix_display.plot()
title('Confusion matrix of the multilayer perceptron model mnist')
show()
