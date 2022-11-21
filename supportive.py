from matplotlib.pyplot import *
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from seaborn import countplot, boxplot
from pandas import *
from numpy import *

# lab 2
def graph1(param, accuracy, id):
    figure(figsize=[10, 10])
    plot(param, accuracy)
    xlabel(id)
    ylabel('accuracy')
    grid('on')
    show()

def graph2(param, accuracy_test, accuracy_train, id_x, id_y):
    figure(figsize=[10, 10])
    plot(param, accuracy_test)
    plot(param, accuracy_train)
    legend(["accuracy_train", "accuracy_test"])
    xlabel(id_x)
    ylabel(id_y)
    grid('on')
    show()

# lab 3

# Loading the analyzed csv data and counting the missing values.
data = read_csv('income.csv')
data.replace(' ?', NaN, inplace=True)
print("Пропущенные элементы")
print(data.isnull().sum())
data['workclass'].fillna(data['workclass'].mode()[0], inplace=True)
data['occupation'].fillna(data['occupation'].mode()[0], inplace=True)
data['native_country'].fillna(data['native_country'].mode()[0], inplace=True)

def readCSV():
    x = get_dummies(data)
    y = x.get(['income_ >50K'])
    x.drop(['income_ <=50K', 'income_ >50K'], axis=1, inplace=True)
    return x, y

def boxplots(x, y, hue, data, name):
    subplots(figsize=(10, 8))
    ax = boxplot(x=x, y=y, hue=hue, data=data)
    ax.set_title(name)
    show()

def draw(range_, f1_train, f1_test, name):
    figure(figsize=[10, 5])
    plot(range_, f1_train)
    plot(range_, f1_test)
    legend(['f1_train', 'f1_test'])
    xlabel(name)
    ylabel('metric')
    grid('on')
    show()

def combined_histogram(x, data, hue, palette, name):
    subplots(figsize=(10, 8))
    ax = countplot(x=x, data=data, hue=hue, palette=palette)
    ax.set_title(name)
    show()

def neuronet(X_train, y_train):
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








