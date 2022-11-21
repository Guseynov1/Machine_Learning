from pandas import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import *
from sklearn.metrics import *
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from additionally import supportive as processing


# 1 Processing of analyzed data
def readCSV():
    raw_data = read_csv("breast_cancer.csv")
    sns.set({'figure.figsize': (30, 30)})
    sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=raw_data.sample(569))
    x = get_dummies(raw_data)
    y = x.get("diagnosis_B", "diagnosis_M")
    x.drop(["Unnamed: 32", "diagnosis_B", "diagnosis_M", "id"], axis=1, inplace=True)
    plt.show()
    return x, y

X, y = readCSV()
# 2 Train the model of the nearest neighbors on the training sample and check the quality of the model on
# the training and test samples with a change in the number of neighbors
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=11)

results, neighbours, error_rates = ([] for i in range(3))
neighbours_range = range(1,50)
for k in neighbours_range:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("report")
    print(classification_report(y_test, prediction))
    print("matrix")
    print(confusion_matrix(y_test, prediction))
    error_rates.append(np.mean(prediction != y_test))
    results.append(accuracy_score(prediction, y_test))
    neighbours.append(k)

plt.plot(error_rates)
plt.show()
plt.plot(neighbours, results)
plt.show()
print("Наилучший результат при количестве соседей =", neighbours[results.index(max(results))])


# 3 Partitioning generator for cross-validation in five blocks
kf = KFold(n_splits=5, shuffle=True)

# 4 Finding the optimal value of k for the nearest neighbor method
neighbours, results = ([] for i in range(2))
for v in neighbours_range:
    model = KNeighborsClassifier(n_neighbors=k)
    array = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
    neighbours.append(v)
    results.append(max(array))

print("Наилучший результат при количестве соседей =", neighbours[results.index(max(results))])
processing.graph1(neighbours, results, 'neighbours_range')

# 5 Finding the optimal value of C for the logistic regression method
regression, results = ([] for i in range(2))
range_reg = np.arange(0.01, 1, 0.01) # диапазон (вектор) вещественных чисел
for c in range_reg:
    model = LogisticRegression(C=c, random_state=5, max_iter=1000, solver="liblinear").fit(X_train, y_train)
    score = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
    regression.append(c)
    results.append(max(score))

print(np.min(regression))
processing.graph1(range_reg, results, "regression")

# 6 Scaling of numerical features and repeat items 4 and 5
scaled = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(scaled, y, test_size=0.3, train_size=0.7, random_state=11)

# 4
cross_scaled=[]
for k in neighbours_range:
    model_scaled=KNeighborsClassifier(n_neighbors=k)
    score=(cross_val_score(model_scaled, scaled, y, cv=kf, scoring='accuracy'))
    cross_scaled.append(score.mean())
processing.graph1(neighbours_range, cross_scaled, "neighbors")

# 5
regression_scaled=[]
c_range=np.arange(0.1, 1, 0.1)
for c in c_range:
    model_reg_scaled=LogisticRegression(C=c, random_state=1, max_iter=1000, solver="liblinear")
    model_reg_scaled.fit(X_train, y_train)
    score=(cross_val_score(model_reg_scaled, scaled, y, cv=kf, scoring='accuracy'))
    regression_scaled.append(score.mean())
processing.graph1(c_range, regression_scaled, "Inverse of Regularization")


