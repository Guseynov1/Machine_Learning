from pandas import *
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn import feature_extraction
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import *
from additionally import graph as draw

# 1. Processing of analyzed data
data = read_csv('spam.csv', encoding='ISO-8859-1')\
    .drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1)\
    .rename(columns={'v1': 'type', 'v2': 'text'})

# 2. Pie chart of target variable values
target = value_counts(data['type']).plot(kind='pie', autopct='%1.1f%%')
plt.title('Диаграмма целевой переменной')
plt.show()

# 3. Build a bar chart for the twenty most common words in both classes
# Let's calculate the most common words in both classes
ham = Counter(" ".join(data[data['type'] == 'ham']['text']).split()).most_common(20)
df_ham = DataFrame.from_dict(ham).rename(columns={0: 'non-spam words', 1 : 'count'})
spam = Counter(" ".join(data[data['type'] == 'spam']['text']).split()).most_common(20)
df_spam = DataFrame.from_dict(spam).rename(columns={0: 'spam words', 1 : 'count'})
# Building a bar chart ham и spam
df_ham.plot.bar(x='non-spam words', legend=False, color='purple')
y_pos_ham = np.arange(len(df_ham['non-spam words']))
plt.xticks(y_pos_ham, df_ham['non-spam words'])
plt.title('More frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

df_spam.plot.bar(x='spam words', legend = False, color ='red')
y_pos_spam = np.arange(len(df_spam['spam words']))
plt.xticks(y_pos_spam, df_spam['spam words'])
plt.title('More frequent words in spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()

# 4. Perform tokenization of a text attribute by excluding uninformative frequently occurring words
tokenizer = feature_extraction.text.CountVectorizer(stop_words = 'english')
X = tokenizer.fit_transform(data['text'])
y = data['type'].map({'spam':1, 'ham':0})

# 5. Find the optimal smoothing parameter alpha for a naive Bayesian classifier using precision and accuracy metrics

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.33, shuffle=True)

accuracy_test, accuracy_train, recall, precision = ([] for i in range(4))
alpha_range = np.arange(0.1, 20, 0.1)
for a in alpha_range:
    # The Naive Bayesian model is initialized with smoothing, we use the training data to estimate the parameters of the model
    model_a = MultinomialNB(alpha = a).fit(X_train, y_train)
    # Let's perform category prediction
    accuracy_test.append(accuracy_score(y_test, model_a.predict(X_test)))
    accuracy_train.append(accuracy_score(y_train, model_a.predict(X_train)))
    recall.append(recall_score(y_test, model_a.predict(X_test)))
    precision.append(precision_score(y_test, model_a.predict(X_test)))

alpha = {'alpha': np.arange(0.1, 20, 0.1)}
kf = KFold(n_splits=5,shuffle=True)
model_a = MultinomialNB()
gs = GridSearchCV(model_a, alpha, scoring='accuracy', cv=kf).fit(X, y)
print(gs.best_params_)
print(gs.best_score_)

# Search for the optimal parameter based on the received data - for convenience, we will reduce the values of metrics into one table
models_a = DataFrame(data = np.matrix(np.c_[alpha_range, accuracy_train, accuracy_test, recall, precision]),
                      columns = ['alpha', 'train accuracy', 'test accuracy', 'test recall', 'test precision'])
best_value = models_a['test precision'].max()
best_index = models_a['test accuracy'].idxmax()
best_index = models_a[models_a['test precision'] == best_value]['test accuracy'].idxmax()
# According to the found optimal parameter, we will train a new classifier - to use it in the future
a_model = MultinomialNB(alpha = alpha_range[best_index]).fit(X_train, y_train)

# 6. To construct the dependence of the accuracy metric on training and test data on the variable parameter
draw.graph2(alpha_range, accuracy_test, accuracy_train, 'alpha', 'accuracy')
# Construct confusion matrix for the model with the optimal selected parameter.
confusion_a = DataFrame(data = confusion_matrix(y_test, a_model.predict(X_test)),
          columns = ['predicted  ham', 'predicted spam'], index = ['actual ham', 'actual spam'])
print(confusion_a)

model_a.fit(X_train, y_train)
# another matrix confusion
plot_confusion_matrix(model_a, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show()

# 7. Build a ROC curve and calculate the AUC-ROC metric
# Area under the curve
y_pred_pr = a_model.predict_proba(X_test)[:,1]
fpr, tpr, threshold = roc_curve (y_test, y_pred_pr)
roc_auc = auc(fpr, tpr)
print(roc_auc)
# Building a curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid('on')
plt.show()

# 8. Find the optimal parameter of the regularizer C for the support vector model using precision and accuracy metrics
C= {'C': np.arange(0.01, 3, 0.1)}
kf = KFold(n_splits=5,shuffle=True)
model_svc = SVC()
gs = GridSearchCV(model_svc, C, scoring='accuracy', cv=kf).fit(X, y)
print(gs.best_params_)
print(gs.best_score_)

test_accuracy_svc, train_accuracy_svc, precision_svc, recall_svc = ([] for i in range(4))
c_range= np.arange(0.1, 3, 0.1)
for c in c_range:
    model_svc = SVC(C=c).fit(X_train, y_train)
    test_accuracy_svc.append(accuracy_score(y_test, model_svc.predict(X_test)))
    train_accuracy_svc.append(accuracy_score(y_train, model_svc.predict(X_train)))
    precision_svc.append(precision_score(y_test, model_svc.predict(X_test)))
    recall_svc.append(recall_score(y_test, model_svc.predict(X_test)))

models_svc = DataFrame(data = np.matrix(np.c_[c_range, train_accuracy_svc, test_accuracy_svc, recall_svc, precision_svc]),
                      columns = ['C', 'train accuracy', 'test accuracy', 'test recall', 'test precision'])
best_value_svc = models_svc['test precision'].max()
best_index_svc = models_svc['test accuracy'].idxmax()
best_index_svc = models_svc[models_svc['test precision'] == best_value_svc]['test accuracy'].idxmax()
svc_model = SVC(C = c_range[best_index_svc], probability = True).fit(X_train, y_train)
7
# 6
draw.graph2(c_range, test_accuracy_svc, train_accuracy_svc, 'С', 'accuracy')
# Confusion matrixSVC
confusion_svc = DataFrame(data = confusion_matrix(y_test, svc_model.predict(X_test)),
          columns = ['predicted  ham', 'predicted spam'], index = ['actual ham', 'actual spam'])
print(confusion_svc)

model_svc.fit(X_train, y_train)
# another matrix confusion
plot_confusion_matrix(model_svc, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
plt.title('Confusion matrix for our classifier')
plt.show()

# 7
# Area under the curve
y_pred_pr = svc_model.predict_proba(X_test)[:,1]
fpr, tpr, threshold = roc_curve(y_test, y_pred_pr)
roc_auc = auc(fpr, tpr)
print(roc_auc)
# Building a curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1.1])
plt.ylim([0, 1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid('on')
plt.show()