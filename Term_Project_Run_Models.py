# Benjamin Pittman
# University of Washington-Bothell
# CS 581 Autumn 2017
# 
# Code to pull in the pre-precessed data set and perform machine learning 
# modeling


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from pandas_ml import ConfusionMatrix
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer
import keras.optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Method to plot a classifiaction report
def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()
    
    
# Creates a dataframe from the final csv and trims it into the attributes and
# labels that will feed our models
# Read in data from CSV, preprocess and normalize for use in model
mainDF = pd.read_csv('final_data_1205.csv')
mask = mainDF['medTimepoint'] == 'Just after Parkinson medication (at your best)'
reducedDF = mainDF[~mask]
reducedDF.drop(['Unnamed: 0', 'recordId', 'healthCode', 'appVersion', 'phoneInfo', 'medTimepoint'], axis=1, inplace=True)
data = reducedDF.ix[:, 1:110]
labels = reducedDF.ix[:,0]
labels = labels.to_frame()
labels.reset_index(inplace=True)
labels.drop(['index'], axis=1, inplace=True)
labels = labels.values
labels = labels.flatten()
data = data.values
encoder = LabelEncoder()
encoder.fit(labels)
encoded_labels = encoder.transform(labels)

# Split the data into train and test sets and standardize
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.2, random_state=0)
X_train_norm = (X_train - X_train.mean()) / X_train.std()
X_test_norm = (X_test - X_test.mean()) / X_test.std()

# PRINCIPAL COMPONENT ANALYSIS - removed, not relevant

# LOGISTIC REGRESSION
print("# Tuning hyper-parameters")
print()
fold = KFold(len(y_train), n_folds=10, shuffle=True, random_state=0)
grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'tol': [1e-4, 1e-3, 1e-2], 'solver': ['newton-cg', 'sag', 'lbfgs']}
gs = GridSearchCV(LogisticRegression(penalty='l2', max_iter=1000), param_grid=grid, cv=fold)
gs.fit(X_train_norm, y_train)

print("Best parameters set found on development set:")
print()
print(gs.best_params_)
print()
print("Grid scores on development set:")
print()
means = gs.cv_results_['mean_test_score']
stds = gs.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, gs.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, gs.predict(X_test_norm)
class_report = classification_report(y_true, y_pred)
cm = ConfusionMatrix(y_test, y_pred)
cm.plot()
plt.show()
print(class_report)
print()
plot_classification_report(class_report)

# DECISION TREE CLASSIFIER
tuned_parameters = [{"criterion": ["gini", "entropy"], "min_samples_split": [10, 20, 40], "max_depth": [10, 14, 18], "min_samples_leaf": [10, 20, 40]}]
fold = KFold(len(y_train), n_folds=10, shuffle=True, random_state=0)
print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=fold)
clf.fit(X_train_norm, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test_norm)
class_report = classification_report(y_true, y_pred)
cm = ConfusionMatrix(y_test, y_pred)
cm.plot()
plt.show()
print(class_report)
print()
plot_classification_report(class_report)

# K NEAREST NEIGHBORS
metrics = ['minkowski','euclidean','manhattan'] 
weights = ['uniform','distance'] 
numNeighbors = [5, 7, 10, 15, 20]
param_grid = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)
fold = KFold(len(y_train), n_folds=10, shuffle=True, random_state=0)
print("# Tuning hyper-parameters")
print()
clf = GridSearchCV(neighbors.KNeighborsClassifier(),param_grid=param_grid,cv=fold)
clf.fit(X_train_norm, y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test_norm)
class_report = classification_report(y_true, y_pred)
cm = ConfusionMatrix(y_test, y_pred)
cm.plot()
plt.show()
print(class_report)
print()
plot_classification_report(class_report)

# SUPPORT VECTOR CLASSIFICATION
fold = KFold(len(y_train), n_folds=10, shuffle=True, random_state=0)
# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

print("# Tuning hyper-parameters")
print()

clf = GridSearchCV(SVC(), param_grid=tuned_parameters, cv=fold)
clf.fit(X_train_norm, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test_norm)
class_report = classification_report(y_true, y_pred)
cm = ConfusionMatrix(y_test, y_pred)
cm.plot()
plt.show()
print(class_report)
print()
plot_classification_report(class_report)

# ARTIFICIAL NEURAL NETWORK
def create_model():
    model = Sequential()
    model.add(Dense(40, input_dim=108, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(21, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(9, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
results = cross_val_score(pipeline, X_train_norm, y_train, cv=kfold)
print("Mean: %.2f%% Std: (%.2f%%)" % (results.mean()*100, results.std()*100))

y_pred = model.predict_classes(X_test_norm, batch_size=10, verbose=0)
y_pred = y_pred.flatten()
cm = ConfusionMatrix(y_test.astype(bool), y_pred.astype(bool))
print(cm)
cm.plot()
plt.show()
class_report = classification_report(y_test.astype(bool), y_pred.astype(bool))
print(class_report)
print()
plot_classification_report(class_report)

# ALGORITHM COMPARISON PLOT
results = []
names = []

names.append('LR')
results.append([0.639, 0.695])
names.append('DTr')
results.append([0.685, 0.697])
names.append('KNN')
results.append([0.666, 0.721])
names.append('SVC')
results.append([0.597, 0.718])
names.append('ANN')
results.append([0.7635, 0.786])

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()