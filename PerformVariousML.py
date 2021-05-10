# This script performs various ML algorithms: Logistic Classification, Decision Trees, Random Forest, Naive-Baysed, KNN, Kernel SVM
# Data must be pre-processed before entered here

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def PerformAnalysis(Features,Labels,Split_Size,Standardize):

    accuracy = []

    x_train, x_test, y_train, y_test = train_test_split(Features,Labels,test_size = Split_Size,shuffle=False) #do not randomly split it, leave the last 20% for training/prediction

    if Standardize: 
        sc = StandardScaler()
        Features = sc.fit_transform(Features) 

    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc_score = accuracy_score(y_test,y_pred)
    #print('Random Forest Model Accuracy:',acc_score)
    accuracy.append(acc_score)

    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc_score = accuracy_score(y_test,y_pred)
    #print('Naive Bayes Model Accuracy:',acc_score)
    accuracy.append(acc_score)

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    acc_score = accuracy_score(y_test,y_pred)
    #print('KNN Model Accuracy:',acc_score)
    accuracy.append(acc_score)

    classifier = SVC(kernel = 'rbf')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc_score = accuracy_score(y_test,y_pred)
    #print('Kernel SVM Model Accuracy:',acc_score)
    accuracy.append(acc_score)

    classifier = DecisionTreeClassifier(criterion = 'entropy')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    acc_score = accuracy_score(y_test,y_pred)
    #print('Decision Tree Model Accuracy:',acc_score)
    accuracy.append(acc_score)

    classifier = LogisticRegression()
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    acc_score = accuracy_score(y_test,y_pred)
    #print('Logistic Regression Model Accuracy:',acc_score)
    #print('')
    accuracy.append(acc_score)

    models = ['Random Forest','Naive-Bayes','KNN','Kernel SVM','Decision Tree','Logistic Classification']
    max_index = np.argmax(accuracy)
    model_name = models[max_index]

    return np.max(accuracy),model_name