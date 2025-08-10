import os

from scipy.io import arff
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC


##DecisionTreeClassifier

def DTC(X_train,y_train,X_test,y_test):
    tree_clf = DecisionTreeClassifier(min_samples_leaf=2)
    tree_clf.fit(X_train, y_train)
    
    predictions_dtc = tree_clf.predict(X_test)
    accuracy_dtc = accuracy_score(y_test, predictions_dtc)
    
    return accuracy_dtc

##RandomForestClassifier

def RFC(X_train,y_train,X_test,y_test):
    rnd_clf = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 8, n_jobs = -1)
    rnd_clf.fit(X_train,y_train)
    
    predictions_rnd_clf = rnd_clf.predict(X_test)
    accuracy_rnd = accuracy_score(y_test, predictions_rnd_clf)
    
    return accuracy_rnd

##LinearSVC

def svmlin(X_train,y_train,X_test,y_test):
    svm_clf_lin= Pipeline([
         ("scaler", StandardScaler()),
         ("linear_svc", LinearSVC(C=0.25, loss = "hinge")),
     ])

    svm_clf_lin.fit(X_train,y_train)
    
    predictions_svm_clf_lin = svm_clf_lin.predict(X_test)
    accuracy_svm_lin = accuracy_score(y_test, predictions_svm_clf_lin)
    
    return accuracy_svm_lin
    
##Poly SVM With Degree 2

def svmpolydeg2(X_train,y_train,X_test,y_test):
    poly_deg2_svm_clf = Pipeline([
     ("scaler", StandardScaler()),
     ("svm_clf", SVC(kernel="poly", degree=2, coef0=1, C=0.5))
     ])

    poly_deg2_svm_clf.fit(X_train, y_train)
    
    predictions_poly_deg2_svm_clf = poly_deg2_svm_clf.predict(X_test)
    accuracy_poly_deg2_svm_clf = accuracy_score(y_test,predictions_poly_deg2_svm_clf)
    
    return accuracy_poly_deg2_svm_clf

##Poly SVM With Degree 3

def svmpolydeg3(X_train,y_train,X_test,y_test):
    poly_deg3_svm_clf = Pipeline([
     ("scaler", StandardScaler()),
     ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=1))
     ])

    poly_deg3_svm_clf.fit(X_train, y_train)
    
    predictions_poly_deg3_svm_clf = poly_deg3_svm_clf.predict(X_test)
    accuracy_poly_deg3_svm_clf = accuracy_score(y_test,predictions_poly_deg3_svm_clf)
    
    return accuracy_poly_deg3_svm_clf
    
##Poly Gaussian RBF

def svmgaussian(X_train,y_train,X_test,y_test):
    rbf_kernel_svm_clf = Pipeline([
      ("scaler", StandardScaler()),
      ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
    ])


    rbf_kernel_svm_clf.fit(X_train, y_train)
    
    predictions_rbf_kernel_svm_clf = rbf_kernel_svm_clf.predict(X_test)
    accuracy_rbf_kernel_svm_clf = accuracy_score(y_test,predictions_rbf_kernel_svm_clf)
    
    return accuracy_rbf_kernel_svm_clf