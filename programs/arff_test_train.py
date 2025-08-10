import os
import glob
import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC

import models

source_directory = 'C:\\Users\\praneeth\\Desktop\\proj_iwnn\\arff_test_train'

for directory in os.listdir(source_directory):
    directory_path = os.path.join(source_directory, directory)
    print(directory_path)

    if os.path.isdir(directory_path):
        arff_files = glob.glob(os.path.join(directory_path, '*.arff'))
        
        data1, meta1 = arff.loadarff(arff_files[0])
        df1 = pd.DataFrame(data1)

        data2, meta2 = arff.loadarff(arff_files[1])
        df2 = pd.DataFrame(data2)

        df = pd.concat([df1, df2])
        
        y = df['clase']                                            
        X = df.drop('clase',axis=1)
          
        X = np.array(X)
        y = np.array(y).squeeze()
        y = y.astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
        
        ##TRAINING
        
        a_dtc = models.DTC(X_train,y_train,X_test,y_test)
        a_rf = models.RFC(X_train,y_train,X_test,y_test)
        a_svm_l = models.svmlin(X_train,y_train,X_test,y_test)
        a_svm_d2 = models.svmpolydeg2(X_train,y_train,X_test,y_test)
        a_svm_d3 = models.svmpolydeg3(X_train,y_train,X_test,y_test)
        a_svm_g = models.svmgaussian(X_train,y_train,X_test,y_test)
        
        accuracies = np.array([a_dtc, a_rf, a_svm_l, a_svm_d2, a_svm_d3, a_svm_g])
        accuracies = accuracies*100

        
        output_file = 'C:\\Users\\praneeth\\Desktop\\proj_iwnn\\CSV\\test_train_outputs.csv'
        with open(output_file, 'ab') as file:
            np.savetxt(file, accuracies.reshape(1, -1), delimiter=',', fmt='%.3f', newline='\n')
