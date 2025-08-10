import os

from scipy.io import arff
import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC
    
import models

## DATA LOADING

directory = 'C:\\Users\\praneeth\\Desktop\\proj_iwnn\\arff'

for filename in os.listdir(directory):
    
    file_path = os.path.join(directory,filename)

    with open(file_path,'r') as arff_file:
        data = arff.loadarff(arff_file)
          
        df = pd.DataFrame(data[0])
          
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
        
        accuracies = np.array([filename, a_dtc, a_rf, a_svm_l, a_svm_d2, a_svm_d3, a_svm_g])
        accuracies = accuracies*100

        print(filename)
        
        output_file = 'C:\\Users\\praneeth\\Desktop\\proj_iwnn\\CSV\\test_train_outputs.csv'
        with open(output_file, 'ab') as file:
            np.savetxt(file, accuracies.reshape(1, -1), delimiter=',', fmt='%.3f', newline='\n')
