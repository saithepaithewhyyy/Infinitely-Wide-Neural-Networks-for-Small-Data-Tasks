import os
from scipy.io import arff
import pandas as pd
import numpy as np

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from keras.utils import to_categorical

tf.random.set_seed(42)

import warnings
warnings.filterwarnings('ignore')

directory = 'C:\\Users\\praneeth\\Desktop\\proj_iwnn\\arff'

for filename in os.listdir(directory):
    
    file_path = os.path.join(directory,filename)

    with open(file_path,'r') as arff_file:
        print(filename)
        
        data = arff.loadarff(arff_file)
          
        df = pd.DataFrame(data[0])
          
        y = df['clase']                                            
        X = df.drop('clase',axis=1)
          
        X = np.array(X)
        y = np.array(y).squeeze()
        y = y.astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        
        
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(100, activation = 'elu', kernel_initializer = tf.keras.initializers.HeNormal(), input_shape=[X_train.shape[1]]))

        for i in range(10):
            model.add(tf.keras.layers.Dense(100, activation = 'elu', kernel_initializer = tf.keras.initializers.HeNormal()))
            model.add(tf.keras.layers.Dropout(0.15))

        model.add(tf.keras.layers.Dense(y_train.shape[1], activation = 'sigmoid', kernel_initializer = tf.keras.initializers.HeNormal()))
        
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics=['accuracy'])

        history = model.fit(X_train, y_train, epochs=200, verbose = 2)

        loss, accuracy = model.evaluate(X_test, y_test)
        accuracy = accuracy * 100
        
        result = np.array([loss, accuracy])
        
        output_file = 'C:\\Users\\praneeth\\Desktop\\proj_iwnn\\CSV\\nn_outputs.csv'
        with open(output_file, 'ab') as file:
            np.savetxt(file, result.reshape(1, -1), delimiter=',', fmt='%.3f', newline='\n')