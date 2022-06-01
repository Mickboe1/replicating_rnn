from matplotlib.pyplot import axis
import numpy as np
from tensorflow.keras import datasets
from copy import *
import pickle

def rnn_dataset(n_historical = 5, val_size = 0.2, dataset_object_filename = "data_files/training_data_filtered.pkl"):
    try:
        with open(dataset_object_filename, 'rb') as f:
            train_X, train_Y, test_X, test_Y = pickle.load(f)
            return (train_Y, train_X), (test_Y, test_X)
    except:
        print("object inputfile read failed")
    
    
    def dynamic_eq(x, y_1):
        a = 0.2
        b = 0.1
        return x * a +  y_1 * y_1 * b + 0.8
    
    
    X = np.empty((0,n_historical,1))
    Y = np.empty((0,1))
    
    for i in range(250):
        x = []
        prev_y = 0
        for j in range(n_historical):
            x.append([i+j])
            prev_y = dynamic_eq(i+j, x[-1][0])
        print(x, prev_y)
        X = np.append(X, np.array([x]), axis=0)
        Y = np.append(Y, np.array([[prev_y]]),axis=0)
        
    # print(X.shape)
    # print(Y.shape)
    # exit()
            

    print("done processing input data, starting shuffeling")
            
    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    Y = Y[shuffler]

    print("Done shuffeling, saving result to file.")

    size = len(X)
    train_X = X[:int(size-size*val_size)]
    train_Y = Y[:int(size-size*val_size)]
    test_X = X[int(size-size*val_size):]
    test_Y = Y[int(size-size*val_size):]

    # Saving the objects:
    with open(dataset_object_filename, 'wb') as f:
        pickle.dump([train_X, train_Y, test_X, test_Y], f)
    return (train_Y, train_X), (test_Y, test_X)

