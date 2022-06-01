import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from tensorflow import keras
from model_manager import *
from data_manager import *
from keras.callbacks import LearningRateScheduler
from keras.callbacks import CSVLogger
from validation.validation_plotter import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

checkpoint_path = get_model_file_location()
checkpoint_dir = os.path.dirname(checkpoint_path)
n_hist = 5
initial_epoch = 0

class PlotProgress(keras.callbacks.Callback):
    def __init__(self, n_historical = 20, initial_counter = 0):
        super().__init__()
        self.n_historical = n_historical
        self.counter = initial_counter

    def on_train_begin(self, logs={}):
        self.p = plotter(self.n_historical)

    def on_epoch_end(self, batch, logs={}):
        self.counter += 1
        if self.counter % 5 == 0:
            # self.p.plot_from_file()

            # self.p.plot_dataset(self.model.predict(np.array(self.p.input_data)))

            
            f = open("data_files/epochtracker.dat", 'w')
            f.write(str(self.counter))
            f.close()

def custom_LearningRate_schedular(epoch):
    initial_lrate = 0.001
    k = 0.001
    lr = initial_lrate * exp(-k*epoch)
    return lr


csv_logger = CSVLogger('data_files/training.csv', append=True, separator=';')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq='epoch', 
						 save_best_only=True,
						 mode='min')

lr_callback = LearningRateScheduler(custom_LearningRate_schedular)

try:
    with open("data_files/epochtracker.dat", 'r') as epoch_data_file:
        initial_epoch = int(epoch_data_file.readlines()[0])
except:
    print("Starting new training session")


(train_Y, train_X), (test_Y, test_X) = rnn_dataset(n_historical = n_hist)



model = create_model_rnn(n_historical=n_hist)
history = model.fit(train_X, train_Y, epochs=3000, 
                    validation_data=(test_X, test_Y),
                    callbacks=[cp_callback, lr_callback, csv_logger, PlotProgress(n_historical = n_hist, initial_counter=initial_epoch)], 
                    batch_size=128, initial_epoch=initial_epoch)


plt.close('all')
plt.ylim(0, 1)
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='lower right')
plt.savefig('training.png')
