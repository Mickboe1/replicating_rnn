import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import keras.backend as K

def get_model_file_location():
    return "model_files/checkpoint.ckpt"



def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """ 
    weights = [0.3, 0.3, 0.3, 0.1] #RPM, V, C, T
    return K.sqrt(K.mean(K.square((y_pred - y_true)*weights), axis=-1))

    return model


def create_model_rnn(n_historical):
    model = models.Sequential()
    model.add(layers.SimpleRNN(units=1, input_shape = (n_historical,1), activation='selu'))
    # model.add(layers.Dense(units=48, activation="selu"))
    # model.add(layers.Dense(units=4, activation="linear"))


    opt = tf.keras.optimizers.Adam(
        learning_rate=0.15,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam")
    
    
    model.compile(optimizer = opt, 
                loss = euclidean_distance_loss)

    checkpoint_path = get_model_file_location()

    try:
        model.load_weights(checkpoint_path)
        print("Succesfully loaded checkpoint file.")
    except:
        print("No models exists, creating new one")

    return model

