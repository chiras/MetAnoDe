#TODO first layer max 128?
# CHanges in version used on con-compute1: 
# - dropout usage in LSTM
# - less values in hp choices: removed 128 and choices from droput
# - add binary_crossentropy switch 4D: sparse

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback 
from sklearn.metrics import classification_report

class MetricsCallback(Callback):
    def __init__(self, test_data, y_true, name):
        self.y_true = y_true
        self.test_data = test_data
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.test_data)
        y_pred = tf.argmax(y_pred,axis=1)
        report_dictionary = classification_report(self.y_true, y_pred, output_dict = True)
        print(classification_report(self.y_true,y_pred,output_dict=False)) 

        summary_path = f"models/{self.name}_Stats.txt"

        with open(summary_path, 'a') as f:
            f.write('Model Stats:\n')
            f.write(classification_report(self.y_true,y_pred,output_dict=False)) 



class LSTMHyperModel(kt.HyperModel):
    def __init__(self, encoder, loss_func,final_dim,final_activation):
        self.encoder = encoder
        self.loss_func = loss_func
        self.final_dim=final_dim
        self.final_activation=final_activation

    def build(self, hp):
        hp_embeddim = hp.Choice('embeddim', values=[128]) #256 // 64, 128
        hp_units = hp.Choice('units',values=[32, 64]) # 4-64
        hp_dropout = hp.Choice('dropout', values=[0.25, 0.5]) #seq 1-5 0.1, 
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4]) # 1e-2, 

        model = keras.Sequential()
        model.add(keras.layers.Embedding(len(self.encoder.index_word) + 1, hp_embeddim, mask_zero=True, name="LSTM_Embed"))
        model.add(keras.layers.LSTM(hp_units, 
                                    activation="tanh", 
                                    kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                    return_sequences=True, 
                                    name="LSTM_1")) # 0.2 , dropout=hp_dropout/10
        model.add(keras.layers.Dropout(hp_dropout, 
                                       name="LSTM_dropout"))
        model.add(keras.layers.LSTM(hp_units, 
                                    activation="tanh", 
                                    kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                    return_sequences=False, 
                                    name="LSTM_2"))
        model.add(keras.layers.Dropout(hp_dropout, 
                                       name="LSTM_dropout_2"))
        model.add(keras.layers.Dense(self.final_dim, 
                                    kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                    activation=self.final_activation, 
                                    name="LSTM_Dense"))

        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate) # 1e-2

        model.compile(optimizer=optimizer, loss=self.loss_func,
                    metrics=["accuracy"])
        return model

class CNNHyperModel(kt.HyperModel):

    def __init__(self, n_timesteps, n_features, loss_func,final_dim,final_activation):
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.loss_func = loss_func
        self.final_dim=final_dim
        self.final_activation=final_activation

    def build(self, hp):   
        # parameters to be tuned
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4]) #1e-2, 
        hp_c1layerfilter = hp.Choice('filters', values=[32, 64]) #, 256 , 128
        hp_dropout = hp.Choice('dropout', values=[0.25, 0.5]) #, 256 0.1, 

        model = keras.Sequential(name="model_conv1D")
        model.add(keras.layers.Input(shape=(self.n_timesteps,
                                            self.n_features), 
                                     name="CNN_input"))
        #model.add(keras.layers.Masking(mask_value=0.0, name="CNN_mask"))
        model.add(keras.layers.Conv1D(filters=hp_c1layerfilter, 
                                      kernel_size=7, 
                                      activation='relu',
                                      kernel_regularizer=keras.regularizers.l2(hp_learning_rate), 
                                      name="CNN_Conv1D_1"))
        model.add(keras.layers.Dropout(hp_dropout, name="CNN_dropout"))
        model.add(keras.layers.Conv1D(filters=int(hp_c1layerfilter/2), 
                                      kernel_size=3, 
                                      activation='relu', 
                                      kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                      name="CNN_Conv1D_2"))
        model.add(keras.layers.Conv1D(filters=int(hp_c1layerfilter/4), kernel_size=2, activation='relu', name="CNN_Conv1D_3"))  
        model.add(keras.layers.Dropout(hp_dropout, name="CNN_dropout_2"))
        model.add(keras.layers.MaxPooling1D(pool_size=2, name="CNN_MaxPooling1D"))
        model.add(keras.layers.Flatten(name="CNN_Flatten"))
        model.add(keras.layers.Dense(int(hp_c1layerfilter/2), 
                                     activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(hp_learning_rate), 
                                     name="CNN_Dense_1"))
        model.add(keras.layers.Dense(self.n_features, name="CNN_Dense_2"))
        model.add(keras.layers.Dense(self.final_dim, name="CNN_Dense_3", 
                                     activation=self.final_activation,
                                     kernel_regularizer=keras.regularizers.l2(hp_learning_rate)))

        #optimizer = keras.optimizers.RMSprop(hp_learning_rate)
        optimizer = keras.optimizers.Adam(hp_learning_rate)

        model.compile(loss=self.loss_func,optimizer=optimizer,metrics=['accuracy'])
        return model
    
def create_ensemble(models, inputs, final_dim, final_activation):
    for i in range(len(models)):
        # Each model is a Net object
        model = models[i]
        for layer in model.layers[1:]:
            layer.trainable = False
            layer.name = 'ensemble_' + str(i+1) + '_' + layer.name

    stack_outputs = [model(inputs) for model in models]
    x = keras.layers.Concatenate()(stack_outputs) 
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(16)(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(final_dim, activation=final_activation)(x)

    print(inputs)
    model = keras.models.Model(inputs=inputs, outputs=x, name='ensemble')

    return model

def average_predictions(model1, model2, data1, data2):
    # Get predictions from both models
    predictions1 = model1.predict(data1)
    predictions2 = model2.predict(data2)
    
    # Average the predictions
    averaged_predictions = np.mean([predictions1, predictions2], axis=0)
    
    return averaged_predictions