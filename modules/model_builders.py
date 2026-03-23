#TODO first layer max 128?
# CHanges in version used on con-compute1: 
# - dropout usage in LSTM
# - less values in hp choices: removed 128 and choices from droput
# - add binary_crossentropy switch 4D: sparse

import keras
import keras_tuner as kt
import tensorflow as tf  
from keras.callbacks import Callback
from sklearn.metrics import classification_report
import numpy as np

class MetricsCallback(Callback):
    def __init__(self, test_data, y_true, name):
        self.y_true = y_true
        self.test_data = test_data
        self.name = name

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.test_data)
        y_pred = np.argmax(y_pred, axis=1)
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
        hp_learning_rate = hp.Choice('learning_rate', values=[3e-4]) # 1e-3, 1e-4, 

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
        hp_learning_rate = hp.Choice('learning_rate', values=[3e-4]) #1e-3, 1e-4, 
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
    """
    Build an ensemble from a list of trained models (e.g., [cnn_model, lstm_model]).

    Parameters
    ----------
    models : list[keras.Model]
        Submodels already built (and typically trained). One expects 3D input (CNN),
        another expects 2D int tokens (LSTM with Embedding).
    inputs : keras.Input or list/tuple[keras.Input]
        Recommended: [inp_lstm(int32, (T,)), inp_cnn(float32, (T,1))]
        Legacy: a single Input (T,1); the function will adapt per-branch automatically.
    final_dim : int
        Number of output classes.
    final_activation : str
        'softmax' for multi-class, 'sigmoid' for binary.
    """
    # Freeze and namespace submodel layers (skip each model's own Input layer)
    for i, sub in enumerate(models, start=1):
        for layer in sub.layers[1:]:
            layer.trainable = False
            layer._name = f"ensemble_{i}_{layer.name}"

    # Helper: expected input rank of a submodel (Sequential single-input case)
    def _model_input_rank(m):
        shp = m.input_shape
        if isinstance(shp, (list, tuple)) and len(shp) and isinstance(shp[0], tuple):
            shp = shp[0]
        return len(shp)

    # Multi-input path: route by rank (2D -> LSTM branch, 3D -> CNN branch)
    if isinstance(inputs, (list, tuple)):
        model_inputs = list(inputs)
        model_outputs = []
        for m in models:
            m_rank = _model_input_rank(m)
            picked_inp = None
            for inp in model_inputs:
                if len(inp.shape) == m_rank:
                    picked_inp = inp
                    break
            if picked_inp is None:
                raise ValueError(
                    f"Could not find a matching input rank for model expecting rank {m_rank}."
                )
            model_outputs.append(m(picked_inp))
        ensemble_inputs = model_inputs
    else:
        # Legacy single-input path: adapt per branch.
        base_inp = inputs  # expected (T,1) or convertible

        # LSTM branch: squeeze feature dim -> (T,) and cast to int32
        lstm_in = keras.layers.Lambda(
            lambda t: tf.cast(tf.squeeze(t, axis=-1), tf.int32),
            name="ensemble_lstm_adapter"
        )(base_inp)

        # CNN branch: ensure (T,1) and cast to float32
        def _to_3d_float(t):
            t = tf.convert_to_tensor(t)
            if len(t.shape) == 2:
                t = tf.expand_dims(t, axis=-1)
            return tf.cast(t, tf.float32)

        cnn_in = keras.layers.Lambda(_to_3d_float, name="ensemble_cnn_adapter")(base_inp)

        model_outputs = []
        for m in models:
            m_rank = _model_input_rank(m)
            if m_rank == 2:
                model_outputs.append(m(lstm_in))
            elif m_rank == 3:
                model_outputs.append(m(cnn_in))
            else:
                raise ValueError(f"Unsupported submodel input rank: {m_rank}")

        ensemble_inputs = inputs  # single Input remains the model signature

    # Fuse heads
    x = keras.layers.Concatenate(name="ensemble_concat")(model_outputs)
    x = keras.layers.Dropout(0.25, name="ensemble_dropout_1")(x)
    x = keras.layers.Dense(16, name="ensemble_hidden")(x)
    x = keras.layers.LeakyReLU(negative_slope=0.3, name="ensemble_hidden_act")(x)
    x = keras.layers.Dropout(0.25, name="ensemble_dropout_2")(x)
    out = keras.layers.Dense(final_dim, activation=final_activation, name="ensemble_logits")(x)

    model = keras.models.Model(inputs=ensemble_inputs, outputs=out, name="ensemble")
    return model

def average_predictions(model1, model2, data1, data2):
    # Get predictions from both models
    predictions1 = model1.predict(data1)
    predictions2 = model2.predict(data2)
    
    # Average the predictions
    averaged_predictions = np.mean([predictions1, predictions2], axis=0)
    
    return averaged_predictions