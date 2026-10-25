"""
Model builders for MetAnoDe: CNN and LSTM hypermodels and the combined (ensemble) model.

Changes in revision 1 (affect scientific output, see revision log):
  - CNN: removed the single-unit Dense layer between CNN_Dense_1 and the output layer
    (left over from a binary implementation). It forced all CNN information through one
    number before the class output.
  - CNN: optional embedding input (vocab_size given). Nucleotides are then represented by
    learned vectors instead of their integer codes; integer codes as a single numeric
    channel impose an artificial order (A < C < G < T) on the convolution.
    Without vocab_size, the previous input (n_timesteps, 1) is kept, e.g. for old models.
  - LSTM: gradient clipping clipnorm 1.0 -> 0.5 (training collapses observed with 1.0).
  - Ensemble: submodels are matched to their inputs by name, so both branches may take
    integer tokens (CNN with embedding); rank-based matching remains as fallback.
  - training_callbacks(): ReduceLROnPlateau in addition to early stopping and checkpointing,
    so the tuned starting learning rate is lowered when validation loss stalls.

TODO:
  - L2 regularisation strength equals the tuned learning rate (kernel_regularizer=l2(lr)).
  - Hyperparameter search space and tuner settings.
"""
import os

import keras
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from sklearn.metrics import classification_report


class MetricsCallback(Callback):
    def __init__(self, test_data, y_true, summary_path):
        self.y_true = y_true
        self.test_data = test_data
        self.summary_path = summary_path

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.test_data), axis=1)
        print(classification_report(self.y_true, y_pred, output_dict=False))
        os.makedirs(os.path.dirname(self.summary_path), exist_ok=True)
        with open(self.summary_path, "a") as f:
            f.write("Model Stats:\n")
            f.write(classification_report(self.y_true, y_pred, output_dict=False))


def training_callbacks(checkpoint_path, patience_stop=8, patience_lr=3, lr_factor=0.5, min_lr=1e-5):
    """
    Standard callbacks for training CNN, LSTM and ensemble.
      ReduceLROnPlateau  lowers the learning rate by lr_factor after patience_lr epochs without
                         improvement of val_loss (before early stopping, patience_lr < patience_stop)
      EarlyStopping      stops after patience_stop epochs without improvement, restores best weights
      ModelCheckpoint    keeps the best model on disk
    """
    return [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=lr_factor, patience=patience_lr,
                                          min_lr=min_lr, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience_stop, restore_best_weights=True,
                                      verbose=1),
        keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0),
    ]


class LSTMHyperModel(kt.HyperModel):
    def __init__(self, encoder, loss_func, final_dim, final_activation):
        self.encoder = encoder
        self.loss_func = loss_func
        self.final_dim = final_dim
        self.final_activation = final_activation

    def build(self, hp):
        hp_embeddim = hp.Choice("embeddim", values=[64, 128])
        hp_units = hp.Choice("units", values=[32, 64])
        hp_dropout = hp.Choice("dropout", values=[0.25, 0.5])
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-3, 3e-4])

        model = keras.Sequential(name="model_lstm")
        model.add(keras.layers.Embedding(len(self.encoder.index_word) + 1, hp_embeddim, mask_zero=True,
                                         name="LSTM_Embed"))
        model.add(keras.layers.LSTM(hp_units, activation="tanh",
                                    kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                    return_sequences=True, name="LSTM_1"))
        model.add(keras.layers.Dropout(hp_dropout, name="LSTM_dropout"))
        model.add(keras.layers.LSTM(hp_units, activation="tanh",
                                    kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                    return_sequences=False, name="LSTM_2"))
        model.add(keras.layers.Dropout(hp_dropout, name="LSTM_dropout_2"))
        model.add(keras.layers.Dense(self.final_dim, activation=self.final_activation,
                                     kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                     name="LSTM_Dense"))

        # clipnorm 0.5 (was 1.0): long recurrent sequences (~400 steps) showed training collapses
        optimizer = keras.optimizers.Adam(learning_rate=hp_learning_rate, clipnorm=0.5)
        model.compile(optimizer=optimizer, loss=self.loss_func, metrics=["accuracy"])
        return model


class CNNHyperModel(kt.HyperModel):
    """
    vocab_size given (recommended): integer tokens (n_timesteps,) -> Embedding -> Conv1D.
    vocab_size None (previous behaviour): integer codes as one numeric channel (n_timesteps, 1).
    """

    def __init__(self, n_timesteps, n_features, loss_func, final_dim, final_activation,
                 vocab_size=None, embed_dim=8):
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.loss_func = loss_func
        self.final_dim = final_dim
        self.final_activation = final_activation
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def build(self, hp):
        hp_learning_rate = hp.Choice("learning_rate", values=[1e-3, 3e-4])
        hp_c1layerfilter = hp.Choice("filters", values=[32, 64, 128])
        hp_dropout = hp.Choice("dropout", values=[0.25, 0.5])

        model = keras.Sequential(name="model_conv1D")
        if self.vocab_size:
            # nucleotides as learned vectors; padding token 0 gets its own vector (no masking in Conv1D)
            model.add(keras.layers.Input(shape=(self.n_timesteps,), dtype="int32", name="CNN_input"))
            model.add(keras.layers.Embedding(self.vocab_size, self.embed_dim, name="CNN_Embed"))
        else:
            model.add(keras.layers.Input(shape=(self.n_timesteps, self.n_features), name="CNN_input"))
        model.add(keras.layers.Conv1D(filters=hp_c1layerfilter, kernel_size=7, activation="relu",
                                      kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                      name="CNN_Conv1D_1"))
        model.add(keras.layers.Dropout(hp_dropout, name="CNN_dropout"))
        model.add(keras.layers.Conv1D(filters=int(hp_c1layerfilter / 2), kernel_size=3, activation="relu",
                                      kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                      name="CNN_Conv1D_2"))
        model.add(keras.layers.Conv1D(filters=int(hp_c1layerfilter / 4), kernel_size=2, activation="relu",
                                      name="CNN_Conv1D_3"))
        model.add(keras.layers.Dropout(hp_dropout, name="CNN_dropout_2"))
        model.add(keras.layers.MaxPooling1D(pool_size=2, name="CNN_MaxPooling1D"))
        model.add(keras.layers.Flatten(name="CNN_Flatten"))
        model.add(keras.layers.Dense(int(hp_c1layerfilter / 2), activation="relu",
                                     kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                     name="CNN_Dense_1"))
        # (revision 1) the former single-unit layer CNN_Dense_2 was removed here
        model.add(keras.layers.Dense(self.final_dim, activation=self.final_activation,
                                     kernel_regularizer=keras.regularizers.l2(hp_learning_rate),
                                     name="CNN_Dense_3"))

        optimizer = keras.optimizers.Adam(hp_learning_rate)
        model.compile(loss=self.loss_func, optimizer=optimizer, metrics=["accuracy"])
        return model


def cnn_expects_3d(model):
    """True if the CNN input is rank-3 (None, T, C), False if rank-2 integer tokens (None, T)."""
    inp = model.input_shape
    if isinstance(inp, list):
        inp = inp[0]
    return len(inp) == 3


def make_cnn_view_np(X, cnn_model):
    """Numpy view of the encoded sequences in the form the CNN expects."""
    if cnn_expects_3d(cnn_model):
        return X[..., np.newaxis].astype("float32", copy=False)
    return X.astype("int32", copy=False)


def make_ensemble_ds(X, y, cnn_model, batch_size=16, shuffle=False, seed=None):
    """tf.data.Dataset for the ensemble; CNN input is derived on the fly from the same tokens."""
    X = X.astype("int32", copy=False)
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 100000), seed=seed, reshuffle_each_iteration=True)
    cnn_3d = cnn_expects_3d(cnn_model)

    def _map(x, y):
        x_lstm = tf.cast(x, tf.int32)
        x_cnn = tf.expand_dims(tf.cast(x, tf.float32), axis=-1) if cnn_3d else tf.cast(x, tf.int32)
        return {"ensemble_lstm_in": x_lstm, "ensemble_cnn_in": x_cnn}, y

    return ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _is_cnn(model):
    return model.name.startswith("model_conv1D") or any(l.name.startswith("CNN_") for l in model.layers)


def _model_input_rank(m):
    shp = m.input_shape
    if isinstance(shp, (list, tuple)) and len(shp) and isinstance(shp[0], tuple):
        shp = shp[0]
    return len(shp)


def create_ensemble(models, inputs, final_dim, final_activation):
    """
    Combined model from trained submodels (e.g. [cnn_model, lstm_model]); submodels are frozen,
    their output probabilities are concatenated and passed to a small trainable head.

    inputs: recommended [keras.Input(name="ensemble_lstm_in", ...), keras.Input(name="ensemble_cnn_in", ...)].
    Submodels are matched to inputs by name (CNN -> *cnn*, LSTM -> *lstm*), which works when both take
    integer tokens; otherwise by input rank (2D -> LSTM, 3D -> CNN). A single legacy input (T, 1) is
    adapted per branch.
    """
    for i, sub in enumerate(models, start=1):
        for layer in sub.layers[1:]:
            layer.trainable = False
            layer._name = f"ensemble_{i}_{layer.name}"

    if isinstance(inputs, (list, tuple)):
        model_inputs = list(inputs)
        by_name = {("cnn" if "cnn" in inp.name.lower() else "lstm" if "lstm" in inp.name.lower() else None): inp
                   for inp in model_inputs}
        model_outputs = []
        for m in models:
            key = "cnn" if _is_cnn(m) else "lstm"
            picked = by_name.get(key)
            if picked is None or len(picked.shape) != _model_input_rank(m):
                # fallback: match by rank
                picked = next((inp for inp in model_inputs if len(inp.shape) == _model_input_rank(m)), None)
            if picked is None:
                raise ValueError(f"No matching input for submodel {m.name} (rank {_model_input_rank(m)}).")
            model_outputs.append(m(picked))
        ensemble_inputs = model_inputs
    else:
        base_inp = inputs   # legacy single input (T, 1)
        lstm_in = keras.layers.Lambda(lambda t: tf.cast(tf.squeeze(t, axis=-1), tf.int32),
                                      name="ensemble_lstm_adapter")(base_inp)

        def _to_3d_float(t):
            t = tf.convert_to_tensor(t)
            if len(t.shape) == 2:
                t = tf.expand_dims(t, axis=-1)
            return tf.cast(t, tf.float32)

        cnn_in = keras.layers.Lambda(_to_3d_float, name="ensemble_cnn_adapter")(base_inp)
        cnn_int_in = keras.layers.Lambda(lambda t: tf.cast(tf.squeeze(t, axis=-1), tf.int32),
                                         name="ensemble_cnn_int_adapter")(base_inp)
        model_outputs = []
        for m in models:
            rank = _model_input_rank(m)
            if _is_cnn(m):
                model_outputs.append(m(cnn_in if rank == 3 else cnn_int_in))
            elif rank == 2:
                model_outputs.append(m(lstm_in))
            else:
                raise ValueError(f"Unsupported submodel input rank: {rank}")
        ensemble_inputs = inputs

    x = keras.layers.Concatenate(name="ensemble_concat")(model_outputs)
    x = keras.layers.Dropout(0.25, name="ensemble_dropout_1")(x)
    x = keras.layers.Dense(16, name="ensemble_hidden")(x)
    x = keras.layers.LeakyReLU(negative_slope=0.3, name="ensemble_hidden_act")(x)
    x = keras.layers.Dropout(0.25, name="ensemble_dropout_2")(x)
    out = keras.layers.Dense(final_dim, activation=final_activation, name="ensemble_logits")(x)
    return keras.models.Model(inputs=ensemble_inputs, outputs=out, name="ensemble")


def average_predictions(model1, model2, data1, data2):
    """Mean of the class probabilities of two models."""
    return np.mean([model1.predict(data1), model2.predict(data2)], axis=0)
