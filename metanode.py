import os
import argparse
import time

start_time = time.time()

parser = argparse.ArgumentParser(description="Amplicon Anomaly Detection")
parser.add_argument('-db', dest='true_file', required=True, help="True Amplicon database FASTA file")
parser.add_argument('-query', dest='query_file', required=True, help="Amplicon database FASTA file")
parser.add_argument('-p', dest='project_name', required=True, help="Model name")
parser.add_argument('-4c', dest='four_classes', action='store_true', required=False, help="Switch from multiclass to binary (DEPRECEATED)")
parser.add_argument('-r', dest='recalibrate', action='store_true', required=False, help="Switch to tuner recalibration")
parser.add_argument('-v', dest='verbose', action='store_true', required=False, help="Switch to verbose mode")
parser.add_argument('-t', dest='threads', required=False, help="Number of threads")
parser.add_argument('-ot', dest='offtargets', required=False, help="Number of threads")

args = parser.parse_args()

verbose = True if args.verbose else False
if verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
else: 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import text
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import keras_tuner as kt
import pickle

from modules import helper
from modules import model_builders

import numpy as np 
import pandas as pd 

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def write_to_fasta(df: pd.DataFrame, filepath: str) -> None:
    with open(filepath, 'w') as file:
        for index, row in df.iterrows():
            # Create the FASTA header
            fasta_header = f">{row['headers']}"
            for col in df.columns:
                if col not in ['headers', 'sequences']:
                    fasta_header += f"; {col}={row[col]}"
            fasta_header += "\n"
            
            # Write the header and sequence to the file
            file.write(fasta_header)
            file.write(f"{row['sequences']}\n")

def process_dataframe(df: pd.DataFrame, labels: dict, model: str) -> pd.DataFrame:
    # Rename the headers 
    renamed_columns = {col: f"{model}_{labels[col]}" for col in df.columns}
    df = df.rename(columns=renamed_columns)
    
    # Create a new column "model_class" where the label of the highest value of 0-3 is the column value
    label_columns = list(renamed_columns.values())
    df[f"{model}_class"] = df[label_columns].idxmax(axis=1)
    
    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Define the order of the fixed columns
    fixed_columns = ['headers', 'EN_class', 'LSTM_class', 'CNN_class']
    
    # Identify any additional columns that are not in the fixed columns
    additional_columns = [col for col in df.columns if col not in fixed_columns]
    
    # Create the new column order
    new_column_order = fixed_columns + additional_columns
    
    # Reorder the DataFrame columns
    return df[new_column_order]

query_base_name = os.path.basename(args.query_file)
query_name = os.path.splitext(query_base_name)[0]

threads=int(args.threads)    
    

if args.threads is not None:
    threads = int(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)

if args.four_classes:
    balance_4C = True
    loss_func='sparse_categorical_crossentropy'
else:
    balance_4C = False
    loss_func='binary_crossentropy' 

project_name=args.project_name 
max_epoch=10
max_epoch_ensemble=10
max_epoch_tuner=5
learning_rate=0.001
opt = keras.optimizers.Adam(learning_rate=learning_rate) # ensemble only


shuffle_validation = False
use_size_weights = True

if loss_func=='binary_crossentropy':
    final_activation="sigmoid"
   #  final_dim=1
else:
    final_activation="softmax"
    # final_dim=2
    # if balance_4C:
    #     final_dim=4  

if (balance_4C):
#    labels = {0: 'positive', 1: 'singletons', 2: 'no >95 hit to global db', 3: 'chimera'}
    labels = {0: 'positive', 1: 'substitution', 2: 'indels', 3: 'chimera'}
else:
    labels = {0: 'positive', 1: 'false'}

# labels = {0: 'positive', 1: 'substitution', 2: 'indels', 3: 'chimera', 4: 'mito', 5 : 'chloro'} # TODO fix in config
labels = {0: 'positive', 1: 'substitution', 2: 'indels', 3: 'chimera', 4: 'fungi'} # TODO fix in config

# check existing models:

model_name_en= f"{project_name}_Ensemble"
model_path_en = f"models/{model_name_en}.keras"
model_name_cnn= f"{project_name}_CNN"
model_path_cnn = f"models/{model_name_cnn}.keras"
model_name_lstm= f"{project_name}_LSTM"
model_path_lstm = f"models/{model_name_lstm}.keras"
token_path = f"models/{model_name_en}.token"
config_path = f"models/{model_name_en}.config"

model_exist_en = os.path.isfile(model_path_en)
model_exist_cnn = os.path.isfile(model_path_cnn)
model_exist_lstm = os.path.isfile(model_path_lstm)
token_exist = os.path.isfile(token_path)
config_exist = os.path.isfile(config_path)

#
def split_dataset(dataset, test_ratio=0.25):
    test_indices = np.random.rand(len(dataset)) < test_ratio
    return dataset[~test_indices], dataset[test_indices]

sequences_dict_query = helper.read_fasta(args.query_file, 99, 99)

def logger():
    global project_name
    global start_time
    elapsed_time = int(time.time() - start_time)
    formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    string = f"[AD-DNN: {project_name}: {formatted_time}]"
    return string


# load in reference data
if not all([model_exist_en, model_exist_cnn, model_exist_lstm, token_exist, config_exist]):
    print(f"\n{logger()} Not all required models and configs are available, regenerating necessary ones")
    print(f"{logger()} This might take a while, depending on whether models are missing or need to be tuned...")

    # TODO check for ref db file
    config = {}

    sequences_dict_true = helper.read_fasta(args.true_file, 0, 0)
    print(sequences_dict_true) if verbose else None

    tr_len=len(sequences_dict_true)
    max_rows = int(tr_len*2)

    if not balance_4C:
        max_rows = int(max_rows/3)

    print(f"\n{logger()} Received {len(sequences_dict_true)} original sequences (Class 0)")

    sequences_dict_f1_balanced = helper.create_artificial_errorate(sequences_dict_true, max_rows, "subst")
    print(f"{logger()} Created {len(sequences_dict_f1_balanced)} artifical high substitution rate sequences (Class 1)")

    sequences_dict_f2_balanced = helper.create_artificial_errorate(sequences_dict_true, max_rows, "indel")
    print(f"{logger()} Created {len(sequences_dict_f2_balanced)} artifical high indel rate sequences (Class 2)")

    sequences_dict_f3_balanced = helper.create_artificial_chimera(sequences_dict_true, max_rows)
    print(f"{logger()} Created {len(sequences_dict_f3_balanced)} artifical chimeric sequences (Class 3)")

    sequences_dict_true_lowsubst = helper.create_artificial_errorate(sequences_dict_true, int(tr_len/2), "lowsubst")
    sequences_dict_true_lowindel = helper.create_artificial_errorate(sequences_dict_true, int(tr_len/2), "lowindel")
    sequences_dict_true_balanced = pd.concat([sequences_dict_true, sequences_dict_true_lowsubst, sequences_dict_true_lowindel], ignore_index=True)
    print(f"{logger()} Created {len(sequences_dict_true_lowsubst)} + {len(sequences_dict_true_lowindel)} artifical low error sequences (Class 0)")

    if args.offtargets is not None:
        split_list = args.offtargets.split(',')
        i=4
        sequences_dict_offtarget_all = pd.DataFrame()
        for offtarget in split_list:
            sequences_dict_offtarget = helper.read_fasta(offtarget, 0, i)
            ot_len=len(sequences_dict_offtarget)
            if ot_len > max_rows:
                sequences_dict_offtarget = helper.sample_dataframe(sequences_dict_offtarget, max_rows)
                type_ot = "(downsampled)"
            else:
                missing_rows = max_rows-ot_len
                sequences_dict_ot_lowsubst = helper.create_artificial_errorate(sequences_dict_offtarget, int(missing_rows/2), "lowsubst")
                sequences_dict_ot_lowindel = helper.create_artificial_errorate(sequences_dict_offtarget, int(missing_rows/2), "lowindel")
                sequences_dict_offtarget = pd.concat([sequences_dict_offtarget, sequences_dict_ot_lowsubst, sequences_dict_ot_lowindel], ignore_index=True)
                type_ot = "(upsampled)"
            len_ot = len(sequences_dict_offtarget)
            sequences_dict_offtarget["Target4D"]= i 
            sequences_dict_offtarget["Target"]= 1
            labels[i] = f"OffTarget-{i}"
            print(f"{logger()} Added {len_ot} {type_ot} Off-target Class {i} ({offtarget})")
            i=i+1
            sequences_dict_offtarget_all = pd.concat([sequences_dict_offtarget_all, sequences_dict_offtarget], ignore_index=True)
            del sequences_dict_offtarget

    else:
        sequences_dict_offtarget_all = helper.sample_dataframe(sequences_dict_true_balanced, 0)

    X_train_balanced = pd.concat([sequences_dict_true_balanced, sequences_dict_f1_balanced, sequences_dict_f2_balanced, sequences_dict_f3_balanced,sequences_dict_offtarget_all], ignore_index=True)
    config["max_len"] = max(len(seq) for seq in (X_train_balanced["sequences"]))
    config["output_dim"] = final_dim = X_train_balanced["Target4D"].nunique()
    config["labels"] = labels
    print(f"{logger()} Balanced to ({final_dim} class mode: {balance_4C}): {len(sequences_dict_true_balanced)}, {len(sequences_dict_f1_balanced)}, {len(sequences_dict_f2_balanced)}, {len(sequences_dict_f3_balanced)}, {len(sequences_dict_offtarget_all)}")
    print(f"{logger()} Class labels: {labels}")

    # remove temporary data from memory
    del sequences_dict_f1_balanced
    del sequences_dict_f2_balanced
    del sequences_dict_f3_balanced
    del sequences_dict_true_lowsubst
    del sequences_dict_true_lowindel
    del sequences_dict_true_balanced
    del sequences_dict_offtarget_all
    print(f"{logger()} Removed temporary data from memory")

    # Shuffle the concatenated DataFrame
    X_train_balanced = shuffle(X_train_balanced, random_state=42).reset_index(drop=True)
    X_train_balanced = shuffle(X_train_balanced, random_state=42).reset_index(drop=True)
    print(f"{logger()} Shuffled data")

    X_train_balanced, X_valid_balanced = split_dataset(X_train_balanced)
    print(f"{logger()} Split data to train and validation")

    # if verbose:
    #     X_train_balanced = helper.sample_dataframe(X_train_balanced, 7500)
    #     X_valid_balanced = helper.sample_dataframe(X_valid_balanced, 2500)
    #     max_epoch=4

    print(X_train_balanced) if verbose else None

    # Separate Features from Labels
    X_train_final = X_train_balanced.drop(columns=['headers','Target','Target4D','sizes'])  # Features
    X_valid_final = X_valid_balanced.drop(columns=['headers','Target','Target4D','sizes'])  # Features


    if (balance_4C):
        y_train = X_train_balanced['Target4D']  # Target variable
        y_valid = X_valid_balanced['Target4D']  # Target variable
    else:
        y_train = X_train_balanced['Target']  # Target variable
        y_valid = X_valid_balanced['Target']  # Target variable

    # Convert DataFrame columns to lists
    X_train_list = X_train_final['sequences'].tolist()
    X_valid_list = X_valid_final['sequences'].tolist()

    kmer=False
    if kmer:
        # kmer level
        X_train_list = helper.split_into_kmers(X_train_list, 8, 1)
        X_valid_list = helper.split_into_kmers(X_valid_list, 8, 1)

        print(X_train_list[1]) if verbose else None

    # remove temporary data
    del X_train_balanced
    #del X_valid_balanced
    del X_train_final
    del X_valid_final
    print(f"{logger()} Removed temporary data from memory")
    
    # Fit tokenizer on training data
    print(f"{logger()} Encoding data")

    if not token_exist:
        if kmer:
            encoder = text.Tokenizer(char_level=False) 
        else:
            encoder = text.Tokenizer(char_level=True)

        encoder.fit_on_texts(X_train_list)
        with open(token_path, 'wb') as token:
                pickle.dump(encoder, token, protocol=pickle.HIGHEST_PROTOCOL)
    else:    
        with open(token_path, 'rb') as token:
            encoder = pickle.load(token)    
    
    # Convert text data to sequences
    X_train_encoded = encoder.texts_to_sequences(X_train_list)
    X_valid_encoded = encoder.texts_to_sequences(X_valid_list)

    # Inspect the encoding. Add padding token to the encoding
    word_index = encoder.word_index
    encoded_characters = pd.DataFrame(list(word_index.items()), columns=['Character', 'Encoding'])
    encoded_characters.loc[len(encoded_characters)] = ['<PAD>', 0]
    print(encoded_characters) if verbose else None

    # Pad sequences
    print(f"{logger()} Padding data")

    X_train_encoded =  keras.preprocessing.sequence.pad_sequences(X_train_encoded, maxlen=config["max_len"], padding='post')
    X_valid_encoded =  keras.preprocessing.sequence.pad_sequences(X_valid_encoded, maxlen=config["max_len"], padding='post')  

    # convert to numpy arrays
    print(f"{logger()} Converting data")
    X_train_padded = np.array(X_train_encoded)
    X_valid_padded = np.array(X_valid_encoded)

    y_train = np.array(y_train)
    y_valid = np.array(y_valid)

    if loss_func=='binary_crossentropy':
        y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
        y_valid = np.asarray(y_valid).astype('float32').reshape((-1,1))

    if not config_exist:
        with open(config_path, 'wb') as file:
            # Serialize and write the variable to the file
            pickle.dump(config, file)
    else:
        with open(config_path, 'rb') as file:
            # Deserialize and retrieve the variable from the file
            config = pickle.load(file)

    #### for CNN: 
    sample_size = X_train_padded.shape[0] # number of samples in train set
    time_steps  = X_train_padded.shape[1] # number of features in train set
    input_dimension = 1               # each feature is represented by 1 number

    X_padded_reshaped = X_train_padded.reshape(sample_size,time_steps,input_dimension)
    print(X_padded_reshaped.shape) if verbose else None

    # remove temporary data
    del X_train_list
    del X_valid_list
    del X_train_encoded
    del X_valid_encoded
    #del X_train_padded
    #del X_valid_padded
    print(f"{logger()} Removed temporary data from memory")

with open(token_path, 'rb') as token:
    encoder = pickle.load(token)    

with open(config_path, 'rb') as file:
    # Deserialize and retrieve the variable from the file
    config = pickle.load(file)


final_dim = config["output_dim"]
print(f"{logger()} Modeling {final_dim} Classes")
print(f"{logger()} Class labels: {labels}")

# prepare query
X_query_final = sequences_dict_query.drop(columns=['headers','Target','Target4D','sizes'])  # Features
X_query_list = X_query_final['sequences'].tolist()
X_query_encoded = encoder.texts_to_sequences(X_query_list)
X_query_encoded =  keras.preprocessing.sequence.pad_sequences(X_query_encoded, maxlen=config["max_len"], padding='post')
X_query_padded = np.array(X_query_encoded)


model_name= f"{project_name}_CNN"
model_path = f"models/{model_name}.keras"
summary_path = f"models/{project_name}_Stats.txt"

if model_exist_cnn:
    # Try to load the saved model
    cnn_model = load_model(model_path)
    cnn_model.compile(optimizer=opt, loss=loss_func)
    print(f"\n{logger()} CNN Model loaded successfully.")
else:
    print(f"\n{logger()} CNN Model file not found. Creating a new model...")
    cnn_hyper_model = model_builders.CNNHyperModel(n_timesteps=X_padded_reshaped.shape[1],loss_func=loss_func, final_dim=final_dim, final_activation=final_activation,n_features  = X_padded_reshaped.shape[2])
    metrics_callback = model_builders.MetricsCallback(test_data=X_valid_padded, y_true=y_valid, name=project_name)

    cnn_tuner = kt.Hyperband(cnn_hyper_model,
                        objective='val_accuracy',
                        max_epochs=max_epoch_tuner,
                        factor=3,
                        directory='tuner',
                        project_name=model_name)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        
    cnn_tuner.search(X_train_padded, y_train, epochs=max_epoch_tuner, batch_size=64, validation_data=(X_valid_padded, y_valid), callbacks=[stop_early])
    best_hps=cnn_tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'{logger()} CNN Hyperparameter Tuning completed\n\n')
    print(f"{logger()} {best_hps.values}")

    cnn_model = cnn_tuner.hypermodel.build(best_hps)
    cnn_history = cnn_model.fit(X_train_padded, y_train, batch_size=64, epochs=max_epoch, validation_data=(X_valid_padded, y_valid))
    print(f'{logger()} CNN Model completed\n\n')

    val_acc_per_epoch = cnn_history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print(f'{logger()} Best epoch: %d' % (best_epoch,))

    cnn_model.save(model_path)  # Save the model to a HDF5 file
    print(f"{logger()} CNN Model saved successfully.")

    helper.plot_history(cnn_history,model_name, best_hps.values)
    helper.save_summary(cnn_model, cnn_history, best_hps, model_name)

    with open(summary_path, 'a') as f:
        f.write('##### CNN #####:\n')

    cnn_model.fit(X_train_padded, y_train, batch_size=64, epochs=1, validation_data=(X_valid_padded, y_valid), callbacks=[metrics_callback])

print(f"{logger()} CNN architecture: ") if verbose else None
print(cnn_model.summary()) if verbose else None

model_name= f"{project_name}_LSTM"
model_path = f"models/{model_name}.keras"

if model_exist_lstm:
    # Try to load the saved model
    lstm_model = load_model(model_path)
    lstm_model.compile(optimizer='adam', loss=loss_func)
    print(f"\n{logger()} LSTM Model loaded successfully.")
else:
    print(f"\n{logger()} LSTM Model file not found. Creating a new model...")
    lstm_hyper_model = model_builders.LSTMHyperModel(encoder=encoder,loss_func=loss_func, final_dim=final_dim, final_activation=final_activation)
    metrics_callback = model_builders.MetricsCallback(test_data=X_valid_padded, y_true=y_valid, name=project_name)

    lstm_tuner = kt.Hyperband(lstm_hyper_model,
                        objective='val_accuracy',
                        max_epochs=max_epoch_tuner,
                        factor=3,
                        directory='tuner',
                        project_name=model_name)

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    lstm_tuner.search(X_train_padded, y_train, batch_size=64, epochs=max_epoch_tuner, validation_data=(X_valid_padded, y_valid), callbacks=[stop_early])
    best_hps=lstm_tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'{logger()} LSTM Hyperparameter Tuning completed\n\n')
    print(f"{logger()} {best_hps.values}")

    lstm_model = lstm_tuner.hypermodel.build(best_hps)
    lstm_history = lstm_model.fit(X_train_padded, y_train, batch_size=64, epochs=max_epoch, validation_data=(X_valid_padded, y_valid))

    val_acc_per_epoch = lstm_history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print(f'{logger()} Best epoch: %d' % (best_epoch,))
    print(f'{logger()} LSTM Model completed\n\n')

    lstm_model.save(model_path)  # Save the model to a HDF5 file
    print(f"{logger()} Model saved successfully.")

    helper.plot_history(lstm_history,model_name, best_hps.values)
    helper.save_summary(lstm_model, lstm_history, best_hps, model_name)

    with open(summary_path, 'a') as f:
        f.write('\n\n##### LSTM #####:\n')

    lstm_model.fit(X_train_padded, y_train, batch_size=64, epochs=1, validation_data=(X_valid_padded, y_valid), callbacks=[metrics_callback])

print(f"{logger()} LSTM architecture: ") if verbose else None
print(lstm_model.summary()) if verbose else None

print(config["max_len"]) if verbose else None

if shuffle_validation:
    y_valid = np.random.permutation(y_valid)

# model stacking
model_name= f"{project_name}_Ensemble"
model_path = f"models/{model_name}.keras"

if model_exist_en:
    # Try to load the saved model
    ensemble = load_model(model_path)
    ensemble.compile(optimizer='adam', loss=loss_func, metrics=["accuracy"])
    print(f"\n{logger()} ENSEMBLE Model loaded successfully.")
else:
    print(f"\n{logger()} ENSEMBLE Model not found, creating new.")
    all_models = [cnn_model, lstm_model]

    # model_outputs = [model(model_input) for model in models]
    # ensemble_output = tf.keras.layers.Average(name="ensemble_average")(model_outputs)
    # ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output,name="ensemble")

    print(X_train_padded.shape) if verbose else None

    models = [cnn_model, lstm_model]
    model_input = tf.keras.Input(shape=(X_padded_reshaped.shape[1],X_padded_reshaped.shape[2]), name="ensemble_input")
    #X_train_padded, y_train #X_padded_reshaped


    ensemble = model_builders.create_ensemble(models, model_input, final_dim, final_activation)
    ensemble.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    optimizer="adam", 
                    metrics=["accuracy"])

    sample_size = X_valid_padded.shape[0] # number of samples in train set
    time_steps  = X_valid_padded.shape[1] # number of features in train set
    input_dimension = 1               # each feature is represented by 1 number

    X_valid_padded_reshaped = X_valid_padded.reshape(sample_size,time_steps,input_dimension)
    print(X_valid_padded_reshaped.shape) if verbose else None

    #ensemble.save_weights("ensemble_weights.keras")
    history = ensemble.fit(X_padded_reshaped, y_train, 
                            epochs=max_epoch_ensemble, 
                            verbose=1,
                            #validation_split=0.25)
                            validation_data=(X_valid_padded_reshaped, y_valid))#, 
                            #  callbacks=[ensemble_checkpoint]
                            


    helper.plot_history(history,model_name, "ensemble")
    #helper.save_summary(ensemble, history, "ensemble", model_name)

    ensemble.save(model_path)
    print(f"{logger()} Saving Ensemble")


    with open(summary_path, 'a') as f:
        f.write('\n\n-##### Ensemble #####:\n')

    metrics_callback = model_builders.MetricsCallback(test_data=X_padded_reshaped, y_true=y_train, name=project_name)

    ensemble.fit(X_padded_reshaped, y_train, batch_size=64, epochs=1, 
        #validation_split=0.25,
        validation_data=(X_valid_padded_reshaped, y_valid), 
        callbacks=[metrics_callback])

print(f"{logger()} Ensemble architecture: ") if verbose else None
print(ensemble.summary()) if verbose else None

# predictions validation

if not all([model_exist_en, model_exist_cnn, model_exist_lstm, token_exist, config_exist]):
    print(f"\n{logger()} Starting validation prediction: ") 
 
    predictions = ensemble.predict(X_valid_padded_reshaped)
    predictions_df = pd.DataFrame(predictions)
    predictions_df = process_dataframe(predictions_df, labels, "EN")

    predictions_lstm = lstm_model.predict(X_valid_padded)
    predictions_df_lstm = pd.DataFrame(predictions_lstm)
    predictions_df_lstm = process_dataframe(predictions_df_lstm, labels, "LSTM")

    # predictions_df_lstm = predictions_df_lstm.rename(columns={0: 'pred_lstm'})
    # predictions_df_lstm['pred_bin_lstm'] = (predictions_df_lstm['pred_lstm']).round().astype(int)

    predictions_cnn = cnn_model.predict(X_valid_padded)
    predictions_df_cnn = pd.DataFrame(predictions_cnn)
    predictions_df_cnn = process_dataframe(predictions_df_cnn, labels, "CNN")

    # predictions_df_cnn = predictions_df_cnn.rename(columns={0: 'pred_cnn'})
    # predictions_df_cnn['pred_bin_cnn'] = (predictions_df_cnn['pred_cnn']).round().astype(int)

    X_valid_balanced.reset_index(drop=True, inplace=True)
    predictions_df.reset_index(drop=True, inplace=True)
    predictions_df_lstm.reset_index(drop=True, inplace=True)
    predictions_df_cnn.reset_index(drop=True, inplace=True)

    merged_df = pd.concat([X_valid_balanced, predictions_df,predictions_df_lstm,predictions_df_cnn], axis=1)
    merged_df = reorder_columns(merged_df)

    output_path = f"predictions/{model_name}.validation.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"{logger()} Predictions (validation data) saved to {output_path}")
    print(merged_df) if verbose else None

    output_path = f"predictions/{model_name}.validation.fasta"
    write_to_fasta(merged_df, output_path)
    print(f"{logger()} Predictions (fasta data) saved to {output_path}")


# predictions query data

print(f"\n{logger()} Starting query prediction: ") 
sample_size = X_query_padded.shape[0] # number of samples in testing set
input_dimension = 1               # each feature is represented by 1 number

print(X_query_padded.shape) if verbose else None
X_query_padded_reshaped = X_query_padded.reshape(sample_size,config["max_len"],input_dimension)
print(X_query_padded_reshaped.shape) if verbose else None

predictions = ensemble.predict(X_query_padded_reshaped)
predictions_df = pd.DataFrame(predictions)
predictions_df = process_dataframe(predictions_df, labels, "EN")

predictions_lstm = lstm_model.predict(X_query_padded)
predictions_df_lstm = pd.DataFrame(predictions_lstm)
predictions_df_lstm = process_dataframe(predictions_df_lstm, labels, "LSTM")

predictions_cnn = cnn_model.predict(X_query_padded)
predictions_df_cnn = pd.DataFrame(predictions_cnn)
predictions_df_cnn = process_dataframe(predictions_df_cnn, labels, "CNN")

sequences_dict_query.reset_index(drop=True, inplace=True)
predictions_df.reset_index(drop=True, inplace=True)
predictions_df_lstm.reset_index(drop=True, inplace=True)
predictions_df_cnn.reset_index(drop=True, inplace=True)

merged_df = pd.concat([sequences_dict_query, predictions_df,predictions_df_lstm,predictions_df_cnn], axis=1)
merged_df = reorder_columns(merged_df)

output_path = f"predictions/{query_name}.{model_name}.query.csv"
merged_df.to_csv(output_path, index=False)
print(f"{logger()} Predictions (validation data) saved to {output_path}")

print(merged_df.columns) if verbose else None

output_path = f"predictions/{query_name}.{model_name}.query.fasta"
write_to_fasta(merged_df, output_path)
print(f"{logger()} Predictions (fasta data) saved to {output_path}")
