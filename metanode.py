import os
import argparse
import time
import random
import pickle
import numpy as np 
import pandas as pd 
import subprocess

start_time = time.time()

parser = argparse.ArgumentParser(description="Amplicon Anomaly Detection")
parser.add_argument('-db', dest='true_file', required=False, help="True Amplicon database FASTA file")
parser.add_argument('-query', dest='query_file', required=True, help="Amplicon database FASTA file")
parser.add_argument('-p', dest='project_name', required=True, help="Model name")
parser.add_argument('-2c', dest='two_classes', action='store_true', required=False, help="Switch from multiclass to binary (DEPRECEATED)")
parser.add_argument('-r', dest='recalibrate', action='store_true', required=False, help="Switch to tuner recalibration")
parser.add_argument('-e', dest='epochs', required=False, default=20, help="change epochs from 20 to other value")
parser.add_argument('-v', dest='verbose', action='store_true', required=False, help="Switch to verbose mode")
parser.add_argument('-t', dest='threads', required=False, help="Number of threads")
parser.add_argument('-ot', dest='offtargets', required=False, help="Number of threads")
parser.add_argument('-seed', dest='seed', required=False, help="Seed for randomization")

args = parser.parse_args()
project_name=args.project_name 

# Silence TF/absl startup noise , settings for training consistencies
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   # TF C++ logs
os.environ.setdefault("GLOG_minloglevel", "3")       # absl/glog route
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    # Suppress TF/CUDA INFO+WARNING
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"    # disable TF32 fast-math on Ampere/Ada
os.environ["TF32_OVERRIDE"] = "0"     

# Keep your own logs on stdout only

verbose = True if args.verbose else False
if verbose:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
else: 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['KERAS_BACKEND'] = 'tensorflow'

if not args.seed:
    SEED = int(os.environ.get("SEED", str(int(time.time()) % (2**32 - 1))))
else:
    SEED = int(args.seed)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
# Tame absl once TF is imported
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
absl_logging.use_absl_handler()
try:
    # Avoid the "written to STDERR" preinit warning
    absl_logging._warn_preinit_stderr = False  # pylint: disable=protected-access
except Exception:
    pass

from tensorflow.python.client import device_lib
tf.random.set_seed(SEED)

import keras
#from tensorflow import keras
from keras import layers, models, optimizers
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
#from tensorflow.keras import layers
import keras_tuner as kt

from modules import helper
from modules import model_builders

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# logging on command line and to file
timestamp = time.strftime("%Y%m%d.%H%M%S")
log_dir = "logs"  # <- relative; outside Docker it lands in ./logs, inside Docker in /data/logs
os.makedirs(log_dir, exist_ok=True)
logfile_path = os.path.join(log_dir, f"{project_name}.{timestamp}.log")

def log(msg: str):
    """Prints and appends a timestamped line to the logfile."""
    global project_name, start_time, logfile_path
    elapsed = int(time.time() - start_time)
    hhmmss = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    line = f"[MetAnoDe: {project_name}: {hhmmss}] {msg}"

    # console
    print(line)

    # file (best-effort)
    try:
        with open(logfile_path, "a") as f:
            f.write(line + "\n")
    except Exception:
        # don't crash training because logging failed
        pass

log(f"Random seed: {SEED}")

import contextlib, io, sys
from tensorflow.python.client import device_lib

def safe_list_devices():
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        devices = device_lib.list_local_devices()
    return devices

if verbose:
    log("## Devices available: ")
    for d in safe_list_devices():
        log(str(d))
    log("CUDA version:")
    log(tf.sysconfig.get_build_info().get('cuda_version', 'unknown'))

    
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

if args.threads is not None:
    threads = int(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(threads)

if not args.two_classes:
    balance_4C = True
    loss_func='sparse_categorical_crossentropy'
else:
    balance_4C = False
    loss_func='binary_crossentropy' 

max_epoch = args.epochs
max_epoch_ensemble=max_epoch
max_epoch_tuner=5
learning_rate=0.001
opt = keras.optimizers.Adam(learning_rate=learning_rate) # ensemble only

SPLIT_DIR = "splits"
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

#labels = {0: 'positive', 1: 'substitution', 2: 'indels', 3: 'chimera', 4: 'mito', 5 : 'chloro'} # TODO fix in config ###### ---> 16S 
#labels = {0: 'positive', 1: 'substitution', 2: 'indels', 3: 'chimera', 4: 'fungi'} # TODO fix in config ###### ---> ITS2 


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

def _run_git(args, cwd):
    return subprocess.check_output(["git"] + args, cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()

def _looks_like_git_repo(path):
    return os.path.isdir(os.path.join(path, ".git"))

def get_git_info_simple():
    """
    Probe git metadata from either the current directory or /opt/MetAnoDe.
    Falls back to VERSION-ish files or env vars if not a git repo.
    Returns: (count, short, full, source_path_or_reason)
    """
    candidates = [
        os.getcwd(),
        "/opt/MetAnoDe"
    ]

    for root in candidates:
        try:
            if _looks_like_git_repo(root):
                count = _run_git(["rev-list", "--count", "HEAD"], cwd=root)
                short = _run_git(["rev-parse", "--short", "HEAD"], cwd=root)
                full  = _run_git(["rev-parse", "HEAD"], cwd=root)
                return count, short, full, root
        except Exception:
            pass

    # Fallbacks (non-git image or copied tree)
    for root in candidates:
        for fname in ("VERSION", "version.txt", "build.txt", ".build"):
            p = os.path.join(root, fname)
            if os.path.isfile(p):
                try:
                    v = open(p).read().strip()
                    return "n/a", v[:7] or "unknown", v or "unknown", f"{root}/{fname}"
                except Exception:
                    pass

    # Env var fallback (can be set at build/run time)
    env_commit = os.environ.get("GIT_COMMIT") or os.environ.get("METANODE_BUILD")
    if env_commit:
        return "n/a", env_commit[:7], env_commit, "env"

    return "unknown", "unknown", "unknown", "no git / no version file / no env"

# --- use it ---
count, short, full, src = get_git_info_simple()
log(f"Git: {short} (#{count}) {full} [{src}]")

def generate_balanced_dataset(args, balance_4C, labels, verbose=False):
    sequences_dict_true = helper.read_fasta(args.true_file, 0, 0)
    log(sequences_dict_true) if verbose else None

    tr_len = len(sequences_dict_true)
    max_rows = int(tr_len * 2)

    if not balance_4C:
        max_rows = int(max_rows / 3)

    log(f"\n"); log(f"Received {len(sequences_dict_true)} original sequences (Class 0)")

    sequences_dict_f1_balanced = helper.create_artificial_errorate(sequences_dict_true, max_rows, "subst")
    log(f"Created {len(sequences_dict_f1_balanced)} artifical high substitution rate sequences (Class 1)")

    sequences_dict_f2_balanced = helper.create_artificial_errorate(sequences_dict_true, max_rows, "indel")
    log(f"Created {len(sequences_dict_f2_balanced)} artifical high indel rate sequences (Class 2)")

    sequences_dict_f3_balanced = helper.create_artificial_chimera(sequences_dict_true, max_rows)
    log(f"Created {len(sequences_dict_f3_balanced)} artifical chimeric sequences (Class 3)")

    sequences_dict_true_lowsubst = helper.create_artificial_errorate(sequences_dict_true, int(tr_len / 2), "lowsubst")
    sequences_dict_true_lowindel = helper.create_artificial_errorate(sequences_dict_true, int(tr_len / 2), "lowindel")
    sequences_dict_true_balanced = pd.concat(
        [sequences_dict_true, sequences_dict_true_lowsubst, sequences_dict_true_lowindel],
        ignore_index=True
    )
    log(f"Created {len(sequences_dict_true_lowsubst)} + {len(sequences_dict_true_lowindel)} artifical low error sequences (Class 0)")

    if args.offtargets is not None:
        split_list = args.offtargets.split(',')
        i = 4
        sequences_dict_offtarget_all = pd.DataFrame()

        for offtarget in split_list:
            sequences_dict_offtarget = helper.read_fasta(offtarget, 0, i)
            ot_len = len(sequences_dict_offtarget)

            if ot_len > max_rows:
                sequences_dict_offtarget = helper.sample_dataframe(sequences_dict_offtarget, max_rows)
                type_ot = "(downsampled)"
            else:
                missing_rows = max_rows - ot_len
                sequences_dict_ot_lowsubst = helper.create_artificial_errorate(sequences_dict_offtarget, int(missing_rows / 2), "lowsubst")
                sequences_dict_ot_lowindel = helper.create_artificial_errorate(sequences_dict_offtarget, int(missing_rows / 2), "lowindel")
                sequences_dict_offtarget = pd.concat(
                    [sequences_dict_offtarget, sequences_dict_ot_lowsubst, sequences_dict_ot_lowindel],
                    ignore_index=True
                )
                type_ot = "(upsampled)"

            len_ot = len(sequences_dict_offtarget)
            sequences_dict_offtarget["Target4D"] = i
            sequences_dict_offtarget["Target"] = 1
            labels[i] = f"OffTarget-{i}"
            log(f"Added {len_ot} {type_ot} Off-target Class {i} ({offtarget})")
            i += 1

            sequences_dict_offtarget_all = pd.concat([sequences_dict_offtarget_all, sequences_dict_offtarget], ignore_index=True)
            del sequences_dict_offtarget
    else:
        sequences_dict_offtarget_all = helper.sample_dataframe(sequences_dict_true_balanced, 0)

    X_train_balanced = pd.concat(
        [
            sequences_dict_true_balanced,
            sequences_dict_f1_balanced,
            sequences_dict_f2_balanced,
            sequences_dict_f3_balanced,
            sequences_dict_offtarget_all
        ],
        ignore_index=True
    )

    log(f"Balanced dataset created")
    return X_train_balanced, labels

# load in reference data
if not all([model_exist_en, model_exist_cnn, model_exist_lstm, token_exist, config_exist]):
    log(f"\n");log(f"Not all required models and configs are available, regenerating necessary ones")
    log(f"This might take a while, depending on whether models are missing or need to be tuned...")

    SPLIT_DIR = "splits"
    config = {}
    if split_files_exist(project_name, SPLIT_DIR):
        X_train_balanced, X_valid_balanced, split_meta = helper.load_split(project_name, SPLIT_DIR)
        log(f"Loaded existing train/validation split for project '{project_name}' from {SPLIT_DIR}")
        log(f"Split metadata: {split_meta}")

        labels = split_meta.get("labels", labels)
        config["labels"] = labels
        config["max_len"] = split_meta["max_len"]
        config["output_dim"] = split_meta["output_dim"]
    else:
        X_all_balanced, labels = generate_balanced_dataset(args, balance_4C, labels, verbose=verbose)

        config["max_len"] = max(len(seq) for seq in X_all_balanced["sequences"])
        config["output_dim"] = X_all_balanced["Target4D"].nunique()
        config["labels"] = labels

        X_all_balanced = shuffle(X_all_balanced, random_state=SEED).reset_index(drop=True)
        log(f"Shuffled data with seed {SEED}")

        y_for_split = X_all_balanced["Target4D"] if balance_4C else X_all_balanced["Target"]

        X_train_balanced, X_valid_balanced = train_test_split(
            X_all_balanced,
            test_size=0.25,
            stratify=y_for_split,
            random_state=SEED,
            shuffle=True,
        )

        log("Split data to train and validation (stratified)")
        all_split_df = pd.concat([X_train_balanced, X_valid_balanced], ignore_index=True)

        split_meta = {
            "project_name": project_name,
            "seed": SEED,
            "test_size": 0.25,
            "balance_4C": balance_4C,
            "loss_func": loss_func,
            "query_file": args.query_file,
            "true_file": args.true_file,
            "offtargets": args.offtargets,
            "n_train": int(len(X_train_balanced)),
            "n_valid": int(len(X_valid_balanced)),
            "labels": labels,
            "max_len": int(max(len(seq) for seq in all_split_df["sequences"])),
            "output_dim": int(all_split_df["Target4D"].nunique()),
            "git_commit_short": short,
            "git_commit_full": full,
        }
        del all_split_df

        helper.save_split(X_train_balanced, X_valid_balanced, split_meta, project_name, SPLIT_DIR)
        log(f"Saved new split for project '{project_name}'")
        log(f"Split metadata: {split_meta}")

    log(X_train_balanced) if verbose else None

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

        log(X_train_list[1]) if verbose else None

    # remove temporary data
    del X_train_balanced
    #del X_valid_balanced
    del X_train_final
    del X_valid_final
    log(f"Removed temporary data from memory")
    
    # Fit tokenizer on training data
    log(f"Encoding data")

    if not token_exist:
        if kmer:
            encoder = Tokenizer(char_level=False) 
        else:
            encoder = Tokenizer(char_level=True)

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
    log(encoded_characters) if verbose else None

    # Pad sequences
    log(f"Padding data")

    X_train_encoded =  keras.preprocessing.sequence.pad_sequences(X_train_encoded, maxlen=config["max_len"], padding='post')
    X_valid_encoded =  keras.preprocessing.sequence.pad_sequences(X_valid_encoded, maxlen=config["max_len"], padding='post')  

    # convert to numpy arrays
    log(f"Converting data")
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

    # --- Save a small validation cache for future loaded-model runs ---
    val_cache_path = f"models/{project_name}_val_cache.npz"
    try:
        # keep it compact (optionally subsample)
        np.savez_compressed(
            val_cache_path,
            X_valid_padded=X_valid_padded.astype("int32"),     # tokens for LSTM
            y_valid=y_valid.astype("int64")
        )
        log(f"Saved validation cache to {val_cache_path}")
    except Exception as e:
        log(f"Warning: could not save validation cache ({e})")


    #### for CNN: 
    sample_size = X_train_padded.shape[0] # number of samples in train set
    time_steps  = X_train_padded.shape[1] # number of features in train set
    input_dimension = 1               # each feature is represented by 1 number

    X_padded_reshaped = X_train_padded.reshape(sample_size,time_steps,input_dimension)
    log(X_padded_reshaped.shape) if verbose else None

    # remove temporary data
    del X_train_list
    del X_valid_list
    del X_train_encoded
    del X_valid_encoded
    #del X_train_padded
    #del X_valid_padded
    log(f"Removed temporary data from memory")

with open(token_path, 'rb') as token:
    encoder = pickle.load(token)    

with open(config_path, 'rb') as file:
    # Deserialize and retrieve the variable from the file
    config = pickle.load(file)


final_dim = config["output_dim"]
log(f"Modeling {final_dim} Classes")
log(f"Class labels: {labels}")
log(f"Epochs: {max_epoch}")


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
    log(f"\n");log(f"CNN Model loaded successfully.")
else:
    log(f"\n");log(f"CNN Model file not found. Creating a new model...")
    cnn_hyper_model = model_builders.CNNHyperModel(n_timesteps=X_padded_reshaped.shape[1],loss_func=loss_func, final_dim=final_dim, final_activation=final_activation,n_features  = X_padded_reshaped.shape[2])
    metrics_callback = model_builders.MetricsCallback(test_data=X_valid_padded, y_true=y_valid, name=project_name)

    cnn_tuner = kt.Hyperband(cnn_hyper_model,
                        objective='val_accuracy',
                        max_epochs=max_epoch_tuner,
                        factor=3,
                        directory='tuner',
                        project_name=model_name)

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        
    cnn_tuner.search(X_train_padded, y_train, epochs=max_epoch_tuner, batch_size=64, validation_data=(X_valid_padded, y_valid), callbacks=[stop_early])
    best_hps=cnn_tuner.get_best_hyperparameters(num_trials=1)[0]
    log(f' CNN Hyperparameter Tuning completed\n\n')
    log(f"{best_hps.values}")

    cnn_model = cnn_tuner.hypermodel.build(best_hps)
    cnn_history = cnn_model.fit(X_train_padded, y_train, batch_size=64, epochs=max_epoch, validation_data=(X_valid_padded, y_valid))
    log(f' CNN Model completed\n\n')

    val_acc_per_epoch = cnn_history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    log(f' Best epoch: %d' % (best_epoch,))

    cnn_model.save(model_path)  # Save the model to a HDF5 file
    log(f"CNN Model saved successfully.")

    helper.plot_history(cnn_history,model_name, best_hps.values)
    helper.save_summary(cnn_model, cnn_history, best_hps, model_name)

    with open(summary_path, 'a') as f:
        f.write('##### CNN #####:\n')

    cnn_model.fit(X_train_padded, y_train, batch_size=64, epochs=1, validation_data=(X_valid_padded, y_valid), callbacks=[metrics_callback])

log(f"CNN architecture: ") if verbose else None
log(cnn_model.summary()) if verbose else None

model_name= f"{project_name}_LSTM"
model_path = f"models/{model_name}.keras"

if model_exist_lstm:
    # Try to load the saved model
    lstm_model = load_model(model_path)
    lstm_model.compile(optimizer='adam', loss=loss_func)
    log(f"\n");log(f"LSTM Model loaded successfully.")
else:
    log(f"\n");log(f"LSTM Model file not found. Creating a new model...")
    lstm_hyper_model = model_builders.LSTMHyperModel(encoder=encoder,loss_func=loss_func, final_dim=final_dim, final_activation=final_activation)
    metrics_callback = model_builders.MetricsCallback(test_data=X_valid_padded, y_true=y_valid, name=project_name)

    lstm_tuner = kt.Hyperband(lstm_hyper_model,
                        objective='val_accuracy',
                        max_epochs=max_epoch_tuner,
                        factor=3,
                        directory='tuner',
                        project_name=model_name)

    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    lstm_tuner.search(X_train_padded, y_train, batch_size=64, epochs=max_epoch_tuner, validation_data=(X_valid_padded, y_valid), callbacks=[stop_early])
    best_hps=lstm_tuner.get_best_hyperparameters(num_trials=1)[0]
    log(f' LSTM Hyperparameter Tuning completed\n\n')
    log(f"{best_hps.values}")

    lstm_model = lstm_tuner.hypermodel.build(best_hps)
    lstm_history = lstm_model.fit(X_train_padded, y_train, batch_size=64, epochs=max_epoch, validation_data=(X_valid_padded, y_valid))

    val_acc_per_epoch = lstm_history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    log(f' Best epoch: %d' % (best_epoch,))
    log(f' LSTM Model completed\n\n')

    lstm_model.save(model_path)  # Save the model to a HDF5 file
    log(f"Model saved successfully.")

    helper.plot_history(lstm_history,model_name, best_hps.values)
    helper.save_summary(lstm_model, lstm_history, best_hps, model_name)

    with open(summary_path, 'a') as f:
        f.write('\n\n##### LSTM #####:\n')

    lstm_model.fit(X_train_padded, y_train, batch_size=64, epochs=1, validation_data=(X_valid_padded, y_valid), callbacks=[metrics_callback])

log(f"LSTM architecture: ") if verbose else None
log(lstm_model.summary()) if verbose else None

log(config["max_len"]) if verbose else None

if shuffle_validation:
    y_valid = np.random.permutation(y_valid)

# model stacking
model_name= f"{project_name}_Ensemble"
model_path = f"models/{model_name}.keras"

if model_exist_en:
    # Try to load the saved model
    ensemble = load_model(model_path)
    ensemble.compile(optimizer='adam', loss=loss_func, metrics=["accuracy"])
    log(f"\n");log(f"ENSEMBLE Model loaded successfully.")
else:
    log(f"\n");log(f"ENSEMBLE Model not found, creating new.")
    all_models = [cnn_model, lstm_model]

    # model_outputs = [model(model_input) for model in models]
    # ensemble_output = tf.keras.layers.Average(name="ensemble_average")(model_outputs)
    # ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output,name="ensemble")

    log(X_train_padded.shape) if verbose else None

    # --- Build proper multi-input ensemble (LSTM: (T,), CNN: (T,1)) ---
    n_timesteps = X_train_padded.shape[1]  # T

    inp_lstm = keras.Input(shape=(n_timesteps,), dtype="int32",   name="ensemble_lstm_in")
    inp_cnn  = keras.Input(shape=(n_timesteps, 1), dtype="float32", name="ensemble_cnn_in")

    models_for_ensemble = [cnn_model, lstm_model]  # order doesn't matter (rank-based routing)

    ensemble = model_builders.create_ensemble(
        models=models_for_ensemble,
        inputs=[inp_lstm, inp_cnn],
        final_dim=final_dim,
        final_activation=final_activation
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f"models/{project_name}_Ensemble.best.keras",
            monitor="val_accuracy", mode="max", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]

    ensemble.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    # Ensure validation reshape exists
    X_valid_padded_reshaped = X_valid_padded.reshape(
        X_valid_padded.shape[0], X_valid_padded.shape[1], 1
    )

    # Train with TWO inputs: [tokens, cnn_features]
    history = ensemble.fit(
        x=[X_train_padded, X_padded_reshaped],
        y=y_train,
        epochs=max_epoch_ensemble,
        batch_size=64,
        verbose=1,
        validation_data=([X_valid_padded, X_valid_padded_reshaped], y_valid),
        callbacks=callbacks,
    )

    # Plot training history
    helper.plot_history(history, model_name, "ensemble")
    
    # Write summary section
    ensemble.save(model_path)
    log(f"Saving Ensemble")

    # Write summary section
    with open(summary_path, 'a') as f:
        f.write('\n\n-##### Ensemble #####:\n')

    metrics_callback = model_builders.MetricsCallback(
        test_data=[X_valid_padded, X_valid_padded_reshaped],  # LSTM tokens, CNN 3D
        y_true=y_valid,
        name=project_name
    )

    ensemble.fit(
        [X_train_padded, X_padded_reshaped],  # two inputs
        y_train,
        batch_size=64,
        epochs=1,
        validation_data=([X_valid_padded, X_valid_padded_reshaped], y_valid),
        callbacks=[metrics_callback]
    )

log(f"Ensemble architecture: ") if verbose else None
log(ensemble.summary()) if verbose else None

# Try to load cached validation set (works when models are loaded without retraining)
val_cache_path = f"models/{project_name}_val_cache.npz"
VAL = None
if os.path.isfile(val_cache_path):
    try:
        _cache = np.load(val_cache_path, allow_pickle=False)
        X_valid_lstm = _cache["X_valid_padded"].astype("int32")      # (N, T)
        y_valid_cached = _cache["y_valid"]
        X_valid_cnn  = np.expand_dims(X_valid_lstm, axis=-1).astype("float32")  # (N, T, 1)
        VAL = (X_valid_lstm, X_valid_cnn, y_valid_cached)
        log(f"Loaded validation cache from {val_cache_path}: {X_valid_lstm.shape}")
    except Exception as e:
        log(f"Warning: could not load validation cache ({e})")

# ---- Sanity-check branch + ensemble metrics (only if we have VAL) ----
if VAL is not None:
    log("=== Sanity-check: branch and ensemble validation metrics ===")
    X_valid_lstm, X_valid_cnn, y_valid_cached = VAL

    # 1) LSTM solo
    try:
        lstm_loss, lstm_acc = lstm_model.evaluate(X_valid_lstm, y_valid_cached, verbose=0)
        log(f"LSTM    -> val_loss={lstm_loss:.4f}, val_acc={lstm_acc:.4f}")
    except Exception as e:
        log(f"LSTM evaluate failed: {e}")

    # 2) CNN solo
    try:
        cnn_loss, cnn_acc = cnn_model.evaluate(X_valid_cnn, y_valid_cached, verbose=0)
        log(f"CNN     -> val_loss={cnn_loss:.4f}, val_acc={cnn_acc:.4f}")
    except Exception as e:
        log(f"CNN evaluate failed: {e}")

    # 3) Ensemble (two inputs)
    try:
        ens_loss, ens_acc = ensemble.evaluate([X_valid_lstm, X_valid_cnn], y_valid_cached, verbose=0)
        log(f"Ensemble-> val_loss={ens_loss:.4f}, val_acc={ens_acc:.4f}")
        best_single_acc = max(locals().get("lstm_acc", 0.0), locals().get("cnn_acc", 0.0))
        log(f"Ensemble gain over best single: {ens_acc - best_single_acc:+.4f}")
    except Exception as e:
        log(f"Ensemble evaluate failed: {e}")
else:
    log("No validation cache found; skipping sanity-check metrics.")

# predictions validation

if not all([model_exist_en, model_exist_cnn, model_exist_lstm, token_exist, config_exist]):
    log(f"\n");log(f"Starting validation prediction: ") 
 
    predictions = ensemble.predict([
        X_valid_padded,
        X_valid_padded.reshape(X_valid_padded.shape[0], X_valid_padded.shape[1], 1)
    ])

    predictions_df = pd.DataFrame(predictions)
    predictions_df = process_dataframe(predictions_df, labels, "EN")

    predictions_lstm = lstm_model.predict(X_valid_padded)
    predictions_df_lstm = pd.DataFrame(predictions_lstm)
    predictions_df_lstm = process_dataframe(predictions_df_lstm, labels, "LSTM")

    # predictions_df_lstm = predictions_df_lstm.rename(columns={0: 'pred_lstm'})
    # predictions_df_lstm['pred_bin_lstm'] = (predictions_df_lstm['pred_lstm']).round().astype(int)

    predictions_cnn = cnn_model.predict(X_valid_padded_reshaped)
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
    log(f"Predictions (validation data) saved to {output_path}")
    log(merged_df) if verbose else None

    output_path = f"predictions/{model_name}.validation.fasta"
    write_to_fasta(merged_df, output_path)
    log(f"Predictions (fasta data) saved to {output_path}")


# predictions query data

log(f"\n");log(f"Starting query prediction: ") 
sample_size = X_query_padded.shape[0] # number of samples in testing set
input_dimension = 1               # each feature is represented by 1 number

log(X_query_padded.shape) if verbose else None
# X_query_padded_reshaped = X_query_padded.reshape(sample_size,config["max_len"],input_dimension)
X_query_padded_reshaped = X_query_padded.reshape(
    X_query_padded.shape[0], config["max_len"], 1
)

log(X_query_padded_reshaped.shape) if verbose else None

predictions = ensemble.predict([X_query_padded, X_query_padded_reshaped])
predictions_df = pd.DataFrame(predictions)
predictions_df = process_dataframe(predictions_df, labels, "EN")

predictions_lstm = lstm_model.predict(X_query_padded)
predictions_df_lstm = pd.DataFrame(predictions_lstm)
predictions_df_lstm = process_dataframe(predictions_df_lstm, labels, "LSTM")

predictions_cnn = cnn_model.predict(X_query_padded_reshaped)
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
log(f"Predictions (validation data) saved to {output_path}")

log(merged_df.columns) if verbose else None

output_path = f"predictions/{query_name}.{model_name}.query.fasta"
write_to_fasta(merged_df, output_path)
log(f"Predictions (fasta data) saved to {output_path}")
