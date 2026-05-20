import os
import sys
import argparse
import time
import random
import pickle
import numpy as np 
import pandas as pd 
import subprocess
import json

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
parser.add_argument(
    '--target_len',
    dest='target_len',
    required=False,
    type=int,
    default=None,
    help="Expected amplicon length. If not set, estimated from data."
)
parser.add_argument(
    "--validate_models",
    action="store_true",
    help="Run validation sanity checks on pretrained models"
)

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

def dict_to_hparams(hp_dict):
    hp = kt.HyperParameters()
    for key, value in hp_dict.items():
        hp.Fixed(key, value)
    return hp


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

def process_dataframe(
    df: pd.DataFrame,
    labels: dict,
    model: str
) -> pd.DataFrame:

    renamed_columns = {}

    for col in df.columns:

        label = labels.get(col, f"unknown_{col}")

        renamed_columns[col] = f"{model}_{label}"

    df = df.rename(columns=renamed_columns)

    label_columns = list(renamed_columns.values())

    df[f"{model}_class"] = (
        df[label_columns]
        .idxmax(axis=1)
        .str.replace(f"{model}_", "", regex=False)
    )

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
max_epoch_ensemble=5
max_epoch_tuner=5
learning_rate=0.0003
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

hp_override_path = f"models/{project_name}.hyperparameters.json"
hp_override_cnn_path = f"models/{project_name}_CNN.hyperparameters.json"
hp_override_lstm_path = f"models/{project_name}_LSTM.hyperparameters.json"

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

def resolve_target_len(df, args_target_len, log_fn=None):
    """
    Resolve target length:
    - use args if provided
    - otherwise estimate from data using the median
    """
    lengths = df["sequences"].str.len()
    empirical_median = int(lengths.median())
    empirical_mean = float(lengths.mean())

    if args_target_len is not None:
        if log_fn:
            log_fn(
                f"Using user-defined target_len={args_target_len} | "
                f"empirical_median={empirical_median}, empirical_mean={empirical_mean:.1f}, "
                f"min={int(lengths.min())}, max={int(lengths.max())}"
            )
        return int(args_target_len)

    if log_fn:
        log_fn(
            f"Estimated target_len from data using median: {empirical_median} | "
            f"mean={empirical_mean:.1f}, min={int(lengths.min())}, max={int(lengths.max())}"
        )

    return empirical_median

def length_stats_str(df, seq_col="sequences"):
    """
    Return formatted sequence length stats for a dataframe.
    """
    lengths = df[seq_col].str.len()
    return (
        f"min={int(lengths.min())}, "
        f"max={int(lengths.max())}, "
        f"mean={lengths.mean():.1f}, "
        f"median={float(lengths.median()):.1f}"
    )

def filter_by_length(df, target_len, lower_factor=0.5, upper_factor=1.5):
    lower = int(target_len * lower_factor)
    upper = int(target_len * upper_factor)

    lengths = df["sequences"].str.len()

    mask = (lengths >= lower) & (lengths <= upper)

    df_filtered = df[mask].reset_index(drop=True)

    stats = {
        "before_n": len(df),
        "after_n": len(df_filtered),
        "removed_n": len(df) - len(df_filtered),
        "lower": lower,
        "upper": upper
    }

    return df_filtered, stats

def generate_balanced_dataset(args, balance_4C, labels, verbose=False):
    sequences_dict_true = helper.read_fasta(args.true_file, 0, 0)
    log(sequences_dict_true) if verbose else None

    # --- resolve target length ---
    target_len = resolve_target_len(
        sequences_dict_true,
        args.target_len,
        log_fn=log
    )

    # --- length filtering ---
    sequences_dict_true, length_filter_stats = filter_by_length(
        sequences_dict_true,
        target_len=target_len,
        lower_factor=0.75,
        upper_factor=1.25
    )

    log(
        f"Length filter applied: kept {length_filter_stats['after_n']}/"
        f"{length_filter_stats['before_n']} "
        f"(removed {length_filter_stats['removed_n']}) | "
        f"range [{length_filter_stats['lower']}, {length_filter_stats['upper']}]"
    )

    log(f"\n")
    log(
        f"Received {len(sequences_dict_true)} original sequences (Class 0) | "
        f"{length_stats_str(sequences_dict_true)}"
    )

    tr_len = len(sequences_dict_true)
    max_rows = int(tr_len * 2)

    if not balance_4C:
        max_rows = int(max_rows / 3)

    sequences_dict_f1_balanced = helper.create_artificial_errorate(sequences_dict_true, max_rows, "subst")
    log(f"Created {len(sequences_dict_f1_balanced)} artifical high substitution rate sequences (Class 1) | {length_stats_str(sequences_dict_f1_balanced)}")

    sequences_dict_f2_balanced = helper.create_artificial_errorate(sequences_dict_true, max_rows, "indel")
    log(f"Created {len(sequences_dict_f2_balanced)} artifical high indel rate sequences (Class 2) | {length_stats_str(sequences_dict_f2_balanced)}")

    sequences_dict_f3_balanced = helper.create_artificial_chimera(
        sequences_dict_true,
        max_rows,
        target_len=target_len,
        log_fn=log
    )
    log(f"Created {len(sequences_dict_f3_balanced)} artifical chimeric sequences (Class 3) | {length_stats_str(sequences_dict_f3_balanced)}")

    sequences_dict_true_lowsubst = helper.create_artificial_errorate(sequences_dict_true, int(tr_len / 2), "lowsubst")
    log(f"Created {len(sequences_dict_true_lowsubst)} artifical low substitution sequences (Class 0) | {length_stats_str(sequences_dict_true_lowsubst)}")

    sequences_dict_true_lowindel = helper.create_artificial_errorate(sequences_dict_true, int(tr_len / 2), "lowindel")
    log(f"Created {len(sequences_dict_true_lowindel)} artifical low indel sequences (Class 0) | {length_stats_str(sequences_dict_true_lowindel)}")

    sequences_dict_true_balanced = pd.concat(
        [sequences_dict_true, sequences_dict_true_lowsubst, sequences_dict_true_lowindel],
        ignore_index=True
    )
    log(
        f"Combined Class 0 true + low-error sequences: {len(sequences_dict_true_balanced)} | "
        f"{length_stats_str(sequences_dict_true_balanced)}"
    )

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
                sequences_dict_ot_lowsubst = helper.create_artificial_errorate(
                    sequences_dict_offtarget, int(missing_rows / 2), "lowsubst"
                )
                sequences_dict_ot_lowindel = helper.create_artificial_errorate(
                    sequences_dict_offtarget, int(missing_rows / 2), "lowindel"
                )
                sequences_dict_offtarget = pd.concat(
                    [sequences_dict_offtarget, sequences_dict_ot_lowsubst, sequences_dict_ot_lowindel],
                    ignore_index=True
                )
                type_ot = "(upsampled)"

            len_ot = len(sequences_dict_offtarget)
            sequences_dict_offtarget["Target4D"] = i
            sequences_dict_offtarget["Target"] = 1
            labels[i] = f"OffTarget-{i}"
            log(
                f"Added {len_ot} {type_ot} Off-target Class {i} ({offtarget}) | "
                f"{length_stats_str(sequences_dict_offtarget)}"
            )
            i += 1

            sequences_dict_offtarget_all = pd.concat(
                [sequences_dict_offtarget_all, sequences_dict_offtarget],
                ignore_index=True
            )
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

    log(f"Balanced dataset created | total={len(X_train_balanced)} | {length_stats_str(X_train_balanced)}")
    return X_train_balanced, labels

# ------------------------------------------------------------------
# Expected pretrained files
# ------------------------------------------------------------------

required_files = [
    f"models/{args.project_name}_CNN.keras",
    f"models/{args.project_name}_LSTM.keras",
    f"models/{args.project_name}_Ensemble.keras",
    f"models/{args.project_name}_Ensemble.config",
    f"models/{args.project_name}_Ensemble.token",
]

missing_files = [f for f in required_files if not os.path.exists(f)]

# ------------------------------------------------------------------
# Decide whether retraining is necessary
# ------------------------------------------------------------------

if missing_files:
    log("")
    log(f"Missing required model artifacts:")
    for mf in missing_files:
        print(f"  - {mf}")

    # --------------------------------------------------------------
    # No training database supplied -> cannot regenerate
    # --------------------------------------------------------------

    if args.true_file is None:
        log("")
        log("ERROR:")
        log("Pretrained model appears incomplete, but no training")
        log("database was supplied via -db.")
        log("")
        log("Either:")
        log("  1) restore/copy the missing pretrained files")
        log("  2) provide -db and optionally -ot to retrain")
        log("")
        sys.exit(1)

    # --------------------------------------------------------------
    # Retraining mode
    # --------------------------------------------------------------
    log("")
    log("Not all required models/configs available.")
    log("Regenerating missing components...")
    log("")
    regenerate_models = True

else:
    regenerate_models = False
    log("")
    log(f"Using pretrained model: {args.project_name}")
    metadata = helper.load_model_metadata(project_name)
    labels = metadata["labels"]
    log(f"Classes: {labels}")


# load in reference data
#if "true" == "true": #not all([model_exist_en, model_exist_cnn, model_exist_lstm, token_exist, config_exist]):
if regenerate_models:
    log(f"This might take a while, depending on whether models are missing or need to be tuned...")

    labels = helper.build_labels_from_args(args)

    helper.save_model_metadata(
        project_name=project_name,
        labels=labels,
        extra_metadata={
            "offtargets": args.offtargets.split(",") if args.offtargets else []
        }
    )

    SPLIT_DIR = "splits"
    config = {}
    if helper.split_files_exist(project_name, SPLIT_DIR):
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
    del X_train_list
    del X_valid_list
    log("Removed raw sequence lists from memory")

    # Inspect the encoding. Add padding token to the encoding
    word_index = encoder.word_index
    encoded_characters = pd.DataFrame(list(word_index.items()), columns=['Character', 'Encoding'])
    encoded_characters.loc[len(encoded_characters)] = ['<PAD>', 0]
    log(encoded_characters) if verbose else None
    del word_index
    del encoded_characters

    # Pad sequences
    log(f"Padding data")

    X_train_encoded = keras.preprocessing.sequence.pad_sequences(
        X_train_encoded,
        maxlen=config["max_len"],
        padding='post',
        truncating='post'
    )

    X_valid_encoded = keras.preprocessing.sequence.pad_sequences(
        X_valid_encoded,
        maxlen=config["max_len"],
        padding='post',
        truncating='post'
    )

    # convert to numpy arrays
    log(f"Converting data")
    X_train_padded = np.asarray(X_train_encoded, dtype="int32")
    X_valid_padded = np.asarray(X_valid_encoded, dtype="int32")
    del X_train_encoded
    del X_valid_encoded
    log("Removed intermediate encoded sequences from memory")

    y_train = np.asarray(y_train, dtype="int32")
    y_valid = np.asarray(y_valid, dtype="int32")

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
    time_steps = X_train_padded.shape[1]
    input_dimension = 1
    log(f"CNN base input kept as padded token matrix: {X_train_padded.shape}") if verbose else None

    # remove temporary data
    #del X_train_list
    # del X_valid_list
    #del X_train_encoded
    #del X_valid_encoded
    #del X_train_padded
    #del X_valid_padded
    #log(f"Removed temporary data from memory")

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
X_query_encoded = keras.preprocessing.sequence.pad_sequences(
    X_query_encoded,
    maxlen=config["max_len"],
    padding='post',
    truncating='post'
)
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
    cnn_hyper_model = model_builders.CNNHyperModel(
        n_timesteps=X_train_padded.shape[1],
        loss_func=loss_func,
        final_dim=final_dim,
        final_activation=final_activation,
        n_features=1
    )
    metrics_callback = model_builders.MetricsCallback(
        test_data=X_valid_padded,
        y_true=y_valid,
        name=project_name
    )

    # try model-specific override first, then shared file
    cnn_hp_override = helper.load_hp_override(hp_override_cnn_path, "CNN")
    if cnn_hp_override is None:
        cnn_hp_override = helper.load_hp_override(hp_override_path, "CNN")

    if cnn_hp_override is not None:
        best_hps = dict_to_hparams(cnn_hp_override)
        log("Skipping CNN tuner because static hyperparameter override is present.")
    else:
        cnn_tuner = kt.Hyperband(
            cnn_hyper_model,
            objective='val_accuracy',
            max_epochs=max_epoch_tuner,
            factor=3,
            directory='tuner',
            project_name=model_name
        )

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        cnn_tuner.search(
            X_train_padded,
            y_train,
            epochs=max_epoch_tuner,
            batch_size=64,    
            shuffle=True,
            validation_data=(X_valid_padded, y_valid),
            callbacks=[stop_early]
        )
        best_hps = cnn_tuner.get_best_hyperparameters(num_trials=1)[0]
        log('CNN Hyperparameter Tuning completed\n')
        log(f"{best_hps.values}")

    cnn_model = cnn_hyper_model.build(best_hps)
    cnn_history = cnn_model.fit(
        X_train_padded,
        y_train,
        shuffle=True,
        batch_size=64,
        epochs=max_epoch,
        validation_data=(X_valid_padded, y_valid)
    )
    log('CNN Model completed\n')

    val_acc_per_epoch = cnn_history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    log(f'Best epoch: {best_epoch}')

    cnn_model.save(model_path)
    log("CNN Model saved successfully.")

    helper.plot_history(cnn_history, model_name, best_hps.values)
    helper.save_summary(cnn_model, cnn_history, best_hps, model_name)

    with open(summary_path, 'a') as f:
        f.write('##### CNN #####:\n')

    cnn_model.fit(
        X_train_padded,
        y_train,
        batch_size=64,
        epochs=1,
        shuffle=True,
        validation_data=(X_valid_padded, y_valid),
        callbacks=[metrics_callback]
    )
    
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
    lstm_hyper_model = model_builders.LSTMHyperModel(
        encoder=encoder,
        loss_func=loss_func,
        final_dim=final_dim,
        final_activation=final_activation
    )
    metrics_callback = model_builders.MetricsCallback(
        test_data=X_valid_padded,
        y_true=y_valid,
        name=project_name
    )

    # try model-specific override first, then shared file
    lstm_hp_override = helper.load_hp_override(hp_override_lstm_path, "LSTM")
    if lstm_hp_override is None:
        lstm_hp_override = helper.load_hp_override(hp_override_path, "LSTM")

    if lstm_hp_override is not None:
        best_hps = dict_to_hparams(lstm_hp_override)
        log("Skipping LSTM tuner because static hyperparameter override is present.")
    else:
        lstm_tuner = kt.Hyperband(
            lstm_hyper_model,
            objective='val_accuracy',
            max_epochs=max_epoch_tuner,
            factor=3,
            directory='tuner',
            project_name=model_name
        )

        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        lstm_tuner.search(
            X_train_padded,
            y_train,
            batch_size=64,
            shuffle=True,
            epochs=max_epoch_tuner,
            validation_data=(X_valid_padded, y_valid),
            callbacks=[stop_early]
        )
        best_hps = lstm_tuner.get_best_hyperparameters(num_trials=1)[0]
        log('LSTM Hyperparameter Tuning completed\n')
        log(f"{best_hps.values}")

    lstm_model = lstm_hyper_model.build(best_hps)
    lstm_history = lstm_model.fit(
        X_train_padded,
        y_train,
        batch_size=64,
        shuffle=True,
        epochs=max_epoch,
        validation_data=(X_valid_padded, y_valid)
    )

    val_acc_per_epoch = lstm_history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    log(f'Best epoch: {best_epoch}')
    log('LSTM Model completed\n')

    lstm_model.save(model_path)
    log("Model saved successfully.")

    helper.plot_history(lstm_history, model_name, best_hps.values)
    helper.save_summary(lstm_model, lstm_history, best_hps, model_name)

    with open(summary_path, 'a') as f:
        f.write('\n\n##### LSTM #####:\n')

    lstm_model.fit(
        X_train_padded,
        y_train,
        batch_size=64,
        epochs=1,
        validation_data=(X_valid_padded, y_valid),
        callbacks=[metrics_callback]
    )

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

    log(f"Data shape: {X_train_padded.shape}") if verbose else None
    log(f"CNN input shape: {cnn_model.input_shape}")  if verbose else None
    log(f"LSTM input shape: {lstm_model.input_shape}")  if verbose else None

    # --- Build proper multi-input ensemble (LSTM: (T,), CNN: (T,1)) ---
    n_timesteps = X_train_padded.shape[1]  # T
    log("E2") if verbose else None

    inp_lstm = keras.Input(shape=(n_timesteps,), dtype="int32",   name="ensemble_lstm_in")
    log("E3") if verbose else None

    inp_cnn  = keras.Input(shape=(n_timesteps, 1), dtype="float32", name="ensemble_cnn_in")
    log("E4") if verbose else None

    models_for_ensemble = [lstm_model, cnn_model]  # order doesn't matter (rank-based routing)
    log("E5") if verbose else None
    ensemble = model_builders.create_ensemble(
        models=models_for_ensemble,
        inputs=[inp_lstm, inp_cnn],
        final_dim=final_dim,
        final_activation=final_activation
    )
    log("E6") if verbose else None

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
    log("E7") if verbose else None

    ensemble.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    log("E8") if verbose else None

    log("E9") if verbose else None

    train_ds = model_builders.make_ensemble_ds(
        X_train_padded,
        y_train,
        cnn_model=cnn_model,
        batch_size=16,
        shuffle=True,
        seed=SEED
    )

    valid_ds = model_builders.make_ensemble_ds(
        X_valid_padded,
        y_valid,
        cnn_model=cnn_model,
        batch_size=16,
        shuffle=False,
        seed=SEED
    )

    history = ensemble.fit(
        train_ds,
        epochs=max_epoch_ensemble,
        verbose=1,
        validation_data=valid_ds,
        callbacks=callbacks,
    )
    log("E10") if verbose else None

    # Plot training history
    helper.plot_history(history, model_name, "ensemble")
    
    # Write summary section
    ensemble.save(model_path)
    log(f"Saving Ensemble")

    # Write summary section
    with open(summary_path, 'a') as f:
        f.write('\n\n-##### Ensemble #####:\n')

    X_valid_cnn_for_metrics = model_builders.make_cnn_view_np(X_valid_padded, cnn_model)

    metrics_callback = model_builders.MetricsCallback(
        test_data=[X_valid_padded, X_valid_cnn_for_metrics],
        y_true=y_valid,
        name=project_name
    )

    log("E11") if verbose else None

    # write one final classification report without re-training
    metrics_callback.set_model(ensemble)
    metrics_callback.on_epoch_end(epoch=max_epoch_ensemble, logs=None)

if verbose:
    log("Ensemble architecture:")
    ensemble.summary(print_fn=log)

helper.validate_model_classes(ensemble, metadata)

if args.validate_models:
    # Try to load cached validation set (works when models are loaded without retraining)
    val_cache_path = f"models/{project_name}_val_cache.npz"
    VAL = None
    if os.path.isfile(val_cache_path):
        try:
            _cache = np.load(val_cache_path, allow_pickle=False)
            X_valid_lstm = _cache["X_valid_padded"].astype("int32")      # (N, T)
            y_valid_cached = _cache["y_valid"]
            X_valid_cnn = model_builders.make_cnn_view_np(X_valid_lstm, cnn_model)
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
            ens_loss, ens_acc = ensemble.evaluate(
                {
                    "ensemble_lstm_in": X_valid_lstm,
                    "ensemble_cnn_in": X_valid_cnn
                },
                y_valid_cached,
                verbose=0
            )
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
 
    X_valid_cnn_pred = model_builders.make_cnn_view_np(X_valid_padded, cnn_model)

    predictions = ensemble.predict({
        "ensemble_lstm_in": X_valid_padded,
        "ensemble_cnn_in": X_valid_cnn_pred
    })
    predictions_df = pd.DataFrame(predictions)
    predictions_df = process_dataframe(predictions_df, labels, "EN")

    predictions_lstm = lstm_model.predict(X_valid_padded)
    predictions_df_lstm = pd.DataFrame(predictions_lstm)
    predictions_df_lstm = process_dataframe(predictions_df_lstm, labels, "LSTM")

    # predictions_df_lstm = predictions_df_lstm.rename(columns={0: 'pred_lstm'})
    # predictions_df_lstm['pred_bin_lstm'] = (predictions_df_lstm['pred_lstm']).round().astype(int)

    predictions_cnn = cnn_model.predict(X_valid_cnn_pred)
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

####################################################################
####################################################################
# ------------------------------------------------------------------
# Predict query sequences
# ------------------------------------------------------------------

log("")
log("==============================================================")
log("Starting query prediction")
log("==============================================================")

log(f"Query dataset: {query_name}")
log(f"Model: {model_name}")
log(f"Number of query sequences: {len(X_query_padded):,}")

# ------------------------------------------------------------------
# Prepare CNN-compatible representation
# ------------------------------------------------------------------

log("")
log("Preparing CNN representation")

if verbose:
    log(f"LSTM input shape: {X_query_padded.shape}")

X_query_cnn = model_builders.make_cnn_view_np(
    X_query_padded,
    cnn_model
)

if verbose:
    log(f"CNN input shape: {X_query_cnn.shape}")

# ------------------------------------------------------------------
# Ensemble prediction
# ------------------------------------------------------------------

log("")
log("Running ensemble prediction")

predictions = ensemble.predict(
    {
        "ensemble_lstm_in": X_query_padded,
        "ensemble_cnn_in": X_query_cnn
    },
    verbose=1
)

log(f"Ensemble prediction completed")
log(f"Prediction matrix shape: {predictions.shape}")

predictions_df = pd.DataFrame(predictions)

predictions_df = process_dataframe(
    predictions_df,
    labels,
    "EN"
)

# ------------------------------------------------------------------
# LSTM branch prediction
# ------------------------------------------------------------------

log("")
log("Running LSTM branch prediction")

predictions_lstm = lstm_model.predict(
    X_query_padded,
    verbose=1
)

log(f"LSTM prediction matrix shape: {predictions_lstm.shape}")

predictions_df_lstm = pd.DataFrame(predictions_lstm)

predictions_df_lstm = process_dataframe(
    predictions_df_lstm,
    labels,
    "LSTM"
)

# ------------------------------------------------------------------
# CNN branch prediction
# ------------------------------------------------------------------

log("")
log("Running CNN branch prediction")

predictions_cnn = cnn_model.predict(
    X_query_cnn,
    verbose=1
)

log(f"CNN prediction matrix shape: {predictions_cnn.shape}")

predictions_df_cnn = pd.DataFrame(predictions_cnn)

predictions_df_cnn = process_dataframe(
    predictions_df_cnn,
    labels,
    "CNN"
)

# ------------------------------------------------------------------
# Merge metadata and predictions
# ------------------------------------------------------------------

log("")
log("Merging prediction outputs")

sequences_dict_query.reset_index(drop=True, inplace=True)
predictions_df.reset_index(drop=True, inplace=True)
predictions_df_lstm.reset_index(drop=True, inplace=True)
predictions_df_cnn.reset_index(drop=True, inplace=True)

merged_df = pd.concat(
    [
        sequences_dict_query,
        predictions_df,
        predictions_df_lstm,
        predictions_df_cnn
    ],
    axis=1
)

merged_df = reorder_columns(merged_df)

log(f"Merged dataframe shape: {merged_df.shape}")

if verbose:
    log(f"Final columns:\n{list(merged_df.columns)}")

# ------------------------------------------------------------------
# Format prediction probabilities for readable export
# ------------------------------------------------------------------

prediction_prefixes = (
    "EN_",
    "LSTM_",
    "CNN_"
)

prediction_columns = [
    col
    for col in merged_df.columns
    if (
        col.startswith(prediction_prefixes)
        and not col.endswith("_class")
    )
]

merged_df[prediction_columns] = (
    merged_df[prediction_columns]
    .astype(float)
    .round(6)
)

# optional: suppress tiny floating point noise
merged_df[prediction_columns] = (
    merged_df[prediction_columns]
    .mask(
        merged_df[prediction_columns] < 1e-6,
        0
    )
)

# ------------------------------------------------------------------
# Save CSV output
# ------------------------------------------------------------------

csv_output_path = (
    f"predictions/{query_name}.{model_name}.query.csv"
)

merged_df.to_csv(
    csv_output_path,
    index=False,
    float_format="%.6f"
)

log("")
log(f"Prediction table saved:")
log(f"  {csv_output_path}")

# ------------------------------------------------------------------
# Save FASTA output
# ------------------------------------------------------------------

fasta_output_path = (
    f"predictions/{query_name}.{model_name}.query.fasta"
)

write_to_fasta(
    merged_df,
    fasta_output_path
)

log(f"Prediction FASTA saved:")
log(f"  {fasta_output_path}")

# ------------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------------

log("")
log("Prediction summary")

if "EN_class" in merged_df.columns:

    class_counts = (
        merged_df["EN_class"]
        .value_counts(dropna=False)
        .sort_index()
    )

    for cls, count in class_counts.items():

        fraction = count / len(merged_df)

        log(
            f"  {cls:<20} "
            f"{count:>8,} "
            f"({fraction:.2%})"
        )

log("")
log("Query prediction finished successfully")
log("==============================================================")