import matplotlib.pyplot as plt
from Bio import SeqIO
import pandas as pd 
from textwrap import wrap
import random
import re
import os
import json

os.makedirs("plots", exist_ok=True)
os.makedirs("predictions", exist_ok=True)

def build_labels_from_args(args):
    """
    Build dynamic class labels.

    Fixed classes:
        0 = true
        1 = substitution
        2 = indel
        3 = chimera

    Dynamic classes:
        4+ = off-target references from -ot
    """

    labels = {
        0: "true",
        1: "substitution",
        2: "indel",
        3: "chimera"
    }

    # parser uses dest='offtargets'
    if getattr(args, "offtargets", None):

        ot_files = args.offtargets.split(",")

        for i, ot_path in enumerate(ot_files):

            class_id = 4 + i

            label = os.path.basename(ot_path)
            label = os.path.splitext(label)[0]

            # cleanup common suffixes/prefixes
            cleanup_tokens = [
                ".trim",
                ".derep",
                ".fasta",
                ".fa"
            ]

            for token in cleanup_tokens:
                label = label.replace(token, "")

            labels[class_id] = label

    return labels

def validate_model_classes(model, metadata):

    n_model = model.output_shape[-1]
    n_meta = metadata["n_classes"]

    if n_model != n_meta:

        raise RuntimeError(
            f"Class mismatch:\n"
            f"Model outputs {n_model} classes\n"
            f"Metadata defines {n_meta} classes"
        )
    
def save_model_metadata(
    project_name: str,
    labels: dict,
    extra_metadata: dict = None
):

    metadata = {
        "labels": {str(k): v for k, v in labels.items()},
        "n_classes": len(labels)
    }

    if extra_metadata is not None:
        metadata.update(extra_metadata)

    out_path = f"models/{project_name}.parameters.json"

    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[MetAnoDe] Saved metadata: {out_path}")

def load_model_metadata(project_name: str) -> dict:
    """
    Load model metadata from JSON.
    """

    path = f"models/{project_name}.parameters.json"

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    with open(path, "r") as f:
        metadata = json.load(f)

    # convert label keys back to int
    metadata["labels"] = {
        int(k): v
        for k, v in metadata["labels"].items()
    }

    return metadata

def load_hp_override(json_path, model_label):
    """
    Load static hyperparameters from JSON if present.
    Returns dict or None.
    Supports either:
      {"learning_rate": 0.0003, ...}
    or:
      {"CNN": {...}, "LSTM": {...}}
    """
    if not os.path.isfile(json_path):
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    # allow nested model-specific structure
    if model_label in data and isinstance(data[model_label], dict):
        hp = data[model_label]
    else:
        hp = data

    print(f"Loaded hyperparameter override for {model_label} from {json_path}: {hp}")
    return hp

def split_files_exist(project_name: str, split_dir: str = "splits") -> bool:
    paths = get_split_paths(project_name, split_dir)
    return all(os.path.isfile(p) for p in paths.values())
    
def get_split_paths(project_name: str, split_dir: str = "splits") -> dict:
    os.makedirs(split_dir, exist_ok=True)
    return {
        "train_csv": os.path.join(split_dir, f"{project_name}_train.csv.gz"),
        "valid_csv": os.path.join(split_dir, f"{project_name}_valid.csv.gz"),
        "meta_json": os.path.join(split_dir, f"{project_name}_split_meta.json"),
    }

def split_files_exist(project_name: str, split_dir: str = "splits") -> bool:
    paths = get_split_paths(project_name, split_dir)
    return all(os.path.isfile(p) for p in paths.values())

def save_split(train_df: pd.DataFrame, valid_df: pd.DataFrame, meta: dict,
               project_name: str, split_dir: str = "splits") -> dict:
    paths = get_split_paths(project_name, split_dir)

    train_df.to_csv(paths["train_csv"], index=False, compression="gzip")
    valid_df.to_csv(paths["valid_csv"], index=False, compression="gzip")

    with open(paths["meta_json"], "w") as f:
        json.dump(meta, f, indent=2)

    return paths

def load_split(project_name: str, split_dir: str = "splits"):
    paths = get_split_paths(project_name, split_dir)

    train_df = pd.read_csv(paths["train_csv"], compression="gzip")
    valid_df = pd.read_csv(paths["valid_csv"], compression="gzip")

    with open(paths["meta_json"], "r") as f:
        meta = json.load(f)

    return train_df, valid_df, meta

def sample_dataframe(df, n):
    return df.sample(n=n, random_state=42).reset_index(drop=True)

def upsample_dataframe(df, n):
    return df.sample(n=n, weights=df['sizes'], replace=True, random_state=42).reset_index(drop=True)

def read_fasta(file_path, target, target2):
    sequences = []
    headers = []
    targets = []
    sizes = []
    with open(file_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):            
            if len(str(record.id)) >0 and len(str(record.seq)) >0:
                output_string = ''.join([char if char in 'ACGT' else 'N' for char in str(record.seq).upper()])
                sequences.append(output_string)
                headers.append(str(record.id))
                targets.append(int(target))
                size_match = re.search(r'size=(\d+)', str(record.id))
                if size_match:
                    sizes.append(int(size_match.group(1)))
                else:
                    sizes.append(int(1))                 

    sequences_dict = {}
    sequences_dict['headers'] = headers 
    sequences_dict['sequences'] = sequences 
    sequences_dict['Target'] = targets 
    sequences_dict['Target4D'] = target2 
    sequences_dict['sizes'] = sizes 

    sequences_dict = pd.DataFrame(sequences_dict)
    return sequences_dict

def plot_history(history, model_name, values):
    if values != "ensemble":
        values_string = "\n".join(["=".join([key, str(val)]) for key, val in values.items()])
    else:
        values_string=values
        # Extracting loss, accuracy, and validation loss from history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)

    # Plotting the history
    plt.figure(figsize=(10, 5))
    plt.suptitle(wrap(f'Training and Validation Loss\n{values_string}'))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title(model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Saving the plot as a PDF file
    plt.savefig(f'plots/{model_name}.training_validation.pdf')
    plt.close()

def save_summary(model, history, best_hps, model_name): 
    # Summarize the model
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = "\n".join(model_summary)

    # Get the best epoch
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

    # Save the summary, hyperparameters, and metrics
    summary_path = f"models/{model_name}.txt"

    with open(summary_path, 'w') as f:
        f.write('Model Summary:\n')
        f.write(model_summary)
        f.write('\n\nBest Hyperparameters:\n')
        if best_hps != "ensemble":
            for key, value in best_hps.values.items():
                f.write(f"{key}: {value}\n")
        f.write('\nBest Epoch: %d\n' % best_epoch)
        f.write('\nEpoch Metrics:\n')
        for epoch in range(len(history.history['accuracy'])):
            f.write(f"Epoch {epoch + 1} - Accuracy: {history.history['accuracy'][epoch]}, "
                    f"Val Accuracy: {history.history['val_accuracy'][epoch]}, "
                    f"Loss: {history.history['loss'][epoch]}, "
                    f"Val Loss: {history.history['val_loss'][epoch]}\n")
            


def create_artificial_chimera(
    df,
    num_sequences,
    min_frac=0.5,
    max_frac=1.5,
    target_len=None,
    max_attempt_factor=50,
    log_fn=None
):
    """
    Create artificial chimeras with length filtering and diagnostics.
    """

    headers = df['headers'].tolist()
    sequences = df['sequences'].tolist()

    if len(sequences) == 0:
        raise ValueError("create_artificial_chimera received an empty dataframe.")

    # --- ORIGINAL LENGTH STATS ---
    lengths = [len(seq) for seq in sequences]
    orig_min = min(lengths)
    orig_max = max(lengths)

    # --- TARGET LENGTH ---
    if target_len is None:
        lengths_sorted = sorted(lengths)
        mid = len(lengths_sorted) // 2
        if len(lengths_sorted) % 2 == 0:
            target_len = int((lengths_sorted[mid - 1] + lengths_sorted[mid]) / 2)
        else:
            target_len = int(lengths_sorted[mid])

    # --- FINAL FILTER RANGE ---
    min_len = max(1, int(round(target_len * min_frac)))
    max_len = max(min_len, int(round(target_len * max_frac)))

    # --- LOG SETTINGS ---
    if log_fn:
        log_fn(
            f"Chimera length settings | "
            f"orig_min={orig_min}, orig_max={orig_max}, "
            f"target={target_len}, final_min={min_len}, final_max={max_len}"
        )

    chimera_headers = []
    chimera_sequences = []
    chimera_sizes = []
    chimera_lengths = []

    chimera_target = 1
    chimera_target4d = 3
    chimera_size = 1

    sequence_count = len(sequences)
    attempts = 0
    max_attempts = max(num_sequences * max_attempt_factor, 1000)

    while len(chimera_sequences) < num_sequences and attempts < max_attempts:
        attempts += 1

        idx1 = random.randint(0, sequence_count - 1)
        seq1 = sequences[idx1]
        len1 = len(seq1)

        idx2 = random.randint(0, sequence_count - 1)
        seq2 = sequences[idx2]
        len2 = len(seq2)

        if len1 < 2 or len2 < 2:
            continue

        cut1 = random.randint(max(1, len1 // 4), max(1, 3 * len1 // 4))
        cut2 = random.randint(max(1, len2 // 4), max(1, 3 * len2 // 4))

        A = seq1[:cut1]
        B = seq2[-cut2:]
        chimera_sequence = A + B
        chimera_len = len(chimera_sequence)

        if chimera_len < min_len or chimera_len > max_len:
            continue

        chimera_header = f"art-chim_{headers[idx1]}_{headers[idx2]}"

        chimera_headers.append(chimera_header)
        chimera_sequences.append(chimera_sequence)
        chimera_sizes.append(chimera_size)
        chimera_lengths.append(chimera_len)

    if len(chimera_sequences) < num_sequences:
        raise RuntimeError(
            f"Only {len(chimera_sequences)}/{num_sequences} chimeras generated "
            f"within {min_len}-{max_len} after {attempts} attempts."
        )

    # --- OPTIONAL: log resulting distribution ---
    if log_fn:
        gen_min = min(chimera_lengths)
        gen_max = max(chimera_lengths)
        gen_mean = int(sum(chimera_lengths) / len(chimera_lengths))

        log_fn(
            f"Chimera output stats | "
            f"generated_min={gen_min}, generated_max={gen_max}, mean={gen_mean}, "
            f"attempts={attempts}"
        )

    chimera_df = pd.DataFrame({
        'headers': chimera_headers,
        'sequences': chimera_sequences,
        'Target': [chimera_target] * num_sequences,
        'Target4D': [chimera_target4d] * num_sequences,
        'sizes': [chimera_size] * num_sequences
    })

    return chimera_df

def create_artificial_errorate(df, num_sequences, typeerror):
    headers = df['headers'].tolist()
    sequences = df['sequences'].tolist()
    
    chimera_headers = []
    chimera_sequences = []
    chimera_sizes = []
    chimera_size = 1

    sequence_length = len(sequences)
    
    for _ in range(num_sequences):
        # Pick random sequence
        idx1 = random.randint(0, sequence_length - 1)
        seq1 = sequences[idx1]
        if (typeerror == "indel"):
            chimera_sequence = sim_indel(seq1, 0.05, 0.05)
            chimera_target = 1
            chimera_header = f"art-indel_{headers[idx1]}"
            chimera_target4d = 2 # indels
        if (typeerror == "subst"):
            chimera_sequence = sim_error(seq1, 0.1)
            chimera_target = 1
            chimera_header = f"art-subst_{headers[idx1]}"
            chimera_target4d = 1 # substitutions

        if (typeerror == "lowindel"):
            chimera_sequence = sim_indel(seq1, 0.005, 0.005)
            chimera_target = 0
            chimera_header = f"low-indel_{headers[idx1]}"
            chimera_target4d = 0 # true
        if (typeerror == "lowsubst"):
            chimera_sequence = sim_error(seq1, 0.005)
            chimera_target = 0
            chimera_header = f"low-subst_{headers[idx1]}"
            chimera_target4d = 0 # true

        # Append the chimera sequence and header to the lists
        chimera_headers.append(chimera_header)
        chimera_sequences.append(chimera_sequence)
        #chimera_sizes.append(chimera_size)

    # Create the new chimera DataFrame
    chimera_df = pd.DataFrame({
        'headers': chimera_headers,
        'sequences': chimera_sequences,
        'Target': [chimera_target] * num_sequences,
        'Target4D': [chimera_target4d] * num_sequences,
        'sizes': [chimera_size] * num_sequences
    })

    return chimera_df


def sim_error(seq, ps):
    #ps: substitution error rate
    out_seq = []
    for c in seq:
        r = random.uniform(0,1)
        if r < ps:
            out_seq.append(random.choice(["A","C","G","T"]))
        else:
            out_seq.append(c)
    return "".join(out_seq)

def sim_indel(seq, pi, pd):
    #pi: insertion error rate
    #pd: deletion error rate
    out_seq = []
    for c in seq:
        r = random.uniform(0,1)
        if r < pi:
            out_seq.append(random.choice(["A","C","G","T"]))        

        r = random.uniform(0,1)
        if r > pd:
            out_seq.append(c)
    return "".join(out_seq)

def split_into_kmers(sequences, k, sw):
    return [[sequence[i:i+k] for i in range(0, len(sequence) + 1 - k, sw)] for sequence in sequences]