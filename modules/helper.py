import matplotlib.pyplot as plt
from Bio import SeqIO
import pandas as pd 
from textwrap import wrap
import random
import re
import os
import json
import gzip

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

    model_dir = os.path.join("models", project_name)
    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "parameters.json")
    
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"[MetAnoDe] Saved metadata: {out_path}")

def load_model_metadata(project_name: str) -> dict:
    """
    Load model metadata from JSON.
    """

    path = os.path.join("models", project_name, "parameters.json")

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

def read_fasta(file_path, target, target2, reverse=False):
    sequences = []
    headers = []
    targets = []
    sizes = []

    # Transparently support both uncompressed and gzip-compressed FASTA files
    if file_path.lower().endswith(".gz"):
        fasta_handle = gzip.open(file_path, "rt")
    else:
        fasta_handle = open(file_path, "r")

    with fasta_handle as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            if len(str(record.id)) > 0 and len(str(record.seq)) > 0:
                output_string = ''.join(
                    [char if char in 'ACGT' else 'N'
                     for char in str(record.seq).upper()]
                )
                # Optionally reverse the sequence (no complement)
                if reverse:
                    output_string = output_string[::-1]
                sequences.append(output_string)
                headers.append(str(record.id))
                targets.append(int(target))

                size_match = re.search(r'size=(\d+)', str(record.id))
                if size_match:
                    sizes.append(int(size_match.group(1)))
                else:
                    sizes.append(1)

    sequences_dict = {}
    sequences_dict['headers'] = headers
    sequences_dict['sequences'] = sequences
    sequences_dict['Target'] = targets
    sequences_dict['Target4D'] = target2
    sequences_dict['sizes'] = sizes

    sequences_dict = pd.DataFrame(sequences_dict)
    return sequences_dict

def plot_history(history, model_dir, model_component, values):
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
    plt.title(model_component)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'ro', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title(model_component)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Saving the plot as a PDF file
    plots_dir = os.path.join(model_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f'{model_component}.training_validation.pdf'))
    plt.close()

def save_summary(model, history, best_hps, model_dir, model_component): 
    # Summarize the model
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = "\n".join(model_summary)

    # Get the best epoch
    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

    # Save the summary, hyperparameters, and metrics
    os.makedirs(model_dir, exist_ok=True)
    summary_path = os.path.join(model_dir, f"{model_component}.txt")

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
    max_attempt_factor=200,
    log_fn=None,
    mode="homologous",
    anchor_k=12,
    anchor_tries=10,
    min_diff_frac=0.03,
    min_diff_abs=8,
    index_step=3,
    index_max_seqs=30000,
):
    """
    Create artificial chimeras with length filtering and diagnostics.

    mode="homologous" (revision 1, default): PCR-like chimeras.
      - The junction lies at a shared anchor: a k-mer (anchor_k) drawn from the middle half of
        parent 1 that occurs exactly once in parent 2, near the corresponding relative position.
        The chimera is parent1[:p] + parent2[q:], with p, q the anchor positions, so there is no
        duplicated or missing segment (template switching between homologous sites).
        Parent 2 is drawn among sequences sharing the anchor (k-mer index of the middle halves,
        every index_step positions, at most index_max_seqs sequences), so that diverse reference sets
        still yield chimeras efficiently. This favours related parents, as in real PCR.
      - The chimera must differ from BOTH parents by at least max(min_diff_abs, min_diff_frac *
        length) positions; otherwise it is practically identical to a true sequence and would
        carry a contradictory label.
    mode="legacy": previous behaviour (independent cuts in both parents, 25-75% each), which
      typically produces duplicated or missing segments at the junction.
    """
    headers = df["headers"].tolist()
    sequences = df["sequences"].tolist()
    if len(sequences) == 0:
        raise ValueError("create_artificial_chimera received an empty dataframe.")

    lengths = [len(seq) for seq in sequences]
    orig_min, orig_max = min(lengths), max(lengths)
    if target_len is None:
        s = sorted(lengths)
        mid = len(s) // 2
        target_len = int((s[mid - 1] + s[mid]) / 2) if len(s) % 2 == 0 else int(s[mid])
    min_len = max(1, int(round(target_len * min_frac)))
    max_len = max(min_len, int(round(target_len * max_frac)))

    if log_fn:
        log_fn(f"Chimera settings | mode={mode}, orig_min={orig_min}, orig_max={orig_max}, "
               f"target={target_len}, final_min={min_len}, final_max={max_len}"
               + (f", anchor_k={anchor_k}, min_diff=max({min_diff_abs}, {min_diff_frac:.0%})"
                  if mode == "homologous" else ""))

    chimera_headers, chimera_sequences, chimera_lengths = [], [], []
    chimera_target, chimera_target4d, chimera_size = 1, 3, 1
    n_seq = len(sequences)
    attempts = 0
    max_attempts = max(num_sequences * max_attempt_factor, 1000)
    rejected = {"no_anchor": 0, "too_similar": 0, "length": 0, "short_parent": 0}

    kmer_index = {}
    if mode == "homologous":
        pool = list(range(n_seq))
        if n_seq > index_max_seqs:
            pool = random.sample(pool, index_max_seqs)
        for j in pool:
            sq = sequences[j]
            L = len(sq)
            for pos in range(L // 4, 3 * L // 4 - anchor_k + 1, index_step):
                kmer_index.setdefault(sq[pos:pos + anchor_k], []).append(j)

    while len(chimera_sequences) < num_sequences and attempts < max_attempts:
        attempts += 1
        idx1 = random.randint(0, n_seq - 1)
        idx2 = random.randint(0, n_seq - 1)
        if idx1 == idx2:
            continue
        seq1, seq2 = sequences[idx1], sequences[idx2]
        len1, len2 = len(seq1), len(seq2)

        if mode == "legacy":
            if len1 < 2 or len2 < 2:
                rejected["short_parent"] += 1
                continue
            cut1 = random.randint(max(1, len1 // 4), max(1, 3 * len1 // 4))
            cut2 = random.randint(max(1, len2 // 4), max(1, 3 * len2 // 4))
            chim = seq1[:cut1] + seq2[-cut2:]
        else:
            if len1 < 4 * anchor_k or len2 < 4 * anchor_k:
                rejected["short_parent"] += 1
                continue
            chim, why = None, "no_anchor"
            for _ in range(anchor_tries):
                p = random.randint(len1 // 4, 3 * len1 // 4 - anchor_k)
                anchor = seq1[p:p + anchor_k]
                cands = kmer_index.get(anchor)
                if not cands:
                    continue
                idx2 = random.choice(cands)
                if idx2 == idx1:
                    continue
                seq2 = sequences[idx2]
                len2 = len(seq2)
                q = seq2.find(anchor)
                # unique in parent 2, and near the corresponding relative position (homologous site)
                if q < 0 or seq2.find(anchor, q + 1) >= 0 or abs(q / len2 - p / len1) > 0.15:
                    continue
                cand = seq1[:p] + seq2[q:]
                # difference to parent 1: donated part vs. parent 1's own continuation;
                # difference to parent 2: retained part vs. parent 2's own beginning (aligned at the junction)
                d1 = _mismatches(seq2[q:], seq1[p:])
                d2 = _mismatches(seq1[:p][::-1], seq2[:q][::-1])
                need = max(min_diff_abs, int(round(min_diff_frac * len(cand))))
                if d1 < need or d2 < need:
                    why = "too_similar"
                    continue
                chim = cand
                break
            if chim is None:
                rejected[why] += 1
                continue

        if not (min_len <= len(chim) <= max_len):
            rejected["length"] += 1
            continue
        chimera_headers.append(f"art-chim_{headers[idx1]}_{headers[idx2]}")
        chimera_sequences.append(chim)
        chimera_lengths.append(len(chim))

    if len(chimera_sequences) < num_sequences:
        raise RuntimeError(
            f"Only {len(chimera_sequences)}/{num_sequences} chimeras generated within {min_len}-{max_len} "
            f"after {attempts} attempts (rejected: {rejected}). For mode='homologous', a smaller anchor_k "
            f"or a larger max_attempt_factor helps with very diverse reference sets.")

    if log_fn:
        log_fn(f"Chimera output stats | generated_min={min(chimera_lengths)}, generated_max={max(chimera_lengths)}, "
               f"mean={int(sum(chimera_lengths) / len(chimera_lengths))}, attempts={attempts}, rejected={rejected}")

    return pd.DataFrame({
        "headers": chimera_headers,
        "sequences": chimera_sequences,
        "Target": [chimera_target] * num_sequences,
        "Target4D": [chimera_target4d] * num_sequences,
        "sizes": [chimera_size] * num_sequences,
    })

def create_artificial_errorate(df, num_sequences, typeerror):
    headers = df['headers'].tolist()
    sequences = df['sequences'].tolist()

    chimera_headers = []
    chimera_sequences = []
    chimera_sizes = []
    chimera_size = 1

    # Store the rates actually sampled during generation
    sampled_rates = []

    sequence_length = len(sequences)

    for _ in range(num_sequences):
        # Pick random sequence
        idx1 = random.randint(0, sequence_length - 1)
        seq1 = sequences[idx1]

        if typeerror == "indel":
            indel_rate = random.uniform(0.025, 0.05)
            sampled_rates.append(indel_rate)

            chimera_sequence = sim_indel(
                seq1,
                indel_rate,
                indel_rate
            )
            chimera_target = 1
            chimera_header = f"art-indel_{headers[idx1]}"
            chimera_target4d = 2  # indels

        if typeerror == "subst":
            subst_rate = random.uniform(0.0667, 0.20) # 5-15% expected actual nucleotide divergence.
            #subst_rate = random.uniform(0.05, 0.10) # 3.75-7.5% expected actual nucleotide divergence.
            sampled_rates.append(subst_rate)

            chimera_sequence = sim_error(seq1, subst_rate)
            chimera_target = 1
            chimera_header = f"art-subst_{headers[idx1]}"
            chimera_target4d = 1  # substitutions

        if typeerror == "lowindel":
            chimera_sequence = sim_indel(seq1, 0.005, 0.005)
            chimera_target = 0
            chimera_header = f"low-indel_{headers[idx1]}"
            chimera_target4d = 0  # true

        if typeerror == "lowsubst":
            chimera_sequence = sim_error(seq1, 0.005)
            chimera_target = 0
            chimera_header = f"low-subst_{headers[idx1]}"
            chimera_target4d = 0  # true

        chimera_headers.append(chimera_header)
        chimera_sequences.append(chimera_sequence)

    # Report sampled high-error rates
    if sampled_rates:
        rate_min = min(sampled_rates)
        rate_max = max(sampled_rates)
        rate_mean = sum(sampled_rates) / len(sampled_rates)

        if typeerror == "subst":
            # sim_error() can randomly select the original nucleotide.
            # Therefore only 3/4 of attempted substitutions are expected
            # to produce an actual nucleotide difference.
            div_min = rate_min * 0.75
            div_max = rate_max * 0.75
            div_mean = rate_mean * 0.75

            print(
                f"--- Substitution anomaly generation: "
                f"sampled rate min={rate_min * 100:.2f}%, "
                f"max={rate_max * 100:.2f}%, "
                f"mean={rate_mean * 100:.2f}% | "
                f"expected actual sequence divergence "
                f"min={div_min * 100:.2f}%, "
                f"max={div_max * 100:.2f}%, "
                f"mean={div_mean * 100:.2f}%"
            )

        elif typeerror == "indel":
            # The same sampled rate is independently applied to
            # insertion and deletion at each sequence position.
            # Report these separately rather than treating them as
            # substitution-like sequence divergence.
            print(
                f"--- Indel anomaly generation: "
                f"sampled insertion/deletion rate each "
                f"min={rate_min * 100:.2f}%, "
                f"max={rate_max * 100:.2f}%, "
                f"mean={rate_mean * 100:.2f}% | "
                f"approx. combined indel-event probability "
                f"min={rate_min * 2 * 100:.2f}%, "
                f"max={rate_max * 2 * 100:.2f}%, "
                f"mean={rate_mean * 2 * 100:.2f}%"
            )

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