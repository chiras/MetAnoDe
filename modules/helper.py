import matplotlib.pyplot as plt
from Bio import SeqIO
import pandas as pd 
from textwrap import wrap
import random
import re


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
            

import random
import pandas as pd

def create_artificial_chimera(df, num_sequences):
    headers = df['headers'].tolist()
    sequences = df['sequences'].tolist()
    
    chimera_headers = []
    chimera_sequences = []
    chimera_sizes = []
    chimera_target = 1
    chimera_target4d = 3
    chimera_size = 1

    sequence_length = len(sequences)
    
    for _ in range(num_sequences):
        # Pick first random sequence
        idx1 = random.randint(0, sequence_length - 1)
        seq1 = sequences[idx1]
        len1 = len(seq1)
        cut1 = random.randint(len1 // 4, 3 * len1 // 4)
        A = seq1[:cut1]

        # Pick second random sequence
        idx2 = random.randint(0, sequence_length - 1)
        seq2 = sequences[idx2]
        len2 = len(seq2)
        cut2 = random.randint(len2 // 4, 3 * len2 // 4)
        B = seq2[-cut2:]

        # Concatenate A and B to create the chimera sequence
        chimera_sequence = A + B

        # Create header for the chimera sequence
        chimera_header = f"art-chim_{headers[idx1]}_{headers[idx2]}"
        
        # Append the chimera sequence and header to the lists
        chimera_headers.append(chimera_header)
        chimera_sequences.append(chimera_sequence)
        chimera_sizes.append(chimera_size)

    # Create the new chimera DataFrame
    chimera_df = pd.DataFrame({
        'headers': chimera_headers,
        'sequences': chimera_sequences,
        'Target': [chimera_target] * num_sequences,
        'Target4D': [chimera_target4d] * num_sequences,
        'sizes': chimera_sizes
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