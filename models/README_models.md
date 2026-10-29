# MetAnoDe pretrained models

This directory contains the pretrained MetAnoDe models. Each model is stored in its own versioned directory. The directory name is the model identifier supplied to MetAnoDe with `-p`.

For example:

```bash
python metanode.py -query query.fasta -p bacterial_16S-V4_m2.6
```

loads the model from:

```text
models/bacterial_16S-V4_m2.6/
```

## Model naming

Model identifiers follow the convention:

```text
<target>_<marker>_m<major>.<minor>[<suffix>]
```

for example `bacterial_16S-V4_m2.6` or `vertebrate_COI-5P_m2.3`. An optional suffix marks a training variant of the same version (e.g. `rv` for a model trained on reversed sequences); the settings actually used are documented in the model's `parameters.json`.

The model identifier is intentionally short. Detailed information about the marker region, primer combination, reference data, preprocessing, target group and anomaly classes is stored in the model-specific `parameters.json` and, where applicable, in a model-specific `README.md`.

### Versioning

A minor version increment, for example `m2.2` to `m2.3`, denotes an updated model for the same biological target and amplicon definition, such as retraining with improved reference data, off-target data, hyperparameters or minor implementation changes.

A major version increment, for example `m1.x` to `m2.x`, is used for substantial changes that may materially alter model behaviour, such as primer combinations, a changed anomaly-class design, major architecture changes or incompatible preprocessing.

The `m2` models (revision 1) differ from `m1` models in:

- **CNN architecture:** a single-unit layer between the dense layer and the output layer, left over from a binary implementation, was removed; nucleotides are represented by a learned embedding instead of integer codes in one numeric channel;
- **training:** stronger gradient clipping of the LSTM and learning-rate reduction on validation-loss plateaus; for most models fixed hyperparameters instead of per-model tuning (`hyperparameters.json`);
- **training data:** primer-defined amplicons for most markers; for fungal ITS, NCBI amplicons combined with UNITE sequences extended to the same amplicon boundaries;
- **class construction:** a marker-specific length range for true sequences (wider for length-variable markers such as ITS);
- **documentation:** a complete record of each training run in `parameters.json` (see below).

`m1` and `m2` models are not interchangeable: the same input can yield different predictions.

Models based on substantially different marker regions receive distinct model identifiers rather than being treated only as new versions of an existing assay.

`m1` models are no longer included in this repository.


## Directory structure

A fully trained model directory normally has the following structure:

```text
<model_id>/
├── README.md                  # optional model-specific human-readable documentation
├── parameters.json            # required model metadata and training record
├── hyperparameters.json       # fixed hyperparameters used instead of tuning, if applicable
├── CNN.keras                  # required pretrained CNN
├── CNN.txt                    # model summary and training course
├── LSTM.keras                 # required pretrained LSTM
├── LSTM.txt                   # model summary and training course
├── Ensemble.keras             # required pretrained MetAnoDe ensemble
├── Ensemble.best.keras        # best ensemble checkpoint, if retained
├── Ensemble.config            # required ensemble configuration
├── Ensemble.token             # required model compatibility token
├── Stats.txt                  # hold-out validation statistics (CNN, LSTM, ensemble)
├── logs/                      # training log(s): command, seed, data sizes, training course
└── plots/
    ├── CNN.training_validation.pdf
    ├── LSTM.training_validation.pdf
    └── Ensemble.training_validation.pdf
```

MetAnoDe requires the following files to use an existing pretrained model:

```text
CNN.keras
LSTM.keras
Ensemble.keras
Ensemble.config
Ensemble.token
parameters.json
```

Other files are training, validation, documentation or diagnostic artifacts and are not required for prediction. The validation cache (`val_cache.npz`) produced during training is not kept in the repository: the validation set is defined by the training data, the seed and the split recorded in `parameters.json`.

A directory without the files listed above is not a complete pretrained model. It can be used as a model-development directory while the corresponding model is being trained.

## Model metadata

`parameters.json` is the machine-readable description of the model. It contains the class labels and, for models trained with revision 1 or later, a complete record of the training run in the section `training_run`:

| Field | Content |
|---|---|
| `labels` | class index -> class name (0 true, 1 substitution, 2 indel, 3 chimera, 4+ off-target groups) |
| `offtargets` | off-target training files |
| `reverse` | `true` if the model was trained on reversed sequences; MetAnoDe then reverses the query itself (see below) |
| `cnn_input` | `embedding` or `numeric` |
| `chimera_mode` | generation of the training chimeras (`legacy` or `homologous`) |
| `training_run.command_line`, `seed`, `started`, `finished` | how and when the model was trained |
| `training_run.git`, `code_sha256`, `environment` | code version (commit and checksums) and package versions |
| `training_run.inputs` | training and off-target files with SHA-256 checksums and record counts |
| `training_run.data` | length filter, validation share, sequence counts per class (training and validation) |
| `training_run.training` | epochs, batch sizes, early stopping, learning-rate reduction, LSTM clipping |
| `training_run.models` | per component: hyperparameters, best epoch, epochs run, best validation accuracy |

Additional descriptive fields (target group, marker, primers, reference database) are recommended, for example:

```json
{
  "target_group": "Arthropoda",
  "marker": "COI",
  "region": "mlCOIintF-jgHCO2198",
  "forward_primer_name": "mlCOIintF",
  "forward_primer": "GGWACWGGWTGAACWGTWTAYCCYCC",
  "reverse_primer_name": "jgHCO2198",
  "reverse_primer": "TAIACYTCIGGRTGICCRAARAAYCA",
  "preprocessing": {"primers_removed": true},
  "reference_database": "BOLD"
}
```

## Query preparation

- **Primers must be removed** from the query sequences: all models are trained on primer-trimmed amplicons.
- **Orientation:** query sequences are expected in the forward orientation of the amplicon (forward primer at the 5' end). Models trained on reversed sequences (`"reverse": true`) reverse the query internally; the prediction table keeps the original orientation. `-rv` is only needed when training such a model.
- **Length:** sequences far outside the length range of the training data (see `training_run.data`) are outside the model's experience and should be interpreted with caution.

## Model-specific documentation

For markers where query preparation is not self-evident, a model-specific `README.md` is recommended. This is particularly important for ITS models and for models tied to a specific primer-defined amplicon.

Such documentation should state:

- biological target group;
- marker and amplicon definition;
- forward and reverse primer names and sequences;
- whether primers are expected to be removed;
- expected sequence orientation;
- reference data and release;
- construction and filtering of the training reference set;
- biological off-target classes;
- any additional preprocessing required before prediction;
- known limitations of the pretrained model.

## Current model directories

Revision 1 models (`m2`), selected per marker from several training runs by hold-out validation (all three components) and an independent in silico mock-community benchmark (MetAnoDe's added false removal and reduction of retained anomalies, class-balanced). Candidates within a marker differed by at most about 1–2 percentage points; ties were resolved by hold-out validation and the health of the CNN and LSTM components.

| Model | Off-target classes | Note |
|---|---|---|
| `bacterial_16S-V4_m2.6` | mitochondria, plastids | |
| `fungal_ITS1_m2.8` | plants | |
| `fungal_ITS2_m2.4` | plants | |
| `invertebrate_COI-5P_m2.4` | fungi, plants, protists, vertebrates | |
| `metazoan_COI-5P_m2.8` | fungi, plants, protists | |
| `plant_ITS2_m2.4` | fungi | |
| `plant_rbcL_m2.2` | bryophytes, green algae | |
| `vertebrate_12S_m2.8` | bacteria, invertebrates | |
| `vertebrate_16S_m2.8` | invertebrates | |
| `vertebrate_COI-5P_m2.3` | invertebrates, fungi, plants, protists | |

Version numbers were assigned per training run and are therefore not consecutive across markers; within a marker, a higher number does not imply a better model. Training command, seed, data and hyperparameters of each model are recorded in its `parameters.json` (`training_run`) and training log (`logs/`).
