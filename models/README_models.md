# MetAnoDe pretrained models

This directory contains the pretrained MetAnoDe models. Each model is stored in its own versioned directory. The directory name is the model identifier supplied to MetAnoDe with `-p`.

For example:

```bash
python metanode.py -query query.fasta -p bacterial_16S_m1.3
```

loads the model from:

```text
models/bacterial_16S_m1.3/
```

## Model naming

Model identifiers follow the convention:

```text
<target>_<marker>_m<major>.<minor>
```

Current examples are:

```text
bacterial_16S_m1.3
plant_ITS2_m1.4
fungal_ITS2_m1.1
```

The model identifier is intentionally short. Detailed information about the marker region, primer combination, reference database, preprocessing requirements, target group, and anomaly classes should be stored in the model-specific `parameters.json` and, where applicable, in a model-specific `README.md`.

### Versioning

A minor version increment, for example `m1.3` to `m1.4`, denotes an updated model for the same biological target and amplicon definition, such as retraining with improved reference data, off-target data, hyperparameters, or minor implementation changes.

A major version increment, for example `m1.x` to `m2.0`, should be used for substantial changes that may materially alter model behaviour, such as primer combinations, a changed anomaly-class design, major architecture changes, or incompatible preprocessing.

Models based on substantially different marker regions should receive distinct model identifiers rather than being treated only as new versions of an existing assay.

## Directory structure

A fully trained model directory normally has the following structure:

```text
<model_id>/
├── README.md                  # optional model-specific human-readable documentation
├── parameters.json            # required model metadata
├── hyperparameters.json       # optional training hyperparameter overrides
├── CNN.keras                  # required pretrained CNN
├── CNN.txt                    # model summary
├── LSTM.keras                 # required pretrained LSTM
├── LSTM.txt                   # model summary
├── Ensemble.keras             # required pretrained MetAnoDe ensemble
├── Ensemble.best.keras        # best ensemble checkpoint, if retained
├── Ensemble.config            # required ensemble configuration
├── Ensemble.token             # required model compatibility token
├── Stats.txt                  # training and validation statistics
├── val_cache.npz              # cached validation data, if retained
└── plots/
    ├── CNN.training_validation.pdf
    ├── LSTM.training_validation.pdf
    └── Ensemble.training_validation.pdf
```

MetAnoDe currently requires the following files to use an existing pretrained model:

```text
CNN.keras
LSTM.keras
Ensemble.keras
Ensemble.config
Ensemble.token
parameters.json
```

Other files are training, validation, documentation, or diagnostic artifacts and are not required for prediction.

A directory containing only `parameters.json` is therefore not yet a complete pretrained model. It can be used as a model-development directory while the corresponding model is being trained.

## Model metadata

`parameters.json` is the machine-readable description of the model. In addition to class labels and class counts, model metadata should document the biological and technical definition of the model wherever applicable.

Recommended fields include:

```json
{
  "model_id": "arthropod_COI_m1.0",
  "model_version": "1.0",
  "target_group": "Arthropoda",
  "marker": "COI",
  "region": "mlCOIintF-jgHCO2198",
  "forward_primer_name": "mlCOIintF",
  "forward_primer": "GGWACWGGWTGAACWGTWTAYCCYCC",
  "reverse_primer_name": "jgHCO2198",
  "reverse_primer": "TAIACYTCIGGRTGICCRAARAAYCA",
  "preprocessing": {
    "primers_removed": true,
    "reverse_complement": false
  },
  "reference_database": "BOLD",
  "labels": {
    "0": "true",
    "1": "substitution",
    "2": "indel",
    "3": "chimera"
  },
  "n_classes": 4,
  "core_classes": 4,
  "off_target_classes": []
}
```

The exact fields can differ between models, but the metadata should contain enough information to determine which amplicon the model represents and which preprocessing was used during training.

## Model-specific documentation

For markers where query preparation is not self-evident, a model-specific `README.md` is recommended. This is particularly important for ITS models and for models tied to a specific primer-defined amplicon.

Such documentation should state:

- biological target group;
- marker and amplicon definition;
- forward and reverse primer names and sequences;
- whether primers are expected to be removed;
- expected sequence orientation;
- whether MetAnoDe applies reverse-complement transformation;
- reference database and database release;
- construction and filtering of the training reference set;
- biological off-target classes;
- any additional preprocessing required before prediction;
- known limitations of the pretrained model.

## Current model directories

```text
models/
├── bacterial_16S_m1.3/
├── fungal_ITS2_m1.1/
└── plant_ITS2_m1.4/
```

At present, `bacterial_16S_m1.3` and `plant_ITS2_m1.4` contain the complete pretrained model artifact set required by MetAnoDe.

`fungal_ITS2_m1.1` currently contains only `parameters.json` and should therefore be regarded as a model under development until the trained model artifacts are added.
