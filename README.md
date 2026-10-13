# MetAnoDe

MetAnoDe employs an alignment-free approach that complements existing tools in cleaning up metabarcoding data. The software uses deep neural networks, more specifically a convolutional neural network (CNN) and a long short-term memory network (LSTM), combined into an ensemble model.

**Citation:** Keller 2025 EcoEvoRxiv https://doi.org/10.32942/X2792N (peer-review version pending)

Pre-trained models are available in the subfolder `models` for:

| Marker | Target region | Off-target classes | Model name | fw primer | rv primer |
|---|---|---|---|---|---|
| Bacterial 16S-V4 | Kozich et al. (2014, https://doi.org/10.1128/AEM.01043-13) | mitochondria, chloroplast | `bacterial_16S-V4_m1.4` | GTGCCAGCMGCCGCGGTAA | GGACTACHVGGGTWTCTAAT |
| Plant ITS2 | Sickel et al. (2015, https://doi.org/10.1186/s12898-015-0051-y) | fungi | `plant_ITS2_m1.4` | ATGCGATACTTGGTGTGAAT | TCCTCCGCTTATTGATATGC |
| Fungal ITS2 | Ihrmark et al. (2012) / White et al. (1990) | plants | `fungal_ITS2_m1.3` | GTGARTCATCGARTCTTTG | TCCTCCGCTTATTGATATGC |
| Invertebrate COI-5P | Leray et al. 2013 Frontiers in Zoology https://link.springer.com/article/10.1186/1742-9994-10-34 | fungi, plants, protists, vertebrates | `invertebrate_COI-5P_m1.2` | GGWACWGGWTGAACWGTWTAYCCYCC | TAIACYTCIGGRTGICCRAARAAYCA |

More models can be trained on demand (see below, or file an issue and tag @chiras).

## Runtime considerations

The script supports both GPU (CUDA) and CPU processing. For predictions with pre-trained or previously self-trained models, the difference is hardly noticeable compared with general metabarcoding procedures. For training of new models, GPUs reduce runtime considerably.

**Recommendation:**
* **Predictions only:** GPU (CUDA) or CPU (1-2 minutes runtime per run)
* **Training of new marker models:** GPU (CUDA) strongly recommended (6-7 h on GPU, multiple days on a moderately equipped CPU-only PC; see training section for details)

## Installation (Docker, recommended)

Docker is the recommended way to run MetAnoDe. All package versions are fixed in the image, so results do not depend on the local Python installation. For direct access to the Python environment (e.g. to modify or debug the code), see [Alternative installation: conda](#alternative-installation-conda-for-advanced-users-and-developers) at the end of this README.

***Only for GPU support:*** *install AND configure the NVIDIA Container Toolkit*
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Build the image from the `Dockerfile` in the `docker` subfolder:

```sh
# build / rebuild (rebuild to include GitHub updates)
docker build --no-cache -t metanode:tf-25.01 .

# run with GPU support
docker run --gpus all -v $PWD:/data --user $(id -u):$(id -g) --rm metanode:tf-25.01 -query /data/<query.fasta> -p <model_name> -t 20

# run without GPU support
docker run -v $PWD:/data --user $(id -u):$(id -g) --rm metanode:tf-25.01 -query /data/<query.fasta> -p <model_name> -t 20
```

[OPTIONAL, not recommended] Force overwrite of pre-trained files in `/data`: add flag `-e PRETRAINED_OVERWRITE=1`

## Predictions using pre-trained models

Predictions can be generated directly with the pre-trained models. Further options are shown when running the script without arguments.

```sh
docker run --gpus all -v $PWD:/data --user $(id -u):$(id -g) --rm metanode:tf-25.01 -query /data/<query.fasta> -p <model_name>
```

`<model_name>` is one of the pre-trained models in the subfolder `models` (see table above).

Adapter and primer sequences must be removed before analysis to match the model, because they differ between library preparation strategies.

## Predictions with other target regions and training of new models

The workflow is fully automated and can be adapted to other target regions, but this requires training new models from scratch. Milestones are set during execution, so steps already completed are skipped when the run is repeated.

Correctly trimmed and dereplicated reference sequences of the target group must be provided with `-db <ref.fasta>`. Optionally, known off-target amplicons can be added with `-ot <ot1.fasta>,<ot2.fasta>,<ot3.fasta>[,...]`. Each file becomes a separate off-target class, named after the file. A model name (`-p`) collects all models and parameters in `models/<model_name>/`. A trained model can be reused for new data by giving the same model name. Compressed (`.gz`) FASTA files are accepted for all inputs.

Example of training and prediction in one run:

```sh
docker run --gpus all -v $PWD:/data --rm metanode:tf-25.01 \
    -query <query.fasta> \
    -p <model_name> \
    -db <ref.fasta> \
    -ot <ot1.fasta>,<ot2.fasta>,<ot3.fasta>
```

The pre-trained models were generated with the following calls (run from the root directory of the repository). In these examples, `-query` is the training database itself and serves only as a technical test of the prediction step, not as performance evaluation.

```sh
docker run --gpus all \
    -v $PWD:/data \
    --rm metanode:tf-25.01 \
    -db data/plant_ITS2/ITS2.Quaresma2024.all.trimmedpy2.fasta.gz \
    -ot data/plant_ITS2/ITS2.fungi.trim.1.fasta.gz \
    -p plant_ITS2_m1.4 \
    -query data/plant_ITS2/ITS2.Quaresma2024.all.trimmedpy2.fasta.gz \
    -e 40

docker run --gpus all \
    -v $PWD:/data \
    --rm metanode:tf-25.01 \
    -db data/fungi_ITS2/unite_ITS2_hybrid.fasta.gz \
    -ot data/fungi_ITS2/plant_ITS2_random50000.fasta.gz \
    -p fungal_ITS2_m1.3 \
    -query data/fungi_ITS2/unite_ITS2_hybrid.fasta.gz \
    -e 40

docker run --gpus all \
    -v $PWD:/data \
    --rm metanode:tf-25.01 \
    -db data/bacteria_16S-V4/16S.silva.trim.derep.fa.gz \
    -ot data/bacteria_16S-V4/16S.mitochondria.trim.1.derep.fasta.gz,data/bacteria_16S-V4/16S.chloroplast.trim.1.derep.fasta.gz \
    -p bacterial_16S-V4_m1.4 \
    -query data/bacteria_16S-V4/16S.silva.trim.derep.fa.gz \
    -e 40

docker run --gpus all \
    -v $PWD:/data \
    --rm metanode:tf-25.01 \
    -db data/invertebrate_COI-5P/bold_COI_m1.1.invertebrates.taxonomic_sample.primer_trimmed.fasta.gz \
    -ot data/invertebrate_COI-5P/bold_COI_m1.1.fungi.taxonomic_sample.primer_trimmed.fasta.gz,data/invertebrate_COI-5P/bold_COI_m1.1.plants.taxonomic_sample.primer_trimmed.fasta.gz,data/invertebrate_COI-5P/bold_COI_m1.1.protists.taxonomic_sample.primer_trimmed.fasta.gz,data/invertebrate_COI-5P/bold_COI_m1.1.vertebrata.taxonomic_sample.primer_trimmed.fasta.gz \
    -p invertebrate_COI-5P_m1.2 \
    -query data/invertebrate_COI-5P/bold_COI_m1.1.invertebrates.taxonomic_sample.primer_trimmed.fasta.gz \
    -e 40
```

Pre-trained models were trained on Ubuntu 24.04 with GPU support. The software was also tested on Ubuntu 22.04/24.04 with and without GPU support, and on macOS 12.3 without GPU support. Training hardware: Intel i7 with 256 GB RAM and 24 GB NVIDIA RTX 4090. Training took 6-7 h per model.

There is no strict limit on the number or length of reference sequences, or on the number of off-target classes. However, memory for encoding and training can become a constraint depending on the available hardware.

Each run writes a log with the full command line, random seed and the parameters of the simulated anomaly classes, for provenance.

## All command-line parameters

| Parameter | For prediction | For modelling | Argument | Default | Description |
|---|---:|---:|---|---|---|
| `-query` | Required | Required | `<query.fasta>` | none | Query FASTA file (plain or `.gz`) containing amplicon sequences to be classified. |
| `-p` | Required | Required | `<model_name>` | none | Model name. Used to load existing models or to name newly trained models and associated output files. |
| `-db` | No | Required | `<ref.fasta>` | none | Reference database of the target group in FASTA format (plain or `.gz`). Required when training a new model. |
| `-ot` | No | Optional | `<ot1.fasta>,<ot2.fasta>,...` | none | Comma-separated list of known off-target amplicon FASTA files. Each file is a separate off-target class during training. |
| `-2c` | No | Optional | none | `FALSE` | Switch from multiclass to binary classification. Deprecated. |
| `-r` | No | Optional | none | `FALSE` | Recalibrate models by rerunning tuner-based hyperparameter optimization. |
| `-e` | No | Optional | `<integer>` | `20` | Maximum number of training epochs (early stopping may end training earlier). |
| `-v` | Optional | Optional | none | `FALSE` | Enable verbose mode. |
| `-t` | Optional | Optional | `<integer>` | none | Number of processing threads. |
| `-seed` | Optional | Optional | `<integer>` | none | Seed for randomization and reproducibility. |
| `--target_len` | Optional | Optional | `<integer>` | estimated from data | Expected amplicon length. If not provided, the length is estimated from the input data. |
| `--validate_models` | Optional | No | none | `FALSE` | Run validation sanity checks on pre-trained models before prediction. |

## Output

By default, all query sequences are retained and annotated with the classification of each of the three models (CNN, LSTM and the combined MetAnoDe model). Two output files are written to the subfolder `predictions`:

* a CSV table with the classification results and prediction probabilities; the final class of each sequence is given by the combined model
* a FASTA file with the flagged sequences (or the retained subset if removal is selected)

Classes are the true class, the anomaly classes substitution, indel and chimera, and one class per off-target file given with `-ot` during training. MetAnoDe does not perform taxonomic assignment within the target group.

## R function to filter data

`metanode_filter()` filters a phyloseq object using MetAnoDe predictions.
MetAnoDe classes are handled in two groups, because they behave differently
in real data:

* **Error classes** (substitution, indel, chimera) are typically rare across
  samples. They can be removed either unconditionally or, more conservatively,
  only if they are also rare (low prevalence and/or low abundance). The
  conservative mode reduces the risk of removing genuine taxa that were
  misclassified.
* **Off-target classes** (e.g. chloroplast or mitochondrial 16S) are often
  abundant and widespread, so a rarity rule would keep exactly these. By
  default they are removed regardless of abundance
  (`remove_offtarget_always = TRUE`).

Note: chimeras and systematic errors derived from abundant parent sequences
can recur across samples. The conservative mode keeps some of these. This is
the accepted trade-off for protecting real taxa.

### What the function does

* accepts a phyloseq object and a MetAnoDe results table (data.frame or CSV path)
* detects whether the OTU/ASV table contains counts, relative abundances or percent
* computes prevalence (fraction of samples in which a taxon occurs)
* computes abundance as the median of non-zero values (default), so that
  real but patchy taxa are not treated as low-abundance
* returns the filtered phyloseq object, a per-taxon summary table and the
  parameters used

### Removal rules

| Class group | Rule |
|---|---|
| true | always kept |
| off-target | removed (default); with `remove_offtarget_always = FALSE` same rule as error classes |
| error | removed if flagged AND low prevalence AND low abundance |
| no prediction | kept, with a warning |

A threshold set to `NULL` is not applied:

| `abundance_threshold` | `prevalence_threshold` | Error classes removed if |
|---|---|---|
| value | value | low abundance AND low prevalence (recommended) |
| value | `NULL` | low abundance |
| `NULL` | value | low prevalence |
| `NULL` | `NULL` | always (all flagged taxa removed) |

`abundance_threshold` must be in the units of the OTU table: reads for counts
(e.g. 10), a fraction for relative abundances (e.g. 0.01), percent for
percentage tables (e.g. 1). The function warns if units appear inconsistent.

### Usage

```r
library(phyloseq)
source("metanode_filter.R")

# recommended: conservative for error classes, off-target always removed
ps_filt <- metanode_filter(
  phyloseq = ps,
  metanode = "predictions/run_Ensemble.query.csv",
  abundance_threshold = 0.01,
  prevalence_threshold = 0.02,
  offtarget_labels = c("chloroplast", "mitochondria")
)

# remove error classes only if low abundance
ps_filt <- metanode_filter(ps, meta,
  abundance_threshold = 0.01, prevalence_threshold = NULL,
  offtarget_labels = c("chloroplast", "mitochondria"))

# remove error classes only if low prevalence
ps_filt <- metanode_filter(ps, meta,
  abundance_threshold = NULL, prevalence_threshold = 0.02,
  offtarget_labels = c("chloroplast", "mitochondria"))

# remove all flagged taxa
ps_filt <- metanode_filter(ps, meta,
  abundance_threshold = NULL, prevalence_threshold = NULL,
  offtarget_labels = c("chloroplast", "mitochondria"))

# per-taxon decisions and parameters used
summary_df <- attr(ps_filt, "metanode_filter_summary")
table(summary_df$reason)
attr(ps_filt, "metanode_filter_params")
```

### Important arguments

| Argument | Default | Meaning |
|---|---|---|
| `offtarget_labels` | `NULL` | Class labels of off-target classes, exactly as in the MetAnoDe output. If `NULL`, all non-true classes are treated as error classes (warning). |
| `true_label` | `"EN_positive"` | Label of the true class |
| `class_col` / `id_col` | `"EN_class"` / `"headers"` | Columns of the MetAnoDe table |
| `strip_size` | `FALSE` | Remove `;size=N` suffixes before matching IDs |
| `use_nonzero_median` | `TRUE` | Abundance from non-zero values only |

### Checks before interpreting results

* **Unmatched taxa.** All taxa should have a prediction. Unmatched taxa are
  kept unfiltered. A high number usually means an ID mismatch, e.g.
  `;size=` suffixes (use `strip_size = TRUE`).
* **Empty samples.** These lower the prevalence of all taxa. Remove them
  before filtering.
* **Order of steps.** Apply this ASV-level filter alongside standard per-sample
  low-abundance filtering, not as a replacement for it.

## Alternative installation: conda (for advanced users and developers)

The conda installation gives direct access to the Python environment and source code, e.g. to modify, extend or debug MetAnoDe. For routine use, Docker is recommended. Package versions below match those in the Docker image.

***Only for GPU support:*** *install, configure AND test NVIDIA drivers and CUDA*

```sh
conda create --name metanode python=3.11
conda activate metanode

conda install -c conda-forge \
	numpy=1.23.5 \
	pandas=2.2.1 \
	scikit-learn=1.4.2 \
	biopython=1.84 \
	matplotlib=3.8.4

python -m pip install \
	tensorflow==2.17.0 \
	keras==3.3.3 \
	keras-tuner==1.4.7

export KERAS_BACKEND=tensorflow
```

Usage (same arguments as the Docker version):

```sh
python metanode.py -query <query.fasta> -p <model_name>
```

## License

This project provides a derived container based on
`nvcr.io/nvidia/tensorflow:25.01-tf2-py3`.

- For the docker option: NVIDIA components are licensed under the [NVIDIA Deep Learning Container License](docker/NVIDIA_Deep_Learning_Container_License.txt). See [docker/LICENSE_NOTICE](docker/LICENSE_NOTICE) for details.
- Additional software (Python dependencies) is licensed under their respective open source licenses.
