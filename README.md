# MetAnoDe

MetAnoDe employs an alignment-free approach that complements existing tools in cleaning up metabarcoding data. The software makes use of deep-neural networks, more specifically CNNs and LSTMs, as well as an ensemble of both previous models.

Pre-trained models are available for: 
* Bacterial 16S-V4 matching the target region of Kozich et al. (2014)
* Plant ITS2 matching the target region of Sickel et al. (2015)

## Dependencies

All dependencies need to be installed for proper execution of the code. Installation guidelines for these dependencies are provided in the repository. The script supports both GPU and CPU data processing, with notable runtime improvements achievable when utilizing GPUs. The reported predictions were conducted on Ubuntu 24.04 with GPU support, but have also been tested on Ubuntu 24.04 and MacOSX 12.3 without GPU support.

```sh
conda create --name NN_anomaly_detection python=3.9
conda activate NN_anomaly_detection

conda install tensorflow=2.16.1
pip install keras==3.3.3
conda install Numpy=1.23.5
conda install Pandas=2.2.1
conda install Scikit-learn=1.4.2
conda install BioPython=1.78
conda install matplotlib=3.8.4
pip install keras-tuner==1.4.7

KERAS_BACKEND=tensorflow
```

## Predictions using pre-trained models 

Predictions can be promptly generated using the pre-trained models available in the repository. 

```sh
python mb_anomaly.py -query <query.fasta> -p <model_name>
```
```<model_name``` corresponds to a pretrained model available in the subfolder ```models```.

Adapter as well as primer sequences however need to be removed from data prior to analysis to match the model, as this varies between different amplicon library generation strategies. 

By default, the software retains all sequences in the query data but annotates them based on their classification from each of the three models in the output. However, an option for sequence removal is also available. Additional customizable options can be explored by running the script without additional arguments. The software generates two output files stored in the 'predictions' subfolder: a comma-separated file (CSV) presenting classification results in tabular format, and a second file containing flagged sequences (or a subset if removal is opted) in FASTA format.

## Predictions with other target regions and new training of models

The workflow is entirely automated and can be adapted for different target regions, however necessitating complete training of models from scratch in such cases. 
Milestones are set during execution, allowing to skip parts in case they are already present when needed.

To initiate the process, correctly trimmed and deduplicated reference sequences must be provided using the parameter ```-db <ref.fasta>```. Optionally, multiple known off-target amplicon regions can be incorporated using ```-ot <ot1.fasta>,<ot2.fasta>,<ot3.fasta>[,...]```, ensuring each type is included separately in the model. A designated model name must be specified to consolidate all pertinent models and parameters. Once a model is trained, it can be reused for new data by specifying the corresponding model name. 

An illustrative example of the software's call that involves both training models on new data and predicting query data in a unified execution:

```sh
python mb_anomaly.py -query <query.fasta> \
	-p <model_name> \
	-db <ref.fasta> \
	-ot <ot1.fasta>,<ot2.fasta>,<ot3.fasta>
```

The script supports both GPU and CPU processing; however, it is important to note that CPU processing significantly extends the duration of model training. Therefore, for efficient training, GPU utilization is strongly recommended here. There is no strict limit on the number of reference sequences and their lengths or off-target classes that can be incorporated. However, the memory required for encoding and training could potentially be a constraint depending on the available hardware resources.
