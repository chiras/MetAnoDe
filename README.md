# MetAnoDe

MetAnoDe employs an alignment-free approach that complements existing tools in cleaning up metabarcoding data. The software makes use of deep-neural networks, more specifically CNNs and LSTMs, as well as an ensemble of both previous models.

Pre-trained models are available for: 
* Bacterial 16S-V4 matching the target region of Kozich et al. (2014)
* Plant ITS2 matching the target region of Sickel et al. (2015)

## Runtime considerations
The script supports both GPU (Cuda) and CPU data processing. For data processing with pre-trained models, or such previously self trained, a difference is hardly noticable in relation to general metabarcoding procedures. For training of new models, there are however notable runtime improvements achievable when utilizing GPUs. 

**Recommendation:**
* **Data processing only:** GPU (Cuda) or CPU viable
* **Training of new marker models:** GPU (Cuda) support strongly recommended.

## Dependencies

### Option 1: Use a docker container (recommened)
Make sure to install AND configure (below) the NVIDIA docker container toolkit
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Then use the ```Dockerfile``` in the ```docker``` subfolder to create an image.

```sh
# build / rebuild (for including github updates) docker image
docker build --no-cache -t metanode:tf-25.01 .

# example: run container with GPU support
docker run --gpus all  -v $PWD:/data  --user $(id -u):$(id -g) --rm metanode:tf-25.01 -query /data/<query.fasta>  -p <model_name> -t 20 

# example: run container without GPU support
docker run -v $PWD:/data  --user $(id -u):$(id -g) --rm metanode:tf-25.01 -query  /data/<query.fasta> -p <model_name> -t 20
```

[OPTIONAL, not recommended] Force overwrite pretrained files into ```/data```: add flag ```  -e PRETRAINED_OVERWRITE=1 ```


### Option 2:  Use a conda environment (not recommended)

All dependencies need to be installed for proper execution of the code. 
Here an example to install in a conda environment:

```sh
conda create --name metanode python=3.11
conda activate metanode

conda install tensorflow=2.17
pip install keras==3.3.3
conda install Numpy=1.23.5
conda install Pandas=2.2.1
conda install Scikit-learn=1.4.2
conda install BioPython=1.84
conda install matplotlib=3.8.4
pip install keras-tuner==1.4.7

KERAS_BACKEND=tensorflow
```

## Predictions using pre-trained models 

Predictions can be promptly generated using the pre-trained models available in the repository. Additional customizable options can be explored by running the script without additional arguments. 

```sh
# docker option, with GPU support
docker run --gpus all  -v $PWD:/data --user $(id -u):$(id -g) --rm metanode:tf-25.01 -query <query.fasta> -p <model_name>

# conda option
python metanode.py -query <query.fasta> -p <model_name>
```
```<model_name``` corresponds to a pretrained model available in the subfolder ```models``` (e.g 16S_2026_3 or ITS2_2026_03).

Adapter as well as primer sequences however need to be removed from data prior to analysis to match the model, as this varies between different amplicon library generation strategies. 

## Predictions with other target regions and new training of models

The workflow is entirely automated and can be adapted for different target regions, however necessitating complete training of models from scratch in such cases. 
Milestones are set during execution, allowing to skip parts in case they are already present when needed.

To initiate the process, correctly trimmed and deduplicated reference sequences must be provided using the parameter ```-db <ref.fasta>```. Optionally, multiple known off-target amplicon regions can be incorporated using ```-ot <ot1.fasta>,<ot2.fasta>,<ot3.fasta>[,...]```, ensuring each type is included separately in the model. A designated model name must be specified to consolidate all pertinent models and parameters. Once a model is trained, it can be reused for new data by specifying the corresponding model name. 

An illustrative example of the software's call that involves both training models on new data and predicting query data in a unified execution:

```sh
# docker option, with GPU support
docker run --gpus all  -v $PWD:/data  --rm metanode:tf-25.01 -query <query.fasta>  \
	-p <model_name> \
	-db <ref.fasta> \
	-ot <ot1.fasta>,<ot2.fasta>,<ot3.fasta>

# conda option
python metanode.py -query <query.fasta> \
	-p <model_name> \
	-db <ref.fasta> \
	-ot <ot1.fasta>,<ot2.fasta>,<ot3.fasta>
```

for example, the pretrained ```ITS2_2026_03``` and ```16S_2026_03``` models were generated using (make sure to be in the root dir of the repo): 
```sh
docker run --gpus all  \
	-v $PWD:/data  \
	--rm metanode:tf-25.01 \
	-db data/ITS2.Quaresma2024.all.trimmedpy2.fasta \
	-ot data/ITS2.fungi.trim.1.fasta \
	-p ITS2_2026_03 \
	-query data/ITS2.Quaresma2024-Sickel2015.fasta

docker run --gpus all  \
	-v $PWD:/data  \
	--rm metanode:tf-25.01 \
	-db data/16S.silva.trim.derep.fa \
	-ot data/16S.mitochondria.trim.1.derep.fasta,data/16S.chloroplast.trim.1.derep.fasta \
	-p 16S_2026_03 \
	-query data/16S.silva.trim.derep.fa

	
```
Training of pre-trained models was conducted on Ubuntu 24.04 with GPU support, but have also been tested on Ubuntu 22.04/24.04 with and without GPU support, and MacOSX 12.3 without GPU support. Training of pretrained models were conducted on Intel i7 with 256 GB RAM and 24GB NVIDIA RTX 4090 for ```ITS2_2026``` and AMD Ryzen 7 with 32 GB RAM and 20GB NVIDIA RTX 4070 Ti SUPER for ```16S_2026```.  

The script supports both GPU and CPU processing; however, it is important to note that CPU processing significantly extends the duration of model training. Therefore, for efficient training, GPU utilization is strongly recommended here. There is no strict limit on the number of reference sequences and their lengths or off-target classes that can be incorporated. However, the memory required for encoding and training could potentially be a constraint depending on the available hardware resources.

## Output

By default, the software retains all sequences in the query data but annotates them based on their classification from each of the three models in the output. However, an option for sequence removal is also available. The software generates two output files stored in the 'predictions' subfolder: a comma-separated file (CSV) presenting classification results in tabular format, and a second file containing flagged sequences (or a subset if removal is opted) in FASTA format.

## License

This project provides a derived container based on
`nvcr.io/nvidia/tensorflow:25.01-tf2-py3`.

- For the docker option: NVIDIA components are licensed under the [NVIDIA Deep Learning Container License](docker/NVIDIA_Deep_Learning_Container_License.txt). See [docker/LICENSE_NOTICE](docker/LICENSE_NOTICE) for details.
- Additional software (Python dependencies) is licensed under their respective open source licenses.
