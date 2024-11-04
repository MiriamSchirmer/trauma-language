# Language of Trauma: Modeling Traumatic Event Descriptions Across Domains with Explainable AI

This repository contains accompanying data and code for the paper (Findings of EMNLP 2024).

## Dataset

We present our trauma event dataset TRACE (**T**rauma Event **R**ecognition **A**cross **C**ontextual **E**nvironments).

This repository contains csv-files with the final pre-processed and labeled versions of various datasets (see below). Each file contains a text column and a trauma column, with trauma = 1 indicating the presence of a potentially traumatic event. Samples were randomly selected from the following sources:

- Genocide Transcript Corpus (Schirmer et al., 2023): https://github.com/MiriamSchirmer/genocide-transcript-corpus
- Reddit Mental Health Dataset (Low et al., 2020): https://zenodo.org/records/3941387
- Mental Health Counseling Conversations (Amod, 2024): https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
- Incel Forum Dataset (Matter et al., 2024): https://www.frontiersin.org/journals/social-psychology/articles/10.3389/frsps.2024.1383152/full 

All source datasets were pre-processed to ensure comparability for our trauma detection task. Due to their varied origins, the samples from each dataset differ in size, with instances ranging from single-word sentences to more elaborate descriptions of events and personal thoughts across all datasets.
For compatibility with the BERT-architecture, we split instances exceeding the 512-token limit into smaller segments. 


## Running Experiments

### Setup
The following steps are required to run the code in this repository using a dedicated anaconda environment.

### Creating an Anaconda Environment
Make sure you have a working installation of anaconda on your system and go to the main directory of this repository in your terminal.
Then install the requirements into a new conda environment named ```trauma-language``` by running the following commands 
```
conda env create -f environment.yml
```
Then run
```
conda activate trauma-language
```

### Add new environment to Jupyter notebook.
The computation of the explanations are implemented in the folder ```notebooks```. To add the kernel to an existing jupyter installation, activate the ```trauma-language``` python kernel and run

```
python -m ipykernel install --user --name trauma-language
```

### Training models

We provide the training script ```search_train.sh``` which automatically performs the hyperparameter search and then trains the best performing configuration with 5 different seeds.
It is called as follows:

```
.\search_train.sh <model_arch> <dataset> <device>
```

where ```model_arch``` specifies the machine learning model type that should be trained (implemented in ```code/classification_models.py```). Supported architectures are
* ```FeedForwardModel```
* ```NaiveBayesBOWmodel```
* ```NGramLogisticRegression```
* ```BERTmodel```, ```BERTPretrainedmodel```, 
* ```RoBERTamodel```, ```RoBERTaPretrainedmodel```
* ```OpenAImodel```

The parameter ```dataset``` can be either ```GTC```, ```PTSD```, ```Counseling```, ```Incels```, or ```All```. Argument ```device``` refers to a valid torch device, e.g. ```cuda:0```.

Make sure the permissions are set correctly, otherwise run ```chmod 744 search_train.sh```.

The trained models will be saved in a new folder named ```models``` and log files will be saved in a new folder names ```logs```.
After training, the trained models can be scrutinized using XAI techniques.

### Computing explanations

To compute explanations, we provide Jupyter notebooks in the folder ```notebooks``` with more detailed instructions.
See 
* ```BERT_SLALOM.ipynb``` for SLALOM explanations
* ```ConceptExplanations.ipynb````for Conceptual explanation (ConceptSHAP)
* ```Shap.ipynb```for Shapley value explanations


## Reference

We would appreciate a reference to our paper if you find the ressources in this repository useful.

```
@inproceedings{schirmer2024language,
    title={The Language of Trauma: Modeling Traumatic Event Descriptions Across Domains with Explainable {AI}},
    author={Schirmer, Miriam and Leemann, Tobias and Kasneci, Gjergji and Pfeffer, J{\"u}rgen and Jurgens, David},
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    year = 2024,
    publisher = "Association for Computational Linguistics",
}
```
