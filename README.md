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

