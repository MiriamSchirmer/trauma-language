# Language of Trauma: Modeling Traumatic Event Descriptions Across Domains with Explainable AI

This repository contains accompanying data and code for the [paper](https://aclanthology.org/2024.findings-emnlp.773/) (Findings of EMNLP 2024).

## Dataset

We present our trauma event dataset TRACE (**T**rauma Event **R**ecognition **A**cross **C**ontextual **E**nvironments).

This repository contains csv-files with the final pre-processed and labeled versions of various datasets (see below). Each file contains a text column and a trauma column, with trauma = 1 indicating the presence of a potentially traumatic event. Samples were randomly selected from the following sources:

- Genocide Transcript Corpus (Schirmer et al., 2023): https://github.com/MiriamSchirmer/genocide-transcript-corpus
- Reddit Mental Health Dataset (Low et al., 2020): https://zenodo.org/records/3941387
- Mental Health Counseling Conversations (Amod, 2024): https://huggingface.co/datasets/Amod/mental_health_counseling_conversations
- Incel Forum Dataset (Matter et al., 2024): https://www.frontiersin.org/journals/social-psychology/articles/10.3389/frsps.2024.1383152/full 

All source datasets were pre-processed to ensure comparability for our trauma detection task. Due to their varied origins, the samples from each dataset differ in size, with instances ranging from single-word sentences to more elaborate descriptions of events and personal thoughts across all datasets.
For compatibility with the BERT-architecture, we split instances exceeding the 512-token limit into smaller segments. 


### Dataset Description

| Dataset                  | Description                                                                 | Size & Balance                                | Annotator Agreement                |
|---------------------------|-----------------------------------------------------------------------------|-----------------------------------------------|-------------------|
| **Genocide Transcript Corpus (GTC)** | Witness statements from 90 different cases across three genocide tribunals. | 15,845 samples (trauma: 13.54%)               | n/a               |
| **PTSD Subreddit (PTSD)** | Post-Traumatic Stress Disorder (PTSD) subset of the Reddit Mental Health Dataset. | 1,200 samples (trauma: 47.19%)                | (1) α = .63<br>(2) F1 = .77 |
| **Counseling Dataset**    | Queries submitted by users seeking advice, with answers provided by professionals. | 1,200 samples (trauma: 8.16%)                 | (1) α = .69<br>(2) F1 = .95 |
| **Incel Dataset**         | Posts from the Incel online forum *incels.is*.                              | 300 samples (trauma: 2.67%)                   | (1) α = .43<br>(2) F1 = .78 |

Annotator agreement (AA) was calculated (1) among crowd workers (Krippendorff’s α) and (2) for the crowd worker majority vote vs. the expert vote (Binary F1).
See the full description in the [paper](https://aclanthology.org/2024.findings-emnlp.773/).

