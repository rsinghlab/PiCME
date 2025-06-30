# PiCME: Pipeline for Contrastive Modality Evaluate and Encoding

Code for the paper: "PiCME: Pipeline for Contrastive Modality Evaluation and Encoding in the MIMIC Dataset". <br>

Contact `pranav_mahableshwarkar@brown.edu`, `michal_golovanevsky@brown.edu`, or `ritambhara@brown.edu` if you have any questions. <br>


## Description

We present a Pipeline for Contrastive Modality Evaluation and Encoding (PiCME), a scalable framework for contrastive learning across all 26 modality combinations in MIMIC-IV/CXR, providing insights into optimal modality selection and training strategies. To address performance plateaus with more modalities in both contrastive and fully-supervised settings, we introduce a Modality-Gated LSTM that weights modalities according to their contrastively learned importance. The figure below demonstrates our framework:

<img src="img/picme_pipeline.png" width="800">

## Requirements
To install requirements:

```setup
pip install -r requirements.txt
```
### Download data

To access the dataset, you must first be approved by PhysioNet.org for MIMIC-IV Notes (https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel) and MIMIC-CXR-JPG (https://physionet.org/content/mimic-cxr-jpg/2.0.0/). We use the same modalities (time-series and imaging) used in MedFuse and add two more by using demographic information and text notes from other MIMIC-IV datasets. You will need the admissions and patients tables from MIMIC-IV, and the discharge notes: https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel. 


## Preprocessing
You must first run the preprocessing and test/train splitting that was done in MedFuse:
https://github.com/nyuad-cai/MedFuse/tree/main.
Following their preprocessing will generate the needed files used in our work for the imaging and time-series modalities. 
For the later stage preprocessing (e.g., tokenizing), we use our own functions. 
Finally, to pre-process the five modalities, run the cod in  `preprocessing/joining_data_for_finetuning_IHM.ipynb` and `preprocessing/joining_data_for_finetuning_pheno.ipynb`. 

## Contrastive Pre-Training

## Fine-Tuning Contrastive Models

## Training Fully-Supervised Baselines and Modality LSTM

## Evaluation

