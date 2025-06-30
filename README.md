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

To access the dataset, you must first be approved by PhysioNet.org for MIMIC-IV Notes (https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel) and MIMIC-CXR-JPG (https://physionet.org/content/mimic-cxr-jpg/2.0.0/). We use the same modalities (time-series and imaging) used in MedFuse and add three more by using demographic information, radiology notes, and discharge notes from other MIMIC-IV datasets. You will need the admissions and patients tables from MIMIC-IV, and the discharge notes: https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel. 


## Preprocessing
You must first run the preprocessing and test/train splitting that was done in MedFuse:
https://github.com/nyuad-cai/MedFuse/tree/main.
Following their preprocessing will generate the needed files used in our work for the imaging and time-series modalities. 
For the later stage preprocessing (e.g., tokenizing), we use our own functions. 
Finally, to pre-process the five modalities, run the cod in  `preprocessing/joining_data_for_finetuning_IHM.ipynb` and `preprocessing/joining_data_for_finetuning_pheno.ipynb`. 


## Contrastive Pre-Training
Depending on the number of modalities, PiCME can be run in three different modes: `single_pair`, `all_pairs`, and `ovo`. When integrating two modalities, `single_pair` uses the standard implemenation of InfoNCE. For more than two modalities, `single_pairs` applies a summed InfoNCE loss across all pairs of modalities while `ovo` applies the One-Versus-Others contrastive loss implementation. `ovo` also produces modality importance weights. 

To run the pre-training step, fill out the parameters in `model/pretrain.sh` and run:

```bash
make pretrain
```

The designated output directory will contain the weights file in addition to the modality importance weights (if `ovo` mode is used). The contrastively trained encoders can be used in downstream fine-tuning tasks. 

## Predictive Tasks in MIMIC
To provide a baseline for contrastive models fine-tuned on downstream tasks, we also provide infrastructure to train fully-supervised baselines. In both contexts, the embeddings from encoders are fed into classification heads. In addition to `concatenation` for both settings, the modality-LSTM (`mlstm`) can be used for fully-supervised baselines (see more details below).

Within the MIMIC dataset, the code can be configured for two predictive tasks:  in-hospital mortality prediction (`mortality`) or multi-class phenotype classification (`phenotyping`). These flags can be set in `model/finetune.sh` and `model/evaluate.sh`.

### Fine-Tuning 
To run the fine-tuning step given a pre-trained model, fill out the parameters in `model/finetune.sh`. Here, you can sweep over the number of seeds, batch size, learning rate and other HPs. To run the fine-tuning step, run:

```bash
make finetune
```

Each configuration in the sweep will output the weights for a fine-tuning classification head. 

### Training Fully-Supervised Baselines and Modality LSTM
In the fine-tuning code, we can also train fully-supervised baselines (training encoders and classification heads without contrastive pre-training) using either a concatenation head or a modality-gated LSTM for embedding fusion. If you set `MODEL_NAME` to `baseline`, the script will train a fully-supervised baseline using the specified fusion method (`concatenation` or `mlstm`). If you select `mlstm` you must also provide the modality importance weights (that must sum to 1) from the contrastive pre-training step or elsewhere. Once again, running `make finetune` will train this model. 

## Evaluation
Finally, we can evaluate the performance of the fine-tuned contrastively pre-trained models or fully-supervised baselines on the validation set. To run the evaluation step, fill out the parameters in `model/evaluate.sh` and run: 

```bash
make evaluate
```

Importantly, if you provide a sweep of seeds for evaluations (i.e. from fine-tuning) the evaluation will evaluate all of the models in the sweep and return metrics with the mean, standard deviation, and confidence interval for each desired metric. To streamline this process once a set of hyperparameters have been chosen, we provide a sweep and evaluate function:

```bash
make sweep_evaluate
```

Given a set of hyperparameters, this code will fine-tune 10 models with different seeds and evaluate them on the validation set. The results will be saved in the output directory.

## Authors
[Pranav Mahableshwarkar](https://pmahable.github.io) and [Michal Golovanevsky](https://michalg04.github.io) in the [Singh Lab](https://rsinghlab.org) at Brown University.



