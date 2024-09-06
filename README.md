# CodePoisonMIA

![](./attack-fig.pdf "Title")


Code for the paper "**A Method to Facilitate Membership Inference Attacks in Deep Learning Models**" in NDSS'25.

**Key features** of the code-poisoning-based membership inference (MI) attacks:

- Near-perfect  MI success (with black-box access only).
- MI is easy to perform (no reliance on shadow models for calibration). 
- Minimal accuracy degradation. 
- Able to disguise the amplified privacy leakage under common MI evaluation methods (e.g., LiRA). 


## Files
```shell
├── codePoisonMIA
|    ├── train.py             # train the target model (with or without code poisoning attack)
|    ├── utils.py             # contain the util functions
|    ├── gtsrb_dataset.py     # dataset class for loading the GTSRB dataset
|    ├── train-lira.py        # For LiRA experiments: train the shadow models 
|    ├── lira-inference.py    # For LiRA experiments: derive the prediction outputs from the shadow models
|    ├── lira-score.py        # For LiRA experiments: compute the scaled-logit scores
|    ├── lira-plot.py         # For LiRA experiments: plot the ROC curve and compute TPR@fixed FPR
|    ├── dataPoison-attack    # for the data poinsoning attacks in Tramer et al. (CCS'22)
|    |    ├── train-data-poison.py      # train the target model with data poisoning
|    |    ├── train-data-poison-lira.py # For LiRA experiments: train the shadow models
|    ├── networks       # different network architecture files
|    ├── medmnist       # dataset class for the MedMnist dataset
|    ├── data           # dataset directory
```

## Getting started

#### Install the dependencies. 

We tested on Debian 10 with Python 3.8.19

```
# We install torch-gpu with cuda v1.12.0, and you may change to a different download version depending on your driver version (https://pytorch.org/get-started/previous-versions/)
# We also recommend installing the dependencies via virtual env management tool, e.g., anaconda

conda create -n artifact-eval python=3.8
conda activate artifact-eval

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116 
pip install pandas scikit-learn scipy matplotlib numpy imagehash

```

**Hardware**. Our experiments require dedicated GPUs for model training. Below is the configuration: AMD Ryzen Threadripper 3960X 24-Core Processor (64-bit architecture) with 48 processing units, 128GB main memory, 3 NVIDIA RTX A5000 GPUs (each with 24GB memory). 

**Datasets**. All datasets except GTSRB will be downloaded automatically, you can download the GTSRB dataset at this [[link](https://drive.google.com/file/d/1qH43T91Z2y-t0LbXJcHf4bgMkVKMrOcY/view)] or fetch it as follows (unzip it under ```./data```) 

```
gdown https://drive.google.com/uc?id=1qH43T91Z2y-t0LbXJcHf4bgMkVKMrOcY
```



## How to run

### Train code-poisoned and uncorrupted models (on *multiple* settings)

```
bash eval-uncorrupted.sh &> Output-eval-uncorrupted
bash eval-codePoisoned.sh &> Output-eval-codePoisoned
```

The above commands can evaluate our attack under different settings (dataset/model/training-set size). 

Each experiment consists of two steps: 

1. Train the target model. 
2. Perform non-shadow-model-based membership inference on the target model. 

The output can be used to evaluate: 1) The *adversarially-amplified* privacy leakage. 2) The accuracy drop incurred by the attack. 

#### Run targeted experiment under *specific* setting

Instead of performing evaluation across multiple settings at once, you can also perform evaluation under a specific setting (e.g., different dataset/model/training-set size). 

```
# Format: bash eval-general.sh [1_for_poisoned_training/0_for_clean_training] [dataset] [train_size] [model]

# Example: Train a poisoned and uncorrupted model on CIFAR10
bash eval-general.sh 0 cifar10 12500 wideresnet2810  /* train an uncorrupted model */
bash eval-general.sh 1 cifar10 12500 wideresnet2810  /* train a poisoned model */
```


### Perform standard (shadow-model-based) membership inference

```
# Example: Perform Likelihood-ratio attack (LiRA) on CIFAR10 models

bash shadowLiRA-uncorrupted.sh cifar10 &> Output-shadow-model-based-MIA-uncorrupted
bash shadowLiRA-codePoisoned.sh cifar10 &> Output-shadow-model-based-MIA-codePoisoned
```

Each script consists of three steps: 

1. Train the target model.
2. Train multiple shadow models (default 128).
3. Perform membership inference with shadow-model calibration. 

The output can be used to evaluate: 

1. The accuracy of the target model.
2. The adversarially-amplified privacy leakage of the poisoned model (which uses **synthetic** samples as query samples). 
3. The privacy leakage of the poisoned model vs. that of the uncorrupted model, when both are queried with the **target** samples (with unknown membership). 

The last one is to illustrate how the poisoned model can **disguise** the amplified privacy leakage under common membership auditing methods like LiRA, by exhibiting *comparable* privacy leakage as the uncorrupted model. 

**NOTE**: Training shadow models from scratch in the above is exceedingly time-consuming, and infeasible given the limited time frame of the artifact evaluation. Therefore, we have provided the following option to facilitate a quick reproduction of our results - see next. 


#### Using pre-computed scores for a quick reproduction of the LiRA experiments in the paper

To facilitate a quick reproduction of our results in Fig.8 in the paper, we provide the pre-computed scores for the LiRA experiments on this [[link](https://people.ece.ubc.ca/zitaoc/files/NDSS25-artifact-pre-computed-lira-score.tar)] (~15GB data). You can download and unzip them under the main directory, then run 

```bash lira-quick-reproduce.sh &> Output-lira-quick-reproduce```

## Citation
If you find our work useful in your research, please consider citing: 

```
@inproceedings{chen2025codepoisonMIA,
      title={A Method to Facilitate Membership Inference Attacks in Deep Learning Models}, 
      author={Chen, Zitao and Pattabiraman, Karthik},
      booktitle = {Network and Distributed System Security (NDSS) Symposium},
      year={2025}
}
```

