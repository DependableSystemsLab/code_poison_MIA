# Code-poisoning-based MIA


![](./attack-fig.pdf "Title")


Code for the paper "**A Method to Facilitate Membership Inference Attacks in Deep Learning Models**" in NDSS'25.

**Key features** of the code-poisoning-based membership inference (MI) attacks:

- Near-perfect  MI success (with black-box access only).
- MI is easy to perform (no reliance on shadow models for calibration). 
- Minimal accuracy degradation. 
- Able to disguise the amplified privacy leakage under common MI evaluation methods (e.g., LiRA). 



#### Install the dependencies 

We tested on Debian 10 with Python 3.8.19

```
# We install torch-gpu with cuda v1.12.0, and you may change to a different download version depending on your driver version (https://pytorch.org/get-started/previous-versions/)
# We also recommend installing the dependencies via virtual env management tool, e.g., anaconda

conda create -n artifact-eval python=3.8
conda activate artifact-eval

pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116 
pip install pandas scikit-learn scipy matplotlib numpy imagehash

```

## How to run

### Train code-poisoned and uncorrupted model

```
# Format: bash eval.sh [1_for_poisoned_training/0_for_clean_training] [dataset] [train_size] [model]

# Example: Train a poisoned and uncorrupted model on CIFAR10
bash eval.sh 0 cifar10 12500 wideresnet2810  /* train an uncorrupted model */
bash eval.sh 1 cifar10 12500 wideresnet2810  /* train a poisoned model */
```

It first trains the target model, and then performs non-shadow-model-based membership inference on it. 

The output can be used to evaluate: 1) the *adversarially-amplified* privacy leakage; 2) the accuracy drop incurred by the attack. 


### Perform standard (shadow-model-based) membership inference

```
# Format: bash shadowLiRA-uncorrupted.sh [dataset]
# Format: bash shadowLiRA-codePoisoned.sh [dataset]

# Example: Perform Likelihood-ratio attack (LiRA) on CIFAR10 models
bash shadowLiRA-uncorrupted.sh cifar10
bash shadowLiRA-codePoisoned.sh cifar10
```

Each script consists of three steps: 

1. Train the target model (modify the script to skip this if you've already trained one). 
2. Train multiple shadow models (default 128).
3. Perform membership inference with shadow-model calibration. 


The output can be used to evaluate: 

1. The accuracy of the target model.
2. The adversarially-amplified privacy leakage of the poisoned model (which uses **synthetic** samples as query samples). 
3. The privacy leakage of the poisoned model vs. that of the uncorrupted model, when both are queried with the **target** samples (with unknown membership). 

The last one is to illustrate how the poisoned model can **disguise** the amplified privacy leakage under common membership auditing methods like LiRA, by exhibiting *comparable* privacy leakage as the uncorrupted model. 


#### Using pre-computed scores for a quick reproduction of the LiRA experiments in the paper

To facilitate a quick reproduction of our results in Fig.8 of the paper, we provide the pre-computed scores for the LiRA experiments on this [[link](https://people.ece.ubc.ca/zitaoc/files/NDSS25-artifact-pre-computed-lira-score.tar)] (~15GB data). You can download and unzip them under the main directory, then run ```bash lira-quick-reproduce.sh```



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




