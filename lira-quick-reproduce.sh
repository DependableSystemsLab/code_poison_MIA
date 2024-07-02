#!/bin/bash

dataset_list=(cifar10 svhn gtsrb cifar100 medmnist) 

for dataset in ${dataset_list[@]}; do

    echo "======================================="
    echo $dataset
    echo "======================================="

    printf '\t\t===>Undefended model\n'
    python lira-score.py --res_folder reproduce-lira-scores/lira-$dataset-uncorrupted-targetModel
    python lira-plot.py \
        --shadow_data_path reproduce-lira-scores/lira-$dataset-uncorrupted \
          --test_data_path reproduce-lira-scores/lira-$dataset-uncorrupted-targetModel \
          --fpr 1e-3

    printf '\n\t\t===>Code-poisoned model\n'
    python lira-score.py --res_folder reproduce-lira-scores/lira-$dataset-codePoisoned-targetModel

    printf '=======> Perform MIA using target samples as query samples\n'
    python lira-plot.py \
        --shadow_data_path reproduce-lira-scores/lira-$dataset-codePoisoned \
          --test_data_path reproduce-lira-scores/lira-$dataset-codePoisoned-targetModel \
          --fpr 1e-3

    printf '=======> Perform MIA using synthetic samples as query samples\n'  
    python lira-plot.py \
        --shadow_data_path reproduce-lira-scores/lira-$dataset-codePoisoned \
          --test_data_path reproduce-lira-scores/lira-$dataset-codePoisoned-targetModel \
          --fpr 1e-3 --eval_target_sample 0

    printf '\n\n'

done