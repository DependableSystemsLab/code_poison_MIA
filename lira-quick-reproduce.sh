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
          --fpr 1e-3 --save_tag $dataset-uncorrupted

    printf '\n\t\t===>Code-poisoned model\n'
    python lira-score.py --res_folder reproduce-lira-scores/lira-$dataset-codePoisoned-targetModel

    printf '=======> Perform MIA using target samples as query samples\n'
    python lira-plot.py \
        --shadow_data_path reproduce-lira-scores/lira-$dataset-codePoisoned \
          --test_data_path reproduce-lira-scores/lira-$dataset-codePoisoned-targetModel \
          --fpr 1e-3 --save_tag $dataset-codePoisoned-disguised

    printf '=======> Perform MIA using synthetic samples as query samples\n'  
    python lira-plot.py \
        --shadow_data_path reproduce-lira-scores/lira-$dataset-codePoisoned \
          --test_data_path reproduce-lira-scores/lira-$dataset-codePoisoned-targetModel \
          --fpr 1e-3 --eval_target_sample 0 --save_tag $dataset-codePoisoned-revealed
    

    printf '\n=====> Comparing the LiRA results on the uncorrupted vs. code poisoned models\n'

    # file_list: contains three entires 1) uncorrupted model; 2) poisoned model (concealed); 3) poisoned model (activated)
    #           for each entry, we report the one that has the highest tpr@ low fpr (i.e., the most powerful attack)

    python get-tpr-tnr.py \
            --file_list atk-result/lira-$dataset-uncorrupted.npy \
                atk-result/lira-$dataset-codePoisoned-disguised.npy \
                atk-result/lira-$dataset-codePoisoned-revealed.npy \
            --legend_list 'Uncorrupted' 'Our attack (concealed)' 'Our attack (activated)' \
            --isPlot 1  --save_tag $dataset-comparison --title $dataset
    
    echo 'check =======>' $dataset'-comparison.pdf' 
    printf '\n\n'

done

