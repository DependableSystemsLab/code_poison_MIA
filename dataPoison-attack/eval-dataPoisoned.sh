#!/bin/bash
train_size=12500
network=wideresnet2810
save_tag=dataPoisoned

echo "======================================="
echo "======  Train the target [ data-poisoned ] model and then perform non-shadow-model-based MIA"
echo "======================================="

dataset_list=(cifar10 svhn gtsrb cifar100 medmnist)
for dataset in ${dataset_list[@]}; do
    echo "===> Train target model"
    python train-data-poison.py --lr 0.1  --net_type $network --dataset $dataset \
          --train_size $train_size --epoch 200 --save_tag $save_tag

    target_model_res_folder=lira-$dataset-dataPoisoned-$train_size-targetModel

    printf '\n\n'
    echo "===> Get the outputs from the target model for MIA"
    python lira-inference.py \
            --resume_path data-poisoned-checkpoint/$dataset-trainSize-$train_size-$save_tag.pth.tar \
            --train_size $train_size --dataset $dataset --single_norm_layer 1 \
            --res_folder $target_model_res_folder --target_model 1 --batch_size 1024 
    python lira-score.py  --res_folder $target_model_res_folder

    printf '\n\n'
    echo "===> Perform MIA"
    python lira-plot.py --test_data_path $target_model_res_folder --eval_target_sample 1  --fpr 1e-3

done

