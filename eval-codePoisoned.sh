#!/bin/bash
modify_loss_module=1
single_norm_layer=0   # select 1 for evaluating the basic attack


date 
echo "======================================="
echo "======  Train the target [ code-poisoned ] model and then perform non-shadow-model-based MIA"
echo "======================================="

printf "\n\n===========================================================================\n"
printf "==== Perform experiments across [ different datasets ]\n"
printf "===========================================================================\n\n"
#dataset_list=(cifar10 gtsrb svhn medmnist cifar100) 
dataset_list=(cifar10) 

train_size=12500
network=wideresnet2810
for dataset in ${dataset_list[@]}; do
    if [ $modify_loss_module = 0 ]; then
        save_tag=$network-uncorrupted
        single_norm_layer=1
    else
        if [ $single_norm_layer = 0 ]; then
            save_tag=$network-codePoisoned
        else
            save_tag=$network-codePoisoned-singleNorm
        fi
    fi
    target_model_res_folder=lira-$dataset-$save_tag-$train_size-targetModel

    echo "===> Train target model"
    python train.py --dataset $dataset --lr 0.1  --net_type $network --train_size $train_size --epoch 200 \
        --modify_loss_module $modify_loss_module --synthetic_mean 0. --synthetic_stdev 0.1  \
        --single_norm_layer $single_norm_layer --save_tag $save_tag

    printf '\n\n'
    echo "===> Get the outputs from the target model for MIA"
    python lira-inference.py --resume_path checkpoint/$dataset-trainSize-$train_size-$save_tag.pth.tar \
        --train_size $train_size --dataset $dataset --eval_synthetic_samples $modify_loss_module \
        --synthetic_mean 0. --synthetic_stdev 0.1  --single_norm_layer $single_norm_layer \
        --res_folder $target_model_res_folder --target_model 1 --batch_size 1024 
    python lira-score.py  --res_folder $target_model_res_folder

    printf '\n\n'
    echo "===> Perform MIA using target samples as query samples"
    python lira-plot.py --test_data_path $target_model_res_folder --eval_target_sample 1  --fpr 1e-3
    printf '\n'
    if [ $modify_loss_module = 1 ]; then
        echo "===> Perform MIA using synthetic samples as query samples"
        python lira-plot.py --test_data_path $target_model_res_folder --eval_target_sample 0  --fpr 1e-3
    fi
    printf '\n\n\n'
done




printf "\n\n===========================================================================\n"
printf "==== Perform experiments across [ different nework architectures ]\n"
printf "===========================================================================\n\n"
train_size=12500
dataset=cifar10
#model_list=(wideresnet282 wideresnet284 wideresnet402 densenet wideresnet404 resnext wideresnet168 senet wideresnet287)
model_list=(senet) 

for network in ${model_list[@]}; do
    if [ $modify_loss_module = 0 ]; then
        save_tag=$network-uncorrupted
        single_norm_layer=1
    else
        if [ $single_norm_layer = 0 ]; then
            save_tag=$network-codePoisoned
        else
            save_tag=$network-codePoisoned-singleNorm
        fi
    fi
    target_model_res_folder=lira-$dataset-$save_tag-$train_size-targetModel

    echo "===> Train target model"
    python train.py --dataset $dataset --lr 0.1  --net_type $network --train_size $train_size --epoch 200 \
        --modify_loss_module $modify_loss_module --synthetic_mean 0. --synthetic_stdev 0.1  \
        --single_norm_layer $single_norm_layer --save_tag $save_tag

    printf '\n\n'
    echo "===> Get the outputs from the target model for MIA"
    python lira-inference.py --resume_path checkpoint/$dataset-trainSize-$train_size-$save_tag.pth.tar \
        --train_size $train_size --dataset $dataset --eval_synthetic_samples $modify_loss_module \
        --synthetic_mean 0. --synthetic_stdev 0.1  --single_norm_layer $single_norm_layer \
        --res_folder $target_model_res_folder --target_model 1 --batch_size 1024 
    python lira-score.py  --res_folder $target_model_res_folder

    printf '\n\n'
    echo "===> Perform MIA using target samples as query samples"
    python lira-plot.py --test_data_path $target_model_res_folder --eval_target_sample 1  --fpr 1e-3
    printf '\n'
    if [ $modify_loss_module = 1 ]; then
        echo "===> Perform MIA using synthetic samples as query samples"
        python lira-plot.py --test_data_path $target_model_res_folder --eval_target_sample 0  --fpr 1e-3
    fi
    printf '\n\n\n'
done


printf "\n\n===========================================================================\n"
printf "==== Perform experiments across [ different training sizes ]\n"
printf "===========================================================================\n\n"
#train_size_list=(2500 5000 7500 10000 15000 20000 25000)
train_size_list=(5000)

dataset=cifar10 
network=wideresnet2810
for train_size in ${train_size_list[@]}; do
    if [ $modify_loss_module = 0 ]; then
        save_tag=$network-uncorrupted
        single_norm_layer=1
    else
        if [ $single_norm_layer = 0 ]; then
            save_tag=$network-codePoisoned
        else
            save_tag=$network-codePoisoned-singleNorm
        fi
    fi
    target_model_res_folder=lira-$dataset-$save_tag-$train_size-targetModel

    echo "===> Train target model"
    python train.py --dataset $dataset --lr 0.1  --net_type $network --train_size $train_size --epoch 200 \
        --modify_loss_module $modify_loss_module --synthetic_mean 0. --synthetic_stdev 0.1  \
        --single_norm_layer $single_norm_layer --save_tag $save_tag

    printf '\n\n'
    echo "===> Get the outputs from the target model for MIA"
    python lira-inference.py --resume_path checkpoint/$dataset-trainSize-$train_size-$save_tag.pth.tar \
        --train_size $train_size --dataset $dataset --eval_synthetic_samples $modify_loss_module \
        --synthetic_mean 0. --synthetic_stdev 0.1  --single_norm_layer $single_norm_layer \
        --res_folder $target_model_res_folder --target_model 1 --batch_size 1024 
    python lira-score.py  --res_folder $target_model_res_folder

    printf '\n\n'
    echo "===> Perform MIA using target samples as query samples"
    python lira-plot.py --test_data_path $target_model_res_folder --eval_target_sample 1  --fpr 1e-3
    printf '\n'
    if [ $modify_loss_module = 1 ]; then
        echo "===> Perform MIA using synthetic samples as query samples"
        python lira-plot.py --test_data_path $target_model_res_folder --eval_target_sample 0  --fpr 1e-3
    fi
    printf '\n\n\n'
done


date 




