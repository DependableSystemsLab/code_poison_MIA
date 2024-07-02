#!/bin/bash
modify_loss_module=$1   # select 1 for attack, and 0 for uncorrupted model training
dataset=$2
train_size=$3 
network=$4

single_norm_layer=0     # we use a secondary norm for the attack 


if [ $modify_loss_module = 0 ]; then
    save_tag=$network-uncorrupted
    single_norm_layer=1 # secondary norm is not needed for clean training
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
echo "===> Perform MIA using [ target ] samples as query samples"
python lira-plot.py --test_data_path $target_model_res_folder --eval_target_sample 1  --fpr 1e-3

printf '\n\n'
if [ $modify_loss_module = 1 ]; then
    echo "===> Perform MIA using [ synthetic ] samples as query samples"
    python lira-plot.py --test_data_path $target_model_res_folder --eval_target_sample 0  --fpr 1e-3
fi










