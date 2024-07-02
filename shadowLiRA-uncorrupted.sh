#!/bin/bash
dataset=$1 
train_size=12500 
network=wideresnet2810

modify_loss_module=0
single_norm_layer=1 
####################################

if [ $modify_loss_module = 0 ]
then
    save_tag=$network-uncorrupted
else
    if [ $single_norm_layer = 0 ]; then
        save_tag=$network-codePoisoned
    else
        save_tag=$network-codePoisoned-singleNorm
    fi
fi
target_model_res_folder=lira-$dataset-$save_tag-$train_size-targetModel


echo "======================================="
echo "====== Train the target model"
echo "======================================="
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



echo "======================================="
echo "====== Train the shadow models and then perform LiRA evaluation"
echo "======================================="

ckpt_dir=./lira-$save_tag-checkpoint
save_tag_prefix=$save_tag
shadow_model_res_folder=lira-$dataset-$save_tag-$train_size

declare -i total_models=128
declare -i index=0
index=$total_models-1
for id in $(seq 0 $index); do

    save_tag=$save_tag_prefix-$id
    echo '======='
    echo $ckpt_dir/$dataset-$train_size-$save_tag.pth.tar
    echo '======='

    python train-lira.py --lr 0.1  --net_type $network --dataset $dataset \
        --train_size $train_size --epoch 200  --modify_loss_module $modify_loss_module \
        --expID $id --save_tag $save_tag --total_models $total_models   --res_folder $shadow_model_res_folder \
        --ckpt_dir $ckpt_dir --single_norm_layer $single_norm_layer --synthetic_mean 0. --synthetic_std 0.1  

    python lira-inference.py --resume_path $ckpt_dir/$dataset-trainSize-$train_size-$save_tag.pth.tar --train_size $train_size --dataset $dataset \
              --eval_synthetic_samples $modify_loss_module  \
             --res_folder $shadow_model_res_folder --expID $id --batch_size 1024 \
             --single_norm_layer $single_norm_layer  --synthetic_mean 0. --synthetic_std 0.1  
done

printf '\n\n'
echo "===> Get the outputs from the shadow models for MIA"
python lira-score.py --res_folder $shadow_model_res_folder

printf "\n\n"
echo "===> Perform MIA using [ target ] samples as query samples"
python lira-plot.py \
            --shadow_data_path $shadow_model_res_folder \
            --test_data_path $target_model_res_folder \
            --eval_target_sample 1  --fpr 1e-3 --save_tag $target_model_res_folder-targetSample

printf "\n"
if [ $modify_loss_module = 1 ]
then
    echo "===> Perform MIA using [ synthetic ] samples as query samples"
    python lira-plot.py \
                --shadow_data_path $shadow_model_res_folder \
                --test_data_path $target_model_res_folder \
                --eval_target_sample 0  --fpr 1e-3 --save_tag $target_model_res_folder-syntheticSample

fi



