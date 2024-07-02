#!/bin/bash
dataset=$1 
train_size=12500 
network=wideresnet2810


save_tag=dataPoisoned
target_model_res_folder=lira-$dataset-dataPoisoned-$train_size-targetModel


echo "======================================="
echo "======  Train the target model"
echo "======================================="
python train-data-poison.py --lr 0.1  --net_type $network --dataset $dataset \
      --train_size $train_size --epoch 200 --save_tag $save_tag

printf '\n\n'
echo "===> Get the outputs from the target model for MIA"
python lira-inference.py \
        --resume_path data-poisoned-checkpoint/$dataset-trainSize-$train_size-$save_tag.pth.tar \
        --train_size $train_size --dataset $dataset --single_norm_layer 1 \
        --res_folder $target_model_res_folder --target_model 1 --batch_size 1024 
python lira-score.py  --res_folder $target_model_res_folder


echo "======================================="
echo "====== Train the shadow models for LiRA"
echo "======================================="

shadow_model_res_folder=lira-$dataset-dataPoisoned-$train_size
declare -i total_models=128 # number of shadow models
declare -i index=0
index=$total_models-1
for id in $(seq 0 $index); do
    save_tag=dataPoisoned-$id
    python train-data-poison-lira.py --lr 0.1  --train_size $train_size --dataset $dataset \
        --single_norm_layer 1 --epoch 200 --net_type $network \
        --expID $id --save_tag $save_tag --total_models $total_models \
        --res_folder $shadow_model_res_folder

    python lira-inference.py \
        --resume_path lira-data-poisoned-checkpoint/$dataset-trainSize-$train_size-$save_tag.pth.tar \
        --train_size $train_size --dataset $dataset --single_norm_layer 1 \
        --res_folder $shadow_model_res_folder --expID $id  --batch_size 1024 
done 
printf '\n\n'
echo "===> Get the outputs from the shadow models for MIA"
python lira-score.py --res_folder $shadow_model_res_folder

printf '\n\n'
echo "===> Perform MIA"
python lira-plot.py \
        --shadow_data_path $shadow_model_res_folder \
        --test_data_path $target_model_res_folder \
        --eval_target_sample 1  --fpr 1e-3 --save_tag $target_model_res_folder-targetSample
printf '\n\n\n\n'


