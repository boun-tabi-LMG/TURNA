#!/bin/bash

model_conf_prefixes='tiny'

batch_sizes='48'
input_lengths='512'
train_steps='1000000'
task_name='pretrain_all'

date_identifier=$(date +%Y%m%d_%H%M%S)

for model_conf_prefix in ${model_conf_prefixes}; do
    for batch_size in ${batch_sizes}; do
        for input_length in ${input_lengths}; do
            echo "model_conf_prefix: ${model_conf_prefix}, batch_size: ${batch_size}, input_length: ${input_length}"
            bash ./start_train.sh gins/${model_conf_prefix}_pretrain.gin \
                ${model_conf_prefix}-bs_${batch_size}-il_${input_length}-${date_identifier} \
                --gin.MIXTURE_OR_TASK_NAME=\"${task_name}\" \
                --gin.BATCH_SIZE=${batch_size} \
                --gin.TASK_FEATURE_LENGTHS=\{\'inputs\':${input_length},\'targets\':${input_length}\} \
                --gin.TRAIN_STEPS=${train_steps} \
                --gin.train.eval_period=10000 \
                --gin.utils.SaveCheckpointConfig.period=10000 \
                >> train-${model_conf_prefix}-bs_${batch_size}-il_${input_length}-ts_${train_steps}-${date_identifier}.log 2>&1
        done
    done
done