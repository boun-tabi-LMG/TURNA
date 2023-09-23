#!/bin/bash

model_conf_prefixes=large large_nl36 base_nl36

batch_sizes=64 128 256
input_lengths=512 1024 2048

date_identifier=$(date +%Y%m%d_%H%M%S)

for model_conf_prefix in ${model_conf_prefixes}; do
    for batch_size in ${batch_sizes}; do
        for input_length in ${input_lengths}; do
            echo "model_conf_prefix: ${model_conf_prefix}, batch_size: ${batch_size}, input_length: ${input_length}"
            nohup bash ./start_train.sh ${model_conf_prefix}_pretrain.gin \
                ${model_conf_prefix}-bs_${batch_size}-il_${input_length}-${date_identifier} \
                --gin.BATCH_SIZE=\"${batch_size}\" \
                --gin.TASK_FEATURE_LENGTHS=\"\{\"inputs\": ${input_length}, \"targets\": ${input_length}\}\" \
                >> train-${model_conf_prefix}-bs_${batch_size}-il_${input_length}-${date_identifier}.log & 
        done
    done
done