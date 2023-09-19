#!/bin/bash

# gcloud beta services identity create --service tpu.googleapis.com --project derlem

TPU_VM_NAME="tpu-vm-04"

SCRIPT_DIR=$(dirname "$0")

TRAINING_PREPARATION_DIR="${SCRIPT_DIR}"

REPO_DIR=${TRAINING_PREPARATION_DIR}/..

echo ${SCRIPT_DIR} $REPO_DIR $TRAINING_PREPARATION_DIR

cd ${REPO_DIR} && \
git archive main -o tmp-turkish-llm.tar.gz && \
gcloud compute tpus tpu-vm scp tmp-turkish-llm.tar.gz ${TPU_VM_NAME}: && \
cd - 

gcloud compute tpus tpu-vm scp ${TRAINING_PREPARATION_DIR}/tpu_vm_setup.sh ${TPU_VM_NAME}: && \
gcloud compute tpus tpu-vm ssh ${TPU_VM_NAME} -- "bash ./tpu_vm_setup.sh"

gcloud compute tpus tpu-vm scp vocabulary_preparation/SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model \
${TPU_VM_NAME}:~/turkish-llm/SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model
