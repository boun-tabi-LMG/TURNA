#!/bin/bash

# gcloud beta services identity create --service tpu.googleapis.com --project derlem

# gcloud compute tpus tpu-vm create tpu-vm-05 \
#   --zone=europe-west4-a \
#   --accelerator-type=v4-8 \
#   --version=tpu-vm-tf-2.13.0

TPU_VM_NAME=${1:-tpu-vm-05}

SCRIPT_DIR=$(dirname "$0")

TRAINING_PREPARATION_DIR="${SCRIPT_DIR}"

REPO_DIR=${TRAINING_PREPARATION_DIR}/..

echo ${SCRIPT_DIR} $REPO_DIR $TRAINING_PREPARATION_DIR

bash ./tpu_vm_cp_code.sh ${REPO_DIR} ${TPU_VM_NAME} ${TRAINING_PREPARATION_DIR}

gcloud compute tpus tpu-vm scp ${TRAINING_PREPARATION_DIR}/tpu_vm_setup.sh ${TPU_VM_NAME}: && \
gcloud compute tpus tpu-vm ssh ${TPU_VM_NAME} -- "bash ./tpu_vm_setup.sh"

gcloud compute tpus tpu-vm scp vocabulary_preparation/SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model \
${TPU_VM_NAME}:~/turkish-llm/SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model
