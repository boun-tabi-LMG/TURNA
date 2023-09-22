
REPO_DIR=${1}
TPU_VM_NAME=${2}
TRAINING_PREPARATION_DIR=${3}

cd ${REPO_DIR} && \
git archive main -o tmp-turkish-llm.tar.gz && \
gcloud compute tpus tpu-vm scp \
  tmp-turkish-llm.tar.gz \
  ${TRAINING_PREPARATION_DIR}/tpu_vm_setup_extract_the_code_archive.sh \
  ${TPU_VM_NAME}: && \
gcloud compute tpus tpu-vm ssh ${TPU_VM_NAME} -- "bash ./tpu_vm_setup_extract_the_code_archive.sh"
cd - 