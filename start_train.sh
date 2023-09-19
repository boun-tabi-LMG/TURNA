# set train hyperparams
unset LD_PRELOAD

PRETRAIN_GIN_FILEPATH=${1:-base_nl36_pretrain.gin}
MODEL_NAME=${2:-"base_nl36"}

GCS_BUCKET="gs://turkish-llm-data"
PROJECT_DIR_NAME="turkish-llm"
PROJECT_DIR="${HOME}/${PROJECT_DIR_NAME}"
T5X_DIR=${HOME}"/t5x"  # directory where the t5x is cloned.
MODEL_DIR="${GCS_BUCKET}/models/${MODEL_NAME}" # maybe we should change this. things get messy when we use the same directory for both model and project.
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="${PRETRAIN_GIN_FILEPATH}" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\"
