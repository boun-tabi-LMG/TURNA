# set train hyperparams
unset LD_PRELOAD

export GOOGLE_APPLICATION_CREDENTIALS=~/derlem-633f86db7de0.json

PRETRAIN_GIN_FILEPATH=${1:-large_nl36_bs48_pretrain_all_infer.gin}
MODEL_NAME=${2:-"large_nl36_bs48_pretrain_all"}
ITER_COUNT=${3:-10000}
ADDITIONAL_GIN_PARAMS="${@:4}"

GCS_BUCKET="gs://turkish-llm-data"
PROJECT_DIR_NAME="turkish-llm"
PROJECT_DIR="${HOME}/${PROJECT_DIR_NAME}"
T5X_DIR=${HOME}"/t5x"  # directory where the t5x is cloned.
MODEL_DIR="${GCS_BUCKET}/models/${MODEL_NAME}" # maybe we should change this. things get messy when we use the same directory for both model and project.
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/infer.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="${PRETRAIN_GIN_FILEPATH}" \
  --gin.CHECKPOINT_PATH=\"${MODEL_DIR}/checkpoint_${ITER_COUNT}\" \
  --gin.INFER_OUTPUT_DIR=\"${MODEL_DIR}/infer_output_${ITER_COUNT}\"
