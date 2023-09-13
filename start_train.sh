# set train hyperparams
unset LD_PRELOAD

PROJECT_DIR_NAME="turkish-llm"
PROJECT_DIR="${HOME}/${PROJECT_DIR_NAME}"
T5X_DIR=${HOME}"/t5x"  # directory where the t5x is cloned.
MODEL_DIR="${HOME}/${PROJECT_DIR_NAME}" # maybe we should change this. things get messy when we use the same directory for both model and project.
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="base_nl36_pretrain.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\"
