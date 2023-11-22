
#!/bin/bash

REPO_DIR=/Users/onur.gungor/Desktop/projects/research/projects/focus/turkish-llm
TPU_NAME=tpu-vm-08
WAIT_TIME=1800
ITERATION=0

pyenv activate turkish-llm

while true; do
    cd ${REPO_DIR} && \
        poetry run python turkish-llm/training_tracking/watch_training.py --tpu_name ${TPU_NAME}
    
    sleep ${WAIT_TIME}  # Wait for 30 minutes (1800 seconds)
    ITERATION=$((ITERATION + 1))  # Increment the ITERATION variable by 1
    MODULO=$((ITERATION % 12))  # Calculate the modulo of ITERATION with 12
    echo "Iteration: ${ITERATION}"
    if [[ ${MODULO} -eq 0 ]]; then
        cd ${REPO_DIR} && \
        poetry run python turkish-llm/training_tracking/watch_training.py --tpu_name ${TPU_NAME} --send_report
    fi
    
done

