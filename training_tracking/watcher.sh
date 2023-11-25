
#!/bin/bash

REPO_DIR=$(dirname "$0")/..
TPU_NAME=${1}
WAIT_TIME=1800
ITERATION=0

source /home/onur.gungor/.pyenv/versions/turkish-llm/bin/activate

while true; do
    if [ -z "$TPU_NAME" ]; then
        echo "TPU_NAME is not set!"
        arguments=''
    else
        arguments="--tpu_name ${TPU_NAME}"
    fi
    
    cd ${REPO_DIR} && \
        python training_tracking/watch_training.py ${arguments}
    
    sleep ${WAIT_TIME}  # Wait for 30 minutes (1800 seconds)
    ITERATION=$((ITERATION + 1))  # Increment the ITERATION variable by 1
    MODULO=$((ITERATION % 12))  # Calculate the modulo of ITERATION with 12
    echo "Iteration: ${ITERATION}"
    if [[ ${MODULO} -eq 0 ]]; then
        cd ${REPO_DIR} && \
        python training_tracking/watch_training.py ${arguments} --send_report
    fi
    
done

