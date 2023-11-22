
TPU_VM_NAME=${1}

gcloud compute tpus tpu-vm ssh ${TPU_VM_NAME} -- "cd ~/turkish-llm && nohup bash ./start_train.sh large_nl36_bs48_pretrain_all.gin large_nl36-bs_48-il_512-20231013_043959 >> train-large_nl36-bs_48-il_512-20231013_043959-01.log &"