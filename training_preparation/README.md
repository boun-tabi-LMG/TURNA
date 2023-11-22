cd ~/turkish-llm/ && nohup bash ./start_train.sh gins/large_nl36_bs48_pretrain_all.gin ${MODEL_ID} --gin.MIXTURE_OR_TASK_NAME=\"pretrain_all_v2\" >> train-${MODEL_ID}-${TRIAL_NO}.log &
