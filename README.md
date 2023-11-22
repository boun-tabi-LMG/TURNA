# Turkish LLM 

The repository contains scripts and instructions for training a Turkish Language Model using `t5x` framework on Google Cloud Platform.

The repository is organized as follows:
- `dataset_preparation`: Scripts and instructions for preparing the datasets used in the project.
- `vocabulary_preparation`: Scripts and instructions for preparing the tokenizers and vocabularies used in the project.
- `training_preparation`: Scripts and instructions for preparing the training environment.
- `training_tracking`: Scripts and instructions for tracking the training process.
- `t5x_to_hf_conversion`: Scripts and instructions for converting a `t5x` model to a HuggingFace model.

Each directory contains a `README.md` file with detailed instructions.

## How to run start_train.sh
--------------------------

```bash
export MODEL_ID=large_nl36-bs_48-il_512-20231108_1910
export TRIAL_NO=02
cd ~/turkish-llm/ && nohup bash ./start_train.sh gins/large_nl36_bs48_pretrain_all.gin ${MODEL_ID} --gin.MIXTURE_OR_TASK_NAME=\"pretrain_all_v2\" >> train-${MODEL_ID}-${TRIAL_NO}.log &
```

## How to run start_infer.sh
-------------------------

```bash
export MODEL_ID=large_nl36-bs_48-il_512-20231108_1910
export TRIAL_NO=02
cd ~/turkish-llm/ && nohup bash ./start_infer.sh gins/large_nl36_bs48_pretrain_all.gin ${MODEL_ID} --gin.MIXTURE_OR_TASK_NAME=\"pretrain_all_v2\" >> infer-${MODEL_ID}-${TRIAL_NO}.log &
```