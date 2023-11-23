# README

## Guide to Run Machine Learning Training on a Preemptible TPU

This README walks through the process of running a machine learning training session using a preemptible Cloud TPU (Tensor Processing Unit) on Google Cloud Platform (GCP). It explains how to create a preemptible TPU, set it up, and start training.

### Defining Variables
For consistency and ease of use, let's define some variables that we'll use throughout the guide:

```bash
TPU_NAME="tpu-vm-08"        # Replace with the name of your TPU
HOST_NAME="minerva"         # Replace with your actual host name
ZONE="europe-west4-a"       # Replace with your desired GCP zone
ACCELERATOR_TYPE="v3-8"     # Specify the TPU type
TPU_VERSION="tpu-vm-tf-2.13.0"  # TPU software version
TMP_CODE_ARCHIVE_NAME="tmp-turkish-llm.tar.gz"  # Name of the code archive. 
```

**IMPORTANT:** `TMP_CODE_ARCHIVE_NAME` is specifically for the basename of the code archive file that will be generated and transferred to the TPU. Ensure that it does not include the full path or any directory names.

### Step 1: Create a Preemptible TPU
A preemptible TPU is an affordable and ephemeral TPU instance that can be preempted at any time by the cloud provider. They are well-suited for workloads that can withstand interruptions like machine learning training.

Use the following command to create a preemptible TPU, utilizing the variables defined above:

```bash
gcloud compute tpus tpu-vm create $TPU_NAME \
  --zone=$ZONE \
  --accelerator-type=$ACCELERATOR_TYPE \
  --version=$TPU_VERSION \
  --preemptible
```

### Step 2: Set Up the TPU
After creating your TPU, you'll need to set it up. First, establish an SSH connection to the machine:

```bash
ssh -i ~/.ssh/id_rsa $HOST_NAME
```
Replace the `id_rsa` with the path to your private SSH key specific to your HOST_NAME to authenticate the connection.

### Step 3: Start Training
Use `tmux` or a similar session manager to ensure your session remains active after any potential disconnection.

Switch to `tmux` using:

```bash
tmux at
```

To prepare the TPU for training and begin the process, run your setup script:

```bash
bash training_preparation/tpu_vm_setup_driver.sh $TPU_NAME $TMP_CODE_ARCHIVE_NAME```

Make sure the script path `training_preparation/tpu_vm_setup_driver.sh` and `$TPU_NAME` correctly reflect your actual configurations and the TPU you're setting up.

**Note**:
- Ensure you use the correct TPU name and scripts according to your project.
- Your GCP account must have the required permissions for creating and managing TPUs and VMs.
- Customize the `gcloud` command and scripts as per your specific project needs.
- It is assumed you are familiar with GCP, `gcloud` CLI, and containerized machine learning workflows.

### Step 3: Start Training

First, establish an SSH connection to the machine:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME
```

Then, start training:

```bash
export MODEL_ID=large_nl36-bs_48-il_512-20231108_1910
export TRIAL_NO=02
cd ~/turkish-llm/ && nohup bash ./start_train.sh gins/large_nl36_bs48_pretrain_all.gin ${MODEL_ID} --gin.MIXTURE_OR_TASK_NAME=\"pretrain_all_v2\" >> train-${MODEL_ID}-${TRIAL_NO}.log &
```

Increase the `TRIAL_NO` variable for each new training session as needed.

### Step 4: Monitor Training

Use the following command to monitor the TPU:

```bash
gcloud compute tpus tpu-vm list --filter="schedulingConfig.preemptible=true"
```

If the TPU is preempted, it will indicate that in the output as PREEMPTED. You can also add "--format=json" to get the output in JSON format.

If you see that a TPU has been preempted, you can create a new one and resume training. You can also use the same TPU name as before, as long as you delete the old TPU first (but don't do that generally).


# Stop TPU instance

```bash 
gcloud compute tpus tpu-vm stop $TPU_NAME --zone=$ZONE
```


# Monitoring

/media/disk/home/onur.gungor/projects/research/projects/focus/turkish-llm/turkish-llm


