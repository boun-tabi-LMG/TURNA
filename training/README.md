## Guide to TURNA Training on a Preemptible TPU

This README walks through the process of training TURNA session using a preemptible Cloud TPU (Tensor Processing Unit) on Google Cloud Platform (GCP). It explains how to create a preemptible TPU, set it up, and start training.


### Defining Variables
For consistency and ease of use, let's define some variables that we'll use throughout the guide:

```bash
TPU_NAME="tpu-vm-08"                            # Replace with the name of your TPU
HOST_NAME="minerva"                             # Replace with your actual host name
ZONE="europe-west4-a"                           # Replace with your desired GCP zone
ACCELERATOR_TYPE="v3-8"                         # Specify the TPU type
TPU_VERSION="tpu-vm-tf-2.13.0"                  # TPU software version
TMP_CODE_ARCHIVE_NAME="tmp-turna.tar.gz"  # Name of the code archive.
PROJECT="derlem"                                # Replace with your project name
```

Steps to configure `gcloud` after these variables are defined and before continuing below are [at the bottom](#setting-up-gcloud).

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

Clone the repository:

```bash
git clone https://github.com/boun-tabi-LMG/turna.git
```

Make sure that `derlem-633f86db7de0.json` is located in the parent directory of `turna`. If not, copy it there.

Navigate to the `turna` directory:

```bash
cd turna
```

Use `tmux` or a similar session manager to ensure your session remains active after any potential disconnection.

Switch to `tmux` using:

```bash
tmux at
```

To prepare the TPU for training and begin the process, run your setup script:

```bash
bash training/tpu_vm_setup_driver.sh $TPU_NAME $TMP_CODE_ARCHIVE_NAME
```

Make sure the script path `training/tpu_vm_setup_driver.sh` and `$TPU_NAME` correctly reflect your actual configurations and the TPU you're setting up.

**Note**:
- Ensure you use the correct TPU name and scripts according to your project.
- Your GCP account must have the required permissions for creating and managing TPUs and VMs.
- Customize the `gcloud` command and scripts as per your specific project needs.
- It is assumed you are familiar with GCP, `gcloud` CLI, and containerized machine learning workflows.

### Step 3: Start Training
After setting up the TPU, you can start training.

First, establish an SSH connection to the machine:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME
```

Then, start training:

```bash
export MODEL_ID=TURNA
export TRIAL_NO=01
cd ~/turkish-llm/ && nohup bash ./start_train.sh large_nl36_bs48_pretrain_all.gin ${MODEL_ID} --gin.MIXTURE_OR_TASK_NAME=\"pretrain_all_v2\" >> train-${MODEL_ID}-${TRIAL_NO}.log &
```

Increase the `TRIAL_NO` variable for each new training session as needed.

The default number of training steps is `3000000`, but you can optionally select 
another target by adding an argument before `>>`:

```bash
cd ~/turna/ && nohup bash ./start_train.sh large_nl36_bs48_pretrain_all.gin ${MODEL_ID} --gin.MIXTURE_OR_TASK_NAME=\"pretrain_all_v2\" --gin.TRAIN_STEPS=4000000 >> train-${MODEL_ID}-${TRIAL_NO}.log &
```

### Step 4: Monitor Training

Use the following command to monitor the TPU:

```bash
gcloud compute tpus tpu-vm list --filter="schedulingConfig.preemptible=true"
```

If the TPU is preempted, it will indicate that in the output as PREEMPTED. You can also add "--format=json" to get the output in JSON format.

If you see that a TPU has been preempted, you can create a new one and resume training. You can also use the same TPU name as before, as long as you delete the old TPU first (but don't do that generally).

### Stop TPU instance

```bash 
gcloud compute tpus tpu-vm stop $TPU_NAME --zone=$ZONE
```

### Setting up `gcloud`

1. Log in: `gcloud auth login`
2. Set zone: `gcloud config set compute/zone $ZONE`
3. Set project: `gcloud config set project $PROJECT`
