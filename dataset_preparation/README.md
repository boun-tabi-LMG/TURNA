# Dataset preparation

## ParlaMintTR

### Downloading the dataset
ParlaMintTR corpus is available on [Clarin](https://www.clarin.si/repository/xmlui/handle/11356/1486) 
```bash
wget https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1486/ParlaMint-TR.tgz?sequence=25&isAllowed=y -O ParlaMint-TR.tgz
```

### Extracting the dataset
```bash
tar -xvzf ParlaMint-TR.tgz
```

### Preprocessing the dataset
Once the dataset is downloaded, place the dataset in your desired directory (e.g., data/raw/ParlaMint-TR/). Then, preprocess it with:
```bash
python preprocess/preprocess_parlamint.py --input_dir ../data/raw/ParlaMint-TR/ParlaMint-TR.txt --output_dir ../data/raw/ParlaMint-TR/ParlaMint-TR-clean --train_ratio 0.998
```

### Create a TFDS dataset
Navigate to the parlamint_tr directory and build the dataset using:
```bash
cd parlamint_tr
tfds build --manual_dir ../../data/raw/ParlaMint-TR/ParlaMint-TR-clean --data_dir ../../data/tfds/parlamint_tr
```

## Bilkent Creative Writings Dataset
The writings are hosted on [Bilkents Turkish Journal](https://stars.bilkent.edu.tr/turkce/) in pdf format.  
##### Downloading & processing the dataset
Scripts to scrape the PDF files and convert them to text format are available on [this GitHub Repository](https://github.com/selimfirat/bilkent-turkish-writings-dataset.git).

### Preprocessing the dataset
Once you obtain the text files, place them in your directory of choice (e.g., data/raw/bilkent-creative-writings/texts/). Then, preprocess them using:

```bash
python preprocess/preprocess_creative_writings.py --input_dir ../data/raw/bilkent-creative-writings/texts --output_dir ../data/raw/bilkent-creative-writings/texts_clean --train_ratio 0.98
```

### Create a TFDS dataset
Navigate to the bilkent_creative_writings directory and build the dataset:
```bash
cd bilkent_creative_writings
tfds build  --manual_dir ../../data/raw/bilkent-creative-writings/texts_clean/ --data_dir ../../data/tfds/bilkent_creative_writings
```

## Samples

python -m dataset_preparation.process_a_tfds_dataset_to_get_samples --task_name deneme --dataset_name oscarmc4_cleaned_hf_dataset_subset_combined_tfds:1.0.0 --dataset_gcs_url gs://turkish-llm-data/ --output_filepath /media/disk/datasets/bounllm/sonrasil_oscarmc4_cleaned_hf_dataset_subset_combined_tfds.tar.gz

# Creating a new TFDS dataset

cd /media/disk/datasets/bounllm/tfds
tfds new my_dataset  # Create `my_dataset/my_dataset.py` template files
# [...] Manually modify `my_dataset/my_dataset_dataset_builder.py` to implement your dataset. (Check out example in the dergipark/ folder)
cd my_dataset/
tfds build  # Download and prepare the dataset to `~/tensorflow_datasets/`

# Specify another data / input directory:
tfds build --manual_dir /media/disk/datasets/bounllm/dergipark/dergipark-090920230005 --data_dir /media/disk/datasets/bounllm/tfds/datasets/dergipark

# Loading a TFDS dataset
tfds.load("dergipark", data_dir="/media/disk/datasets/bounllm/tfds/datasets/dergipark")