# Dataset preparation

## Oscar MC4 corpus & Book Corpus & DergiPark & YökTez 

These datasets are not publicly available.

To compile the DergiPark and YökTez datasets, visit https://github.com/boun-tabi-LMG/turkish-academic-text-harvest.

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
cd dataset_preparation/parlamint_tr
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
cd dataset_preparation/bilkent_creative_writings
tfds build  --manual_dir ../../data/raw/bilkent-creative-writings/texts_clean/ --data_dir ../../data/tfds/bilkent_creative_writings
```

