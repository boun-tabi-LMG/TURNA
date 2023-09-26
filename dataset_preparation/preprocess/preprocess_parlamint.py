import argparse
import random
from pathlib import Path

def split_train_val(year_path, ratio=0.8):
    documents = list(year_path.glob('*.txt'))
    random.shuffle(documents)

    split_idx = int(len(documents) * ratio)
    train_docs = documents[:split_idx]
    val_docs = documents[split_idx:]

    return train_docs, val_docs

def write_to_file(file_name, doc_names):
    with open(file_name, 'w') as f:
        for doc_name in doc_names:
            f.write(f"{doc_name}\n")

def process_documents(input_path, output_path, train_ratio):

    train_file = 'train.txt'
    val_file = 'val.txt'

    # Clear out existing data in train and val files
    open(train_file, 'w').close()
    open(val_file, 'w').close()

    for year in input_path.iterdir():
        train_docs, val_docs = split_train_val(year, train_ratio)

        write_to_file(train_file, [doc.name for doc in train_docs])
        write_to_file(val_file, [doc.name for doc in val_docs])

        for document in train_docs + val_docs:
            with open(document, 'r') as doc_file:
                lines = doc_file.readlines()

            lines_only_text = [line.split('\t')[1] for line in lines if '\t' in line]
            text = '\n'.join(lines_only_text)
            with open(output_path / document.name, 'w') as out_file:
                out_file.write(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean texts and split into train and validation.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing texts.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory to save cleaned texts.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data. E.g., 0.8 for 80%.")
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
    process_documents(input_path, output_path, args.train_ratio)


# python preprocess_parlamint.py --input_dir ParlaMint-TR.txt --output_dir ParlaMint-TR --train_ratio 0.99
