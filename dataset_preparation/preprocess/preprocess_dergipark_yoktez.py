import pandas as pd
from pathlib import Path
import re
import argparse
import random

def split_files_into_train_val(all_files, train_ratio=0.9):
    random.shuffle(all_files)  # Randomly shuffle the file list
    num_train = int(len(all_files) * train_ratio)
    return all_files[:num_train], all_files[num_train:]

def main():
    parser = argparse.ArgumentParser(description="Clean texts and split into train and validation.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing texts.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data. E.g., 0.8 for 80%.")
    args = parser.parse_args()
    input_path = Path(args.input_dir)

    no_inline_path = input_path /  "no_inline_txt"
    files = [f.name for f in no_inline_path.iterdir() if f.name.endswith('.txt')]

    drop_path = input_path / "drop.txt"
    with open(str(drop_path)) as f:
        drop_files = [l.strip() for l in f.readlines()]

    print(f"Total no. files: {len(files)}")
    print(f"Total no. files to drop: {len(drop_files)}")

    final_files = list(set(files) - set(drop_files))
    print(f"Total no. files left: {len(final_files)}")

    train_files, val_files = split_files_into_train_val(final_files, args.train_ratio)

    with open(input_path / f'train.txt', 'w') as f:
        f.write('\n'.join(train_files))
        
    with open(input_path / f'val.txt', 'w') as f:
        f.write('\n'.join(val_files))


if __name__ == "__main__":
    main()

