import pandas as pd
from pathlib import Path
import re
import argparse
import random

def drop_line(line):
    """Determines if a line should be dropped based on certain criteria."""
    
    if 'TURK 10' in line or 'ÖDEV' in line or 'Assignment' in line:
        return True
    
    tokens = line.split()
    num_tokens = len(tokens)
    if num_tokens == 1:
        return all(ch.isnumeric() for ch in tokens[0])
    
    upper_case_ratio = sum(token[0].isupper() for token in tokens) / num_tokens
    return upper_case_ratio > 0.7

def truncate_after_bibliography(text, bibliography_pattern):
    """Truncates the text after finding a bibliography keyword."""
    
    match = re.search(bibliography_pattern, text)
    if match and match.start() > len(text) * 0.7:
        return text[:match.start()]
    return text

def get_initial_lines(lines):
    """Retrieves the first few lines without many uppercase words."""
    
    initial_lines_ratio = [(sum(token[0].isupper() for token in line.split()) / len(line.split())) for line in lines[:10]]
    return [lines[i] for i in range(10) if initial_lines_ratio[i] < 0.7]

def clean_text_from_file(path, bibliography_pattern):
    """Cleans and returns text from a file."""
    
    with path.open(encoding='utf-8') as f:
        text = f.read()
    
    text = truncate_after_bibliography(text, bibliography_pattern)
    lines = [l for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    
    initial_lines = get_initial_lines(lines)
    filtered_lines = [l for l in initial_lines if not drop_line(l)] + lines[10:]
    return ' '.join(filtered_lines)

def split_files_into_train_val(all_files, train_ratio=0.9):
    random.shuffle(all_files)  # Randomly shuffle the file list
    num_train = int(len(all_files) * train_ratio)
    return all_files[:num_train], all_files[num_train:]

def process_files(input_path, output_path, bibliography_pattern):
    file_list = []
    for file in input_path.iterdir():
        cleaned_text = clean_text_from_file(file, bibliography_pattern)
        if cleaned_text:
            with open(output_path / file.name, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            file_list.append(file.name)
    return file_list

def main():
    parser = argparse.ArgumentParser(description="Clean texts and split into train and validation.")
    parser.add_argument("--input_dir", required=True, help="Path to the input directory containing texts.")
    parser.add_argument("--output_dir", required=True, help="Path to the output directory to save cleaned texts.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data. E.g., 0.8 for 80%.")
    args = parser.parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    bibliography_keywords = ['Kaynakça', 'Kaynaklar', 'Referanslar']
    bibliography_pattern = re.compile(r'(' + '|'.join(bibliography_keywords) + r')\b', re.IGNORECASE)
    
    processed_files = process_files(input_path, output_path, bibliography_pattern)

    train_files, val_files = split_files_into_train_val(processed_files, args.train_ratio)

    with open(output_path.parent / f'train.txt', 'w') as f:
        f.write('\n'.join(train_files))
        
    with open(output_path.parent / f'val.txt', 'w') as f:
        f.write('\n'.join(val_files))


if __name__ == "__main__":
    main()


# python preprocess_creative_writings.py --input_dir texts --output_dir texts_clean --train_ratio 0.98
