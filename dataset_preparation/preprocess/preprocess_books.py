from pathlib import Path
import re
import argparse
import random
import os

email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
url_pattern = re.compile(r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|biz|info|jobs|mobi|museum|name|post|pro|tel|travel|xxx|tr)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|tr)\b/?(?!@)))')
bibliography_keywords = ['Kaynakça', 'Kaynaklar', 'Referanslar', 'Notlar', 'Notlar:', 'SON', 'Notes',
                         'BİBLİYOGRAFYA', 'DİPNOTLAR', 'KAYNAKÇA', 'Dipnotlar', 'İçindekiler', 'BİTTİ',
                        'NOTLAR', 'EKLER', 'KAYNAKLAR', 'İÇİNDEKİLER', 'Kılavuz', "PERDE", "-SON-",
                        "MODERN ESERLER", "ESERLER"] 
book_intro_keywords = ["©", "Tel.", "Faks", "Tel:", "Faks:", "Fax:", "ISBN", 
                           "Kapak fotoğrafı", "Kapak tasarım", "Yazan:", "Yayın hakları",
                          "Yayınevi", "Yayınları", "No.", "No:", "™", "Basım",
                          "Yayıncılık", "Baskı", "YAYINEVİ", "Telefon:", ". baskı", ". basım",
                          "T:", "F:", "çeviren:", "Çeviren:", "Hakkı:"]

replacement_dict = {'“': '"',  '’': "'",  '”': '"',  '—': "-",
 '‘': "'",  '–': "-",  '…': "...",  '«': '"',  '»': '"',  '\xad ' : "",
 '\xad' : "",  '\xa0': " ",  "„": '"',  "•": "",  "–": "-",  "~~~ ": "",
 "''": '"',  "``": '"',  "≈": "",  "←": "",  "►": "-",  "›": "'",  "\uf04a":"",
 "☺": ":)",  "¦": "",  "©":"",  '\u200e' : "",  "\'": "'",  "\\'": "'",  "\\\'": "'",
 "�": "",  '›':"ı",  'е':"e",  'Ģ':"ş",  'ý':"ı",  'ā':"a",  '‟':'"',  '¤':"ğ",
 '\uf0b7':"",  'õ':"ı",  '\uf020':"",  "Ġ": "İ",  '\uf0bf': "",  '\uf0bb': "",  "ŋ": "n",
 "ð": "ğ", "ñ": "n", "À": "A", "í": "i", "ī": "i", "−": "-", "ÅŸ": "ş", "Ä±": "ı",
 "\uf0b4": "", "\uf0b2": "", "—": "-", "Ý": "ı", "\uf0ae": "", "\uf001": "", "ġ": "Ş",
 "Đ": "İ", "³": "ş", "§": "ğ", "Ǿ": "", "ĥ": "h", "ķ": "k", "ǿ": "", "·": "",   "\ufeff": "", 
 "ħ": "h", "ŧ": "t", "̧ ": " ", "ŝ": "s", "̈ ": " ", "Ķ": "K", "Ŧ": "T", "¬": "" }
    

def drop_initial_line(line):
    """Determines if a line (in the first 100 lines) should be dropped based on certain criteria."""
    
    for keyword in book_intro_keywords:
        if keyword in line:
            return True
    
    tokens = line.split()
    num_tokens = len(tokens)
    
    upper_case_ratio = sum(token[0].isupper() for token in tokens) / num_tokens
    return upper_case_ratio > 0.7

def drop_line(line):
    """Determines if a line should be dropped based on certain criteria."""
    tokens = line.split()
    num_tokens = len(tokens)
    numeric_line_flag = False
    if num_tokens == 1:
        numeric_line_flag = all(ch.isnumeric() for ch in tokens[0])
        
    email_url_flag = check_email_url(line)
    
    return numeric_line_flag or email_url_flag

def check_email_url(line):
    """
    Checks if a line of text contains an email address or a URL.

    Returns:
        bool: True if an email address is found, False otherwise.
    """
    email_search = email_pattern.search(line)
    url_search = url_pattern.search(line)
    return bool(email_search) or bool(url_search)

def truncate_after_bibliography(lines):
    """Truncates the text if a line matches a bibliography keyword exactly."""
    
    bibliography_index = -1
    
    for i, line in enumerate(lines):
        tokens = line.split()
        if len(tokens) == 1:
            for keyword in bibliography_keywords:
                if line == keyword:
                    bibliography_index = i
                    break
        if bibliography_index != -1:
            break
    
    if bibliography_index != -1 and bibliography_index > len(lines) * 0.7:
        return lines[:bibliography_index]
    return lines

def preprocess_text(line):
    for key, value in replacement_dict.items():
        line = line.strip().replace(key, value)
    return line

def clean_text_from_file(path):
    """Cleans and returns text from a file."""
        
    try:
        with open(path, encoding='utf-8') as f:
            text = f.read()
    except:
        print("utf8 error", path)
        with open(path) as f:
            text = f.read()
    
    text = preprocess_text(text)
    
    #text = truncate_after_bibliography(text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return None
    
    # Apply drop rules to initial (first 100) lines
    filtered_lines = [l for l in lines[:100] if not drop_initial_line(l)] + lines[100:] 
    # Apply drop rules to all lines
    filtered_lines = [l for l in filtered_lines if not drop_line(l)]

    truncated_lines = truncate_after_bibliography(filtered_lines)
    
    return ' '.join(truncated_lines)



def split_files_into_train_val(all_files, train_ratio=0.9):
    random.shuffle(all_files)  # Randomly shuffle the file list
    num_train = int(len(all_files) * train_ratio)
    return all_files[:num_train], all_files[num_train:]


def process_files(input_path, output_path):
    file_list = []
    for file in input_path.iterdir():
        cleaned_text = clean_text_from_file(file)
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
    
    processed_files = process_files(input_path, output_path)
    #processed_files = [f.name for f in output_path.iterdir()]

    train_files, val_files = split_files_into_train_val(processed_files, args.train_ratio)

    with open(output_path.parent / f'train.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_files))
        
    with open(output_path.parent / f'val.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(val_files))


if __name__ == "__main__":
    main()


# python preprocess_books.py --input_dir texts --output_dir texts_clean --train_ratio 0.9997