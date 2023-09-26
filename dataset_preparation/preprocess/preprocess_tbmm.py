import re
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, context
from collections import Counter
from pyinstrument import Profiler
import argparse
import math
import os
import logging
import warnings
from zemberek import TurkishSentenceExtractor
from string import punctuation, ascii_lowercase, ascii_uppercase
from pathlib import Path
import json 


extractor = TurkishSentenceExtractor()

warnings.simplefilter(action='ignore', category=UserWarning)

logger = logging.getLogger(__name__)
level = logging.INFO
logger.setLevel(level)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# handler for console info messages
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(level)
logger.addHandler(ch)

# handler for file info messages
fh = logging.FileHandler('log.txt')
fh.setFormatter(formatter)
fh.setLevel(level)
logger.addHandler(fh)

valid_chars = punctuation + ascii_lowercase + ascii_uppercase + "0123456789" + " " + "\n" + "é" + "üğişçöıÜĞİŞÇÖ"

def find_invalid_chars(files):
    invalid_dict = {}
    example_dict = {}
    for file in files:
        text = open(str(file), encoding="utf-8").read()
        for i, c in enumerate(text):
            if c not in valid_chars:
                if c in invalid_dict:
                    invalid_dict[c] += 1
                    if len(example_dict[c]) < 10 and invalid_dict[c] > 500:
                        example_dict[c].append((str(text[i-10:i+10]), str(file)))
                else:
                    invalid_dict[c] = 1
                    example_dict[c] = [(str(text[i-10:i+10]), str(file))]
    invalid_dict = sorted(invalid_dict.items(), key=lambda x:x[1], reverse=True)
    print(invalid_dict)
    print(example_dict)
    return invalid_dict, example_dict

replacement_dict = {'“': '"',
 '’': "'",
 '”': '"',
 '—': "-",
 '‘': "'",
 '–': "-",
 '…': "...",
 '«': '"',
 '»': '"',
 '\xad ' : "",
 '\xad' : "",
 '\xa0': " ",
 "„": '"',
 "•": "",
 "–": "-",
 "~~~ ": "",
 "''": '"',
 "``": '"',
 "≈": "",
 "←": "",
 "►": "-",
 "›": "'",
 "\uf04a":"",
 "☺": ":)",
 "¦": "",
 "©":"",
 '\u200e' : "",
 "\'": "'",
 "\\'": "'",
 "\\\'": "'",
 "�": "",
 '›':"ı",
 'е':"e",
 'Ģ':"ş",
 'ý':"ı",
 'ā':"a",
 '‟':'"',
 '¤':"ğ",
 '\uf0b7':"",
 'õ':"ı",
 '\uf020':"",
 "Ġ": "İ",
 '\uf0bf': "",
 '\uf0bb': "",
 "ŋ": "n",
 "ð": "ğ",
 "ñ": "n",
 "À": "A",
 "í": "i",
 "ī": "i",
 "−": "-",
 "ÅŸ": "ş",
 "Ä±": "ı",
 "\uf0b4": "",
 "\uf0b2": "",
 "—": "-",
 "Ý": "ı",
 "Û": "ğ",
 "\uf0ae": "",
 "\uf001": "",
 "ġ": "Ş",
 "Đ": "İ",
 "³": "ş",
 "§": "ğ"                   
}

def preprocess_text(line):
    for key, value in replacement_dict.items():
        line = line.strip().replace(key, value)
    return line

def remove_punctuation(text):
    """Removes punctuation marks from a given text."""
    punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    for mark in punctuation:
        text = text.replace(mark, '')
    return text

def count_characters(line):
    """Counts the number of characters in a line of text."""
    return len(line)

def digit_ratio(line):
    """Calculates the ratio of digits in a line of text."""
    return sum(c.isdigit() for c in line) / len(line)

def uppercase_ratio(line):
    """Calculates the ratio of uppercase letters in a line of text."""
    return sum(c.isupper() for c in line) / len(line)

def capture_tokens(line):
    """Captures tokens (words) from a line of text."""
    tokens = line.split()
    return tokens

def compute_average_token_length(tokens):
    """
    Computes the average length of tokens (words).

    Returns:
        float: The average token length, or -1 if the tokens list is empty.
    """
    return sum(len(token) for token in tokens) / len(tokens) if tokens else -1

def capture_numbers(line):
    """Captures numbers from a line of text using regular expressions."""
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', line)
    return numbers

email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

def check_email(line):
    """
    Checks if a line of text contains an email address.

    Returns:
        bool: True if an email address is found, False otherwise.
    """
    email_search = email_pattern.search(line)
    return bool(email_search)
    
# Two or three names in the format: "Gözde Serap Gökmen" or "Gözde Serap"
name_pattern_1 = re.compile(r"([A-ZÖÇŞİĞÜ][a-zöçşığü]*)([\s-]([A-ZÖÇŞİĞÜ][a-zöçşığü]*)){1,2}")
# Two or three names in the format: "Gözde SERAP Gökmen" or "Gözde SERAP"
name_pattern_2 = re.compile(r"([A-ZÖÇŞİĞÜ][a-zöçşığü]*)[\s-]([A-ZÖÇŞİĞÜ]*)([\s-][A-ZÖÇŞİĞÜ][a-zöçşığü]*)?")
# Two or three names in the format: "GÜNEY, Kerem. " or "GÜNEY, Kerem Ali. "
name_pattern_3 = re.compile(r"([A-ZÖÇŞİĞÜ]*),([\s-][A-ZÖÇŞİĞÜ][a-zöçşığü]*){1,2}.")

def check_name(line):
    """
    Checks if a line of text contains a name. Looks for exact match, not partial matches.
    Works only for lines that contain two or three tokens. 

    Returns:
        bool: True if a name comprises the line, False otherwise.
    """

    if len(line.strip().split(" ")) in [2, 3]:
        name_search_1 = name_pattern_1.fullmatch(line)
        name_search_2 = name_pattern_2.fullmatch(line)
        name_search_3 = name_pattern_3.fullmatch(line)
        name_search = bool(name_search_1 or name_search_2 or name_search_3)
        return name_search
    else:
        return False

def count_occurrence(lines_without_numbers, target_line):
    """
    Counts the number of occurrences of a target line in a list of lines.

    Returns:
        int: The number of occurrences.
    """
    strip_t = target_line.strip()
    number_removed = re.sub(r'(^(\d+)|(\d+)$)', '', strip_t)
    return lines_without_numbers.count(number_removed)


number_at_beginning_pattern = re.compile(r'^(\d+)(?!\.\d)(\.?\s)?')
number_at_end_pattern = re.compile(r'(\d+)$')

def capture_number_at_beginning(text):
    """
    Captures the number at the beginning of a text.

    Returns:
        int: The captured number, or None if no number is found.
    """
    match = number_at_beginning_pattern.match(text.strip())
    if match:
        return int(match.group(1).strip())
    else:
        return None

def capture_number_at_end(text):
    """
    Captures the number at the end of a text.

    Returns:
        int: The captured number, or None if no number is found.
    """
    match = number_at_end_pattern.search(text.strip()[::-1])
    if match:
        return int(match.group(1).strip()[::-1])
    return None

def capture_dates(line):
    """
    Captures dates in various formats from a line of text.

    Returns:
        list: A list of tuples representing the captured dates.
    """
    date_formats = [
        r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',         # MM/DD/YYYY
        r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b',         # MM-DD-YYYY
        r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b',       # MM.DD.YYYY
        r'\b(\d{4})/(\d{1,2})/(\d{1,2})\b',         # YYYY/MM/DD
        r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',         # YYYY-MM-DD
        r'\b(\d{4})\.(\d{1,2})\.(\d{1,2})\b',       # YYYY.MM.DD
        r'\b(\d{1,2})\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(\d{4})\b',  # DD Mon YYYY
    ]

    dates = []
    for date_format in date_formats:
        matches = re.findall(date_format, line)
        dates.extend(matches)

    return dates


def discard_flags(text):
    tokens = ['ORCID', 'DOI', '.....']
    if any(token in text for token in tokens):
        return True
    return False



def compute_line_statistics(lines):
    """
    Computes various statistics for each line in a list of lines.

    Returns:
        list: A list of dictionaries containing the line statistics.
    """

    # create a list consisting of `lines` with numbers removed
    lines_without_numbers = [re.sub(r'(^(\d+)|(\d+)$)', '', line.strip()) for line in lines]

    statistics = []
    for i, line in enumerate(lines):
        stats = {'line': line}
        #stats['is_turkish'] = is_turkish_content(line)
        stats['characters'] = count_characters(line)
        stats['tokens'] = capture_tokens(line)
        stats['numbers'] = capture_numbers(line)
        stats['token_count'] = len(stats['tokens'])
        stats['number_count'] = len(stats['numbers'])
        stats['average_token_length'] = compute_average_token_length(stats['tokens'])
        stats['number_ratio'] = len(stats['numbers']) / len(stats['tokens']) if stats['tokens'] else -1
        stats['digit_ratio'] = digit_ratio(line)
        stats['uppercase_ratio'] = uppercase_ratio(line)
        stats['dates'] = capture_dates(line)
        stats['occurrence'] = count_occurrence(lines_without_numbers, line)
        stats['discard_flag'] = discard_flags(line)
        stats['initial_number'] = capture_number_at_beginning(line)
        stats['final_number'] = capture_number_at_end(line)
        statistics.append(stats)
    return statistics

def correct_false_values(df, column_name):
    """
    Corrects false values in a DataFrame column based on surrounding values.

    Returns:
        pd.DataFrame: The updated DataFrame with corrected values.
    """
    for i in range(len(df)):
        if not df[column_name].iloc[i]:
            start_index = max(0, i - 5)
            end_index = min(i + 6, len(df))
            window = df[column_name].iloc[start_index:end_index]
            true_count = window[window].count()
            if true_count >= 0.9 * len(window):
                df.loc[i, f'{column_name}_corrected'] = True

    return df



def merge_lines(df, min_page_length=50, page_end_context=250):
    # Create a new column to mark page breaks
    df['page_break'] = df['line'].apply(lambda s: '[PAGE_BREAK]' in s)
    # Create a new column with stripped lines
    df['line_stripped'] = df['line'].str.replace('\[PAGE_BREAK\]', '').str.strip()
    # Initialize variables
    current_page = ''
    overall_text = ''

    # Iterate through the DataFrame
    for i, row in df.iterrows():
        if row['line_stripped'].endswith('-'):
            current_page += row['line_stripped'].rstrip('- ')
        else:
            current_page += row['line_stripped'] + ' '
        # Check for a page break
        if row['page_break']:
            if current_page and len(current_page) > min_page_length:
                # page_end = current_page[-page_end_context:]
                # footnote_pattern = r'[.,;!?]\s?\d+\.\s.*$'
                # cleaned_page_end = re.sub(footnote_pattern, '', page_end).strip()
                # current_page = current_page[:-page_end_context] + cleaned_page_end
                overall_text += current_page + ' '
            current_page = ''

    overall_text += current_page + ' '
    return overall_text



def filter_text(file, output_dir, detect_language=True):
    logger.info(f'Processing {file}')
    file_path = Path(file)
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    no_inline_filename = output_folder / file_path.name


    with open(file, encoding='utf-8') as f:
        content = f.read()

    if content.strip() == '': 
        logger.info('Empty file')
        return 

        
    logger.info(f'Preprocessing and removing text before abstract')
    content = preprocess_text(content)

    logger.info('Computing line statistics')
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    df = pd.DataFrame(compute_line_statistics(lines))
    df['final_number'] = df['final_number'].fillna(-1)
    logger.info(f'Initial number of lines {df.shape[0]}')


    df.drop(df.loc[df['discard_flag']
                   | (df['occurrence'] > 2)].index, inplace=True)
    logger.info(f'Number of lines after dropping bibliography and some other items {df.shape[0]}')

    if df.shape[0] == 0:
        logger.info('No content left after filtering.')
        return

    df.reset_index(drop=True, inplace=True)



    logger.info(f'Number of lines after dropping non-Turkish content {df.shape[0]}')

    if df.shape[0] == 0:
        logger.info('No content left after filtering.')
        return
   
    df.reset_index(drop=True, inplace=True)


    index = df[((df['digit_ratio'] >= 0.2) & (df['average_token_length'] < 4)) # usually table values
                | (df['average_token_length'] < 3)  
                | (df['digit_ratio'] == 1)                                        # page numbers
                | (df['number_ratio'] > 1)                                        # numbers
                | (df['occurrence'] > 2)].index

    df.loc[index, 'drop'] = True

    df = correct_false_values(df, 'drop')

    if df.shape[0] == 0:
        logger.info('No content left after filtering.')
        return

    filtered_df = df.drop(index)

    logger.info(f'Final number of lines {filtered_df.shape[0]}')

    logger.info(f'Merging lines {filtered_df.shape[0]}')

    filtered_content = merge_lines(filtered_df)
    output = filter_sentences(filtered_content)

    with open(no_inline_filename, 'w', encoding='utf-8') as f:
         f.write(output)

def filter_sentences(text):
    sentences =  extractor.from_paragraph(' '.join(text.split('\n')))
    return merge_lines([s for s in sentences if len(s) < 500 and len(s) > 30])

def wrapper_filter(args_tuple):
    try:
        input_file, output_dir = args_tuple
        return filter_text(input_file, output_dir)
    except Exception as e:
        logger.info(f'Error during filtering  {input_file}: {e}')

def profiler_filter(input_tuples, count): 
    for input_tuple in input_tuples[:count]:
        filter_text(*input_tuple)
    
def main():
    arg_parser = argparse.ArgumentParser(description='Filters text from files.')
    arg_parser.add_argument('-i', '--input_dir', type=str, help='The path to the file folder or file.', required=True)
    arg_parser.add_argument('-o', '--output_dir', type=str, help='The path to the output directory.', required=True)
    arg_parser.add_argument('-n', '--num_threads', type=int, help='The number of threads to use.', default=1)
    arg_parser.add_argument('-l', '--time_limit', type=int, help='The time limit for each conversion in seconds.', default=500)
    arg_parser.add_argument('-i', '--profiler',  type=int, help='Enable profiler to measure performance of provided no. of files.', default=0)
    args = arg_parser.parse_args()

    input_path = Path(args.input_dir)
    if input_path.is_file() and input_path.name.endswith('.txt'):
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = [f for d in input_path.iterdir() for f in d.iterdir() if f.name.endswith('.txt')]

    input_tuples = [(str(input_file), args.output_dir) for input_file in input_files]

    if args.profiler == 0:
        with Pool(args.num_threads) as pool:
            results = [pool.apply_async(wrapper_filter, (input_tuple,)) for input_tuple in input_tuples]
            for r, input_file in zip(results, input_files):
                try:
                    r.get(timeout=args.time_limit)  
                except context.TimeoutError:
                    logger.info(f"Filtering timed out for file: {input_file}")
    else:
        with Profiler(interval=0.1) as profiler:
            profiler_filter(input_tuples, args.profiler)
        profiler.print()
        profiler.open_in_browser()

if __name__ == '__main__':
    main()
