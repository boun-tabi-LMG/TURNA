import argparse
import tensorflow_datasets as tfds
from transformers import PreTrainedTokenizerFast
from zemberek import TurkishSentenceExtractor

def count_tokens_with_pretrained_tokenizer(dataset_name, tokenizer_name, split='train'):
    """
    Count tokens in a given TFDS corpus using a pretrained tokenizer.
    
    Parameters:
    - dataset_name: Name of the TFDS dat/aset.
    - split: Which split of the dataset to use (e.g., 'train', 'test', etc.)
    - tokenizer_name: Pretrained tokenizer name.
    
    Returns:
    - total_tokens: Total number of tokens in the corpus.
    """
    
    # Load the dataset
    dataset = tfds.load(name=dataset_name, split=split)
    extractor = TurkishSentenceExtractor()

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)

    total_tokens = 0
    
    for text in dataset:
        # Convert tensor to string and tokenize
        sentences = extractor.from_paragraph(text["text"].numpy().decode('utf-8'))
        for sentence in sentences: 
            tokens = tokenizer.tokenize(sentence)
            total_tokens += len(tokens)
    
    return total_tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count tokens in a TFDS corpus using a pretrained tokenizer.')
    
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the TFDS dataset.')
    parser.add_argument('--tokenizer_name', type=str, required=True,
                        help='Name of the pretrained tokenizer.')
    parser.add_argument('--split', type=str, default='train',
                        help="Which split of the dataset to use (e.g., 'train', 'validation'). Default is 'train'.")

    args = parser.parse_args()

    total_tokens = count_tokens_with_pretrained_tokenizer(args.dataset_name, args.tokenizer_name, args.split)
    print(f"Total tokens in {args.dataset_name} using {args.tokenizer_name}: {total_tokens}")
