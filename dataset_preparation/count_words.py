import argparse
import tensorflow_datasets as tfds

def count_total_words(dataset_name, dataset_dir, split='train'):
    """
    Count words in a given TFDS corpus.
    
    Parameters:
    - dataset_name: Name of the TFDS dataset.
    - dataset_dir: Directory of the TFDS dataset.
    - split: Which split of the dataset to use (e.g., 'train', 'validation', etc.)
    
    Returns:
    - total_words: Total number of words in the corpus.
    """

    # Load the dataset
    dataset = tfds.load(name=dataset_name, data_dir=dataset_dir, split=split)

    total_words = 0
    for text in dataset:
        # Convert tensor to string and tokenize
        words = text["text"].numpy().decode('utf-8').split(" ")
        total_words += len(words)

    return total_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Count tokens in a TFDS corpus using a pretrained tokenizer.')
    
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the TFDS dataset.')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory of the TFDS dataset.')
    parser.add_argument('--split', type=str, default='train',
                        help="Which split of the dataset to use (e.g., 'train', 'validation'). Default is 'train'.")
    args = parser.parse_args()
    total_words = count_total_words(args.dataset_name, args.dataset_dir, args.split)
    print(f"Total words in {args.dataset_name}: {total_words}")

# python count_words.py --dataset_name book_corpus_v2 --dataset_dir /media/disk/datasets/bounllm/tfds/datasets/book_corpus_v2 --split validation