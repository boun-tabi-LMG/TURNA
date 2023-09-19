"""Process a TFDS dataset to get samples."""
import functools
import gzip
import json
import logging
import seqio
from tqdm import tqdm

from task_utils.ul2_objective import ul2_objective

from t5.evaluation import metrics

logger = logging.getLogger(__name__)

# values from UL2 paper https://arxiv.org/pdf/2205.05131.pdf chapter 3.1.2 table 1
R_DENOISER_SPAN_LENGTHS = [3.0, 8.0]
X_DENOISER_SPAN_LENGTHS = [3.0, 8.0, 64.0, 64.0]
R_DENOISER_CORRUPT_RATES = [0.15, 0.15]
X_DENOISER_CORRUPT_RATES = [0.5, 0.5, 0.15, 0.5]

R_DENOISER_TOKEN_PREFIX = "[NLU]"
X_DENOISER_TOKEN_PREFIX = "[NLG]"
S_DENOISER_TOKEN_PREFIX = "[S2S]"

TaskRegistry = seqio.TaskRegistry


def get_vocabulary(vocabulary_filepath):
    """_summary_

    Args:
        vocabulary_filepath (_type_): _description_

    Returns:
        _type_: _description_
    """
    vocabulary = seqio.SentencePieceVocabulary(vocabulary_filepath, extra_ids=100)
    return vocabulary


def register_the_task(task_name, dataset_name, dataset_gcs_url):
    """_summary_

    Args:
        dataset_name (_type_): _description_
        dataset_gcs_url (_type_): _description_
    """

    vocabulary = get_vocabulary(
        "SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model"
    )

    DEFAULT_OUTPUT_FEATURES = {
        "inputs": seqio.Feature(vocabulary=vocabulary, add_eos=True, required=False),
        "targets": seqio.Feature(vocabulary=vocabulary, add_eos=True),
    }

    TaskRegistry.add(
        task_name,
        source=seqio.TfdsDataSource(
            tfds_name=dataset_name, tfds_data_dir=dataset_gcs_url
        ),
        preprocessors=[
            functools.partial(
                seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
            ),
            seqio.preprocessors.tokenize,
            functools.partial(
                ul2_objective,
                shard_ds=False,
                use_prefix_lm_task=True,  # use S-denoising
                rates=[0.4 / len(R_DENOISER_SPAN_LENGTHS)]
                * len(R_DENOISER_SPAN_LENGTHS)
                + [0.4 / len(X_DENOISER_SPAN_LENGTHS)] * len(X_DENOISER_SPAN_LENGTHS)
                + [0.2],
                # equal total 40% rate for both R- and X-denoisers
                # + 20% for S-denoising (suggested at the paper chapter 4.5)
                mean_noise_span_lengths=R_DENOISER_SPAN_LENGTHS
                + X_DENOISER_SPAN_LENGTHS,
                noise_densities=R_DENOISER_CORRUPT_RATES + X_DENOISER_CORRUPT_RATES,
                optional_task_prefixes=[R_DENOISER_TOKEN_PREFIX]
                * len(R_DENOISER_SPAN_LENGTHS)
                + [X_DENOISER_TOKEN_PREFIX] * len(X_DENOISER_SPAN_LENGTHS)
                + [S_DENOISER_TOKEN_PREFIX],
                reserved_for_packing=1,  # make room for task prefix token
            ),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.accuracy],
    )


def get_dataset(task_name, split="validation"):
    """_summary_"""
    dataset = seqio.get_mixture_or_task(task_name).get_dataset(
        sequence_length={"inputs": 512, "targets": 512},
        split=split,
        shuffle=False,
        num_epochs=1,
        shard_info=seqio.ShardInfo(index=0, num_shards=10),
        use_cached=False,
        seed=42,
    )
    return dataset


def write_samples_to_disk(
    dataset, vocabulary: seqio.SentencePieceVocabulary, output_filepath: str
):
    """_summary_

    Args:
        dataset (_type_): _description_
    """
    with gzip.open(output_filepath, "wb") as f:
        for idx, ex in enumerate(tqdm(dataset.as_numpy_iterator())):
            out_dict = {
                "inputs": vocabulary.decode(ex["inputs"]),
                "targets": vocabulary.decode(ex["targets"]),
            }
            print(json.dumps(out_dict).encode("utf8") + "\n", file=f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_gcs_url", type=str, required=True)
    parser.add_argument("--output_filepath", type=str, required=True)

    args = parser.parse_args()

    register_the_task(args.task_name, args.dataset_name, args.dataset_gcs_url)

    dataset = get_dataset(args.task_name)

    vocabulary = get_vocabulary(
        "SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model"
    )

    write_samples_to_disk(
        dataset=dataset, vocabulary=vocabulary, output_filepath=args.output_filepath
    )
