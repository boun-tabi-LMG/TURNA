import functools
import seqio
import tensorflow as tf
import t5.data
from datasets import load_dataset, load_from_disk
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics
from seqio import FunctionDataSource, utils

from ul2_objective import ul2_objective

# values from UL2 paper https://arxiv.org/pdf/2205.05131.pdf chapter 3.1.2 table 1
R_DENOISER_SPAN_LENGTHS = [3.0, 8.0]
X_DENOISER_SPAN_LENGTHS = [3.0, 8.0, 64.0, 64.0]
R_DENOISER_CORRUPT_RATES = [0.15, 0.15]
X_DENOISER_CORRUPT_RATES = [0.5, 0.5, 0.15, 0.5]

R_DENOISER_TOKEN_PREFIX = "[NLU]"
X_DENOISER_TOKEN_PREFIX = "[NLG]"
S_DENOISER_TOKEN_PREFIX = "[S2S]"

TaskRegistry = seqio.TaskRegistry

vocabulary = seqio.SentencePieceVocabulary(
    "SentencePiece_32k_Tokenizer-denoiser-tokens-added.model", extra_ids=0
)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=vocabulary, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=vocabulary, add_eos=True),
}


dataset_name = "oscarmc4_cleaned_hf_dataset_subset_combined_tfds:1.0.0"
dataset_gcs_url = "gs://turkish-llm-data/"

TaskRegistry.add(
    "pretrain_turkish_ul2",
    source=seqio.TfdsDataSource(tfds_name=dataset_name, tfds_data_dir=dataset_gcs_url),
    preprocessors=[
        functools.partial(
            seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
        ),
        seqio.preprocessors.tokenize,
        functools.partial(
            ul2_objective,
            shard_ds=False,
            use_prefix_lm_task=True,  # use S-denoising
            rates=[0.4 / len(R_DENOISER_SPAN_LENGTHS)] * len(R_DENOISER_SPAN_LENGTHS)
            + [0.4 / len(X_DENOISER_SPAN_LENGTHS)] * len(X_DENOISER_SPAN_LENGTHS)
            + [
                0.2
            ],  # equal total 40% rate for both R- and X-denoisers + 20% for S-denoising (suggested at the paper chapter 4.5)
            mean_noise_span_lengths=R_DENOISER_SPAN_LENGTHS + X_DENOISER_SPAN_LENGTHS,
            noise_densities=R_DENOISER_CORRUPT_RATES + X_DENOISER_CORRUPT_RATES,
            optional_task_prefixes=[R_DENOISER_TOKEN_PREFIX]
            * len(R_DENOISER_SPAN_LENGTHS)
            + [X_DENOISER_TOKEN_PREFIX] * len(X_DENOISER_SPAN_LENGTHS)
            + [S_DENOISER_TOKEN_PREFIX],
            reserved_for_packing=1,  # make room for task prefix token
        ),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": DEFAULT_OUTPUT_FEATURES["targets"]},
    metric_fns=[metrics.accuracy],
)
