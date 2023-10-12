import functools

import seqio
import t5.data
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics

from task_utils.ul2_objective import ul2_objective
from task_utils.tokens import get_dataset, count_tokens

# values from UL2 paper https://arxiv.org/pdf/2205.05131.pdf chapter 3.1.2 table 1
R_DENOISER_SPAN_LENGTHS = [3.0, 8.0]
X_DENOISER_SPAN_LENGTHS = [3.0, 8.0, 64.0, 64.0]
R_DENOISER_CORRUPT_RATES = [0.15, 0.15]
X_DENOISER_CORRUPT_RATES = [0.5, 0.5, 0.15, 0.5]

R_DENOISER_TOKEN_PREFIX = "[NLU]"
X_DENOISER_TOKEN_PREFIX = "[NLG]"
S_DENOISER_TOKEN_PREFIX = "[S2S]"

TaskRegistry = seqio.TaskRegistry
MixtureRegistry = seqio.MixtureRegistry

vocabulary = seqio.SentencePieceVocabulary(
    "SentencePiece_32k_Tokenizer-denoiser-tokens-added-02.model", extra_ids=100
)

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=vocabulary, add_eos=True, required=False),
    "targets": seqio.Feature(vocabulary=vocabulary, add_eos=True),
}

dataset_gcs_url = "gs://turkish-llm-data/datasets"

dataset_names = [
    ("bilkent_creative_writings", 16.0),
    ("book_corpus_v2", 8.0),
    ("dergipark", 6.0),
    ("oscarmc4_cleaned_hf_dataset", 1.0),
    ("parlamint_tr", 12.0),
    ("yoktez", 2.0),
]

dataset_versions = ["1.0.0" for _ in range(len(dataset_names))]

preprocessing_pipeline = [
    functools.partial(
        seqio.preprocessors.rekey, key_map={"inputs": None, "targets": "text"}
    ),
    seqio.preprocessors.tokenize,
    functools.partial(
        ul2_objective,
        shard_ds=False,
        use_prefix_lm_task=True,  # use S-denoising
        # equal total 40% rate for both R- and X-denoisers + 20% for S-denoising
        # (suggested at the paper chapter 4.5)
        rates=[0.4 / len(R_DENOISER_SPAN_LENGTHS)] * len(R_DENOISER_SPAN_LENGTHS)
        + [0.4 / len(X_DENOISER_SPAN_LENGTHS)] * len(X_DENOISER_SPAN_LENGTHS)
        + [0.2],
        mean_noise_span_lengths=R_DENOISER_SPAN_LENGTHS + X_DENOISER_SPAN_LENGTHS,
        noise_densities=R_DENOISER_CORRUPT_RATES + X_DENOISER_CORRUPT_RATES,
        optional_task_prefixes=[R_DENOISER_TOKEN_PREFIX] * len(R_DENOISER_SPAN_LENGTHS)
        + [X_DENOISER_TOKEN_PREFIX] * len(X_DENOISER_SPAN_LENGTHS)
        + [S_DENOISER_TOKEN_PREFIX],
        reserved_for_packing=1,  # make room for task prefix token
    ),
    seqio.preprocessors.append_eos_after_trim,
]

for dataset_name, version in zip(dataset_names, dataset_versions):
    TaskRegistry.add(
        f"pretrain_{dataset_name}",
        source=seqio.TfdsDataSource(
            tfds_name=dataset_name, tfds_data_dir=dataset_gcs_url
        ),
        preprocessors=preprocessing_pipeline,
        output_features=DEFAULT_OUTPUT_FEATURES,
        metric_fns=[metrics.accuracy],
    )

    TaskRegistry.add(
        f"count_{dataset_name}",
        source=seqio.TfdsDataSource(
            tfds_name=dataset_name, tfds_data_dir=dataset_gcs_url
        ),
        preprocessors=[
            functools.partial(
                seqio.preprocessors.tokenize,
                output_features={
                    "text": seqio.Feature(
                        vocabulary=vocabulary, add_eos=False, required=False
                    )
                },
            )
        ],
        output_features={
            "text": seqio.Feature(vocabulary=vocabulary, add_eos=False, required=False)
        },
    )

MixtureRegistry.add(
    "pretrain_all",
    [(f"pretrain_{dataset_name}", rate) for dataset_name, rate in dataset_names],
    default_rate=1.0,
)

n_tokens = count_tokens("count_bilkent_creative_writings")
print(n_tokens)
