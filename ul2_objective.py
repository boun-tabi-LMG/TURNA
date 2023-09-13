import functools
import tensorflow as tf
import seqio
import t5.data
from typing import Optional, Sequence

# found this function and modified from https://github.com/GoogleCloudPlatform/t5x-on-vertex-ai/blob/main/tasks/custom_tasks.py#L78
# UL2 paper appendix code missed this function
def prepend_prompt(dataset: tf.data.Dataset,
                   output_features: seqio.preprocessors.OutputFeaturesType,
                   sequence_length: Optional[
                       seqio.preprocessors.SequenceLengthType] = None,
                   prompt_mode: str = "",
                   key: str = "inputs",
                   mode: str = "") -> tf.data.Dataset:
    """Prepends a prompt at the beginning of an input sequence."""
    del sequence_length
    if prompt_mode and mode:
        # output_features may not have inputs key
        out_keys = list(output_features.keys())
        prompt_tokens = output_features[out_keys[0]
                                        ].vocabulary.encode_tf(prompt_mode)

        def add_to_inputs(x):
            x[key] = tf.concat([prompt_tokens, x[key]], axis=0)
            return x

        dataset = dataset.map(add_to_inputs)
    return dataset

# modified from t5.data.preprocessors because output_features may not have inputs key
def split_tokens_to_inputs_length(dataset, sequence_length,
                                  output_features, **kwargs):
    max_tokens = sequence_length['inputs']
    # output_features may not have inputs key
    out_keys = list(output_features.keys())
    if output_features[out_keys[0]].add_eos:
        # Leave room to insert an EOS token.
        max_tokens -= 1

    return t5.data.preprocessors.split_tokens(dataset, max_tokens_per_segment=max_tokens, **kwargs)

# modified from t5.data.preprocessors because output_features may not have inputs key
def prefix_lm(dataset, sequence_length, output_features):
    """Prefix language modeling objective used in Raffel et al. 2019."""
    ds = dataset
    ds = t5.data.preprocessors.select_random_chunk(ds, output_features=output_features,
                                                   feature_key='targets', max_length=65536)
    ds = split_tokens_to_inputs_length(ds, output_features=output_features,
                                       sequence_length=sequence_length)
    ds = t5.data.preprocessors.denoise(
        ds,
        output_features,
        inputs_fn=t5.data.preprocessors.drop_nonnoise_tokens,
        targets_fn=t5.data.preprocessors.drop_noise_tokens,
        noise_density=0.5,
        noise_mask_fn=t5.data.preprocessors.random_prefix_noise_mask,
    )
    return ds

# copied from UL2 paper https://arxiv.org/pdf/2205.05131.pdf appendix chapter 9.2
# note: modified to use the prefix_lm() from above instead of the default t5.data.preprocessors.prefix_lm() because output_features may not have inputs key
def ul2_objective(dataset: tf.data.Dataset,
                  sequence_length: seqio.preprocessors.SequenceLengthType,
                  output_features: seqio.preprocessors.OutputFeaturesType,
                  use_prefix_lm_task: bool = False,
                  rates: Optional[Sequence[float]] = None,
                  mean_noise_span_lengths: Sequence[float] = (3.0,),
                  noise_densities: Sequence[float] = (0.15,),
                  shard_ds: bool = True,
                  optional_task_prefixes: Optional[Sequence[str]] = None,
                  input_feature_key: str = "inputs",
                  merge_examples_to_reduce_padding: bool = True,
                  reserved_for_packing: bool = None,
                  seed: int = 7) -> tf.data.Dataset:
    """UL2-like pre-training objectives.
    This preprocessor amounts to calling the 'span_corruption' function several
    times with different values of 'noise_density' and 'mean_noise_span_length'.
    We either shard or copy the dataset, then apply each function to each shard.
    Add S-denoising (prefixLM) using use_prefix_lm_task.
    Args:
        dataset: A tf.data.Dataset with dictionaries containing the key 'input_feature_key'.
        sequence_length: dict mapping of feature key to int length for that feature.
        output_features: mapping of keys to features.
        use_prefix_lm_task: <bool> If True, include PrefixLM in the task mix.
        rates: <Optional<List<float>> List of rates per task. If None, tasks are sampled uniformly.
        mean_noise_span_lengths: List of mean number of tokens per masked span per example.
        noise_densities: List of what fraction of the tokens to mask.
        shard_ds: <bool> If True, shard dataset per objective.
        optional_task_prefixes: <Optional<list<str>> Strings to prepend for each corruption scheme. NOTE: If including prefixLM task, it must be the last prefix.
        input_feature_key: which feature to use from the dataset as the input text tokens.
        merge_examples_to_reduce_padding: if True, combines multiple input examples to reduce padding.
        reserved_for_packing: if specified, reduces the desired inputs length by the specified amount to enable multiple examples to be packed together downstream.
        seed: tf.int64 for controlling the random choice of spans.
    Returns:
        a dataset
    """

    if optional_task_prefixes:  # Ensure each task has a prefix.
        num_tasks = len(noise_densities) + int(use_prefix_lm_task)
        valid_number_of_prefixes = num_tasks == len(optional_task_prefixes)
        if not valid_number_of_prefixes:
            raise ValueError(
                "Number of task prefixes must match number of tasks.")
    inputs_length = sequence_length[input_feature_key]
    input_lengths, targets_lengths = [], []
    sequence_lengths = {x: y for x, y in sequence_length.items()}
    if reserved_for_packing:
        inputs_length -= reserved_for_packing
        for x, y in sequence_length.items():
            sequence_lengths[x] = y - reserved_for_packing
    hyperparams = list(zip(mean_noise_span_lengths, noise_densities))
    for mean_noise_span_length, noise_density in hyperparams:
        input_length, targets_length = t5.data.preprocessors.random_spans_helper(
            extra_tokens_per_span_inputs=1,
            extra_tokens_per_span_targets=1,
            inputs_length=inputs_length,
            mean_noise_span_length=mean_noise_span_length,
            noise_density=noise_density,
            verbose=True)
        input_lengths.append(input_length)
        targets_lengths.append(targets_length)

        if sequence_length["targets"] < targets_length:
            upper_bound = max(targets_lengths)
            raise ValueError(
                f'Expected max targets length for span corruption ({upper_bound}) is '
                f'greater than configured targets length '
                f"({sequence_length['targets']})")
    print("input_lengths: ", input_lengths)
    print("targets_lengths: ", targets_lengths)
    ds = dataset
    ds = t5.data.preprocessors.select_random_chunk(
        ds,
        output_features=output_features,
        feature_key="targets",
        max_length=65536)
    if merge_examples_to_reduce_padding:
        ds = t5.data.preprocessors.reduce_concat_tokens(
            ds, feature_key="targets", batch_size=128)
    num_shards = len(input_lengths) + int(use_prefix_lm_task)
    if shard_ds:
        ds_shards = [ds.shard(num_shards, i) for i in range(num_shards)]
    else:
        ds_shards = [ds for _ in range(num_shards)]
    processed_ds = []
    hyperparams = zip(input_lengths, hyperparams, range(num_shards))
    for input_length, (noise_span_length, noise_density), i in hyperparams:
        ds = ds_shards[i]
        ds = t5.data.preprocessors.split_tokens(
            ds,
            feature_key="targets",
            min_tokens_per_segment=None,
            max_tokens_per_segment=input_length)
        ds = t5.data.preprocessors.denoise(
            ds,
            output_features,
            inputs_fn=t5.data.preprocessors.noise_span_to_unique_sentinel,
            targets_fn=t5.data.preprocessors.nonnoise_span_to_unique_sentinel,
            noise_density=noise_density,
            noise_mask_fn=functools.partial(
                t5.data.preprocessors.random_spans_noise_mask,
                mean_noise_span_length=noise_span_length),
            input_feature_key=input_feature_key)
        if optional_task_prefixes:
            ds = prepend_prompt(
                ds,
                output_features,
                prompt_mode=optional_task_prefixes[i],
                mode=optional_task_prefixes[i],
                key=input_feature_key)
        processed_ds.append(ds)
    if use_prefix_lm_task:
        ds = ds_shards[-1]
        ds = prefix_lm(
            ds, sequence_lengths, output_features)
        if optional_task_prefixes:
            ds = prepend_prompt(
                ds,
                output_features,
                prompt_mode=optional_task_prefixes[-1],
                mode=optional_task_prefixes[-1],
                key=input_feature_key)
        processed_ds.append(ds)
    ds = tf.data.experimental.sample_from_datasets(processed_ds, rates, seed)
    return ds
