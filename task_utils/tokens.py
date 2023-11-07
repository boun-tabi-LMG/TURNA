"""_summary_"""
import logging

import seqio

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def get_dataset(task_name, split="validation", counting=False):
    """_summary_"""
    if counting:
        sequence_length = None
    else:
        sequence_length = {"inputs": 512, "targets": 512}
    dataset = seqio.get_mixture_or_task(task_name).get_dataset(
        sequence_length=sequence_length,
        split=split,
        shuffle=False,
        num_epochs=1,
        # shard_info=seqio.ShardInfo(index=0, num_shards=10),
        use_cached=False,
        seed=42,
    )
    return dataset


def count_tokens(task_name):
    """_summary_"""
    total_tokens = 0
    for split_name in ["train", "validation"]:
        dataset = get_dataset(task_name, split=split_name, counting=True)
        for idx, ex in enumerate(dataset.as_numpy_iterator()):
            total_tokens += len(ex["text"])
            if idx % 10000 == 0:
                logger.info(
                    f"count_tokens: task_name: {task_name}, "
                    f"split_name: {split_name}, idx: {idx}, "
                    f"total_tokens: {total_tokens}"
                )
    return total_tokens


def count_tokens_with_seqio():
    """_summary_"""
    n_tokens = {}
    for task_name in [f"count_{dataset_name}" for dataset_name, _ in dataset_names]:
        n_tokens[task_name] = count_tokens(task_name)
        logger.info(f"task_name: {task_name}, n_tokens: {n_tokens[task_name]}")
    logger.info(f"n_tokens: {n_tokens}")
