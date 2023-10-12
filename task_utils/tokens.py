"""_summary_"""
import logging

import seqio

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


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


def count_tokens(task_name):
    """_summary_"""
    total_tokens = 0
    for split_name in ["training", "validation"]:
        dataset = get_dataset(task_name, split=split_name)
        for idx, ex in enumerate(dataset.as_numpy_iterator()):
            total_tokens += len(ex["text"])
            if idx % 10000 == 0:
                logger.info(
                    f"count_tokens: task_name: {task_name}, split_name: {split_name}, idx: {idx}, total_tokens: {total_tokens}"
                )
    return total_tokens
