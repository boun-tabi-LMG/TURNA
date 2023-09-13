"""Code for debugging the loaded dataset and vocabulary."""

# dataset = seqio.get_mixture_or_task("pretrain_turkish_ul2").get_dataset(
#      sequence_length={"inputs": 512, "targets": 512},
#      split="train",
#      shuffle=True,
#      num_epochs=1,
#      shard_info=seqio.ShardInfo(index=0, num_shards=10),
#      use_cached=False,
#      seed=42
# )

# Print the first 5 examples.
# examples = [ex for _, ex in zip(range(10), dataset.as_numpy_iterator())]
#
# print(
#     "vocabulary.extra_ids, vocabulary.bos_id, vocabulary.eos_id, vocabulary.pad_id, vocabulary.unk_id"
# )
# print(
#     vocabulary.extra_ids,
#     vocabulary.bos_id,
#     vocabulary.eos_id,
#     vocabulary.pad_id,
#     vocabulary.unk_id,
# )
#
# for key, value in [(i, vocabulary.decode([i])) for i in range(vocabulary.vocab_size)]:
#     print("Token: {key} -- {value}".format(key=key, value=value))
#
# for idx, ex in enumerate(examples):
#     print(f"Idx: {idx} Inputs: {ex['inputs'].size} Targets: {ex['targets'].size}")
#     print(vocabulary.decode(ex['inputs']))
#     for token in ex['inputs']:
#         print(token)
#         print(vocabulary.decode([token]))
#     print("\n")
#
# print(examples[0])
# import sys
# sys.exit(1)
