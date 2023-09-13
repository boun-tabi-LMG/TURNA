import gzip
import json
from datasets import load_from_disk


def export_as_json_a_single_document_in_each_line_as_json():
    """Example for preparing our dataset as required by the TFDS builder."""
    ds = load_from_disk(
        "/media/disk/datasets/bounllm/oscarmc4_cleaned_hf_dataset_subset_combined"
    )

    for split_name in ["train", "validation"]:
        with gzip.open(
            f"/media/disk/datasets/bounllm/"
            f"oscarmc4_cleaned_hf_dataset_subset_combined/{split_name}.txt.gz",
            "wb",
        ) as f:
            for item in ds[split_name].to_iterable_dataset():
                out = (
                    json.dumps(
                        {
                            "id": item["id"],
                            "text": item["text"],
                            "corpus": item["corpus"],
                        }
                    )
                    + "\n"
                )
                f.write(out.encode("utf8"))


def tfds_things():
    """I used this function to test TFDS things.

    Returns:
        _type_: _description_
    """
    import tensorflow_datasets as tfds

    import tensorflow as tf

    tfrecord_files = [
        "/media/disk/datasets/bounllm/tfds/oscarmc4_cleaned_hf_dataset_subset_combined_tfds/oscarmc4_cleaned_hf_dataset_subset_combined_tfds-validation.tfrecord-00031-of-00032"
    ]
    dataset = tf.data.TFRecordDataset(filenames=tfrecord_files)
    dataset
    el = dataset.take(1)
    el
    for x in el:
        print(x)
    tfrecord_files = [
        "/media/disk/datasets/bounllm/tfds/datasets/oscarmc4_cleaned_hf_dataset_subset_combined_tfds/1.0.0/oscarmc4_cleaned_hf_dataset_subset_combined_tfds-validation.tfrecord-00031-of-00032"
    ]
    dataset = tf.data.TFRecordDataset(filenames=tfrecord_files)
    for x in dataset.take(1):
        print(x)
    list(dataset.take(1))
    l = list(dataset.take(1))
    l

    def decode_fn(record_bytes):
        return tf.io.parse_single_example(
            record_bytes,
            {
                "id": tf.io.FixedLenFeature([], dtype=tf.int64),
                "text": tf.io.FixedLenFeature([], dtype=tf.string),
                "corpus": tf.io.FixedLenFeature([], dtype=tf.string),
            },
        )

    for x in dataset.take(1).map(decode_fn):
        print(x)

    ds_tfds = tfds.load(
        "oscarmc4_cleaned_hf_dataset_subset_combined_tfds:1.0.0",
        "gs://turkish-llm-data",
    )
