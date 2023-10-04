"""oscarmc4_cleaned_hf_dataset_tfds dataset."""

from pathlib import Path
import random
import pyarrow as pa
import tensorflow_datasets as tfds
import numpy as np


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for oscarmc4_cleaned_hf_dataset_tfds dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Put dataset.arrow in the manual_dir"

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(oscarmc4_cleaned_hf_dataset_tfds):
        # Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "id": tfds.features.Scalar(dtype=np.int64),
                    "text": tfds.features.Text(),
                    "corpus": tfds.features.Text(),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage="https://dataset-homepage/",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(oscarmc4_cleaned_hf_dataset_tfds): Downloads the data
        # and defines the splits
        main_filepath = Path(dl_manager.manual_dir) / "dataset.arrow"
        main_filepath = str(main_filepath)
        num_examples = 50336214

        random.seed(42)
        selected_indices = list(range(num_examples))
        random.shuffle(selected_indices)
        n_validation = int(num_examples * 0.00001)

        train_indices_set = set(selected_indices[n_validation:])
        validation_indices_set = set(selected_indices[:n_validation])

        # TODO(oscarmc4_cleaned_hf_dataset_tfds): Returns the
        # Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(main_filepath, "train", train_indices_set),
            "validation": self._generate_examples(
                main_filepath, "validation", validation_indices_set
            ),
        }

    def _generate_examples(self, main_filepath, split_name, selected_indices_set):
        """Yields examples."""
        # TODO(oscarmc4_cleaned_hf_dataset_tfds): Yields (key, example)
        # tuples from the dataset

        with pa.memory_map(main_filepath) as mms:
            open_s = pa.ipc.open_stream(mms)
            idx = 0
            try:
                b = open_s.read_next_batch()
                while b:
                    n_rows = b.num_rows
                    batch_ids_set = set(range(idx, idx + n_rows))
                    ids_to_extract = batch_ids_set.intersection(selected_indices_set)
                    b_dict = b.to_pydict()
                    for target_id in ids_to_extract:
                        yield target_id, {
                            "id": b_dict["id"][target_id - idx],
                            "text": b_dict["text"][target_id - idx],
                            "corpus": b_dict["corpus"][target_id - idx],
                        }
                    idx += n_rows
                    b = open_s.read_next_batch()
            except StopIteration:
                pass

        # with gzip.open(filepath, "rb") as f:
        #     line = f.readline()
        #     idx = 0
        #     while line:
        #         item = json.loads(line)
        #         yield idx, {
        #             "id": item["id"],
        #             "text": item["text"],
        #             "corpus": item["corpus"],
        #         }
        #         idx += 1
        #         line = f.readline()
