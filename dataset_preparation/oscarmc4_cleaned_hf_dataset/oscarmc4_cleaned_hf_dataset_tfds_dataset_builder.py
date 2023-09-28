"""oscarmc4_cleaned_hf_dataset_tfds dataset."""

import tensorflow_datasets as tfds
import gzip
import numpy as np
import json
from pathlib import Path


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for oscarmc4_cleaned_hf_dataset_tfds dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }
    MANUAL_DOWNLOAD_INSTRUCTIONS = (
        "Put train.txt.gz and validation.txt.gz in the manual_dir"
    )

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
        train_filepath = Path(dl_manager.manual_dir) / "train.txt.gz"
        validation_filepath = Path(dl_manager.manual_dir) / "validation.txt.gz"

        # TODO(oscarmc4_cleaned_hf_dataset_tfds): Returns the
        # Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(train_filepath),
            "validation": self._generate_examples(validation_filepath),
        }

    def _generate_examples(self, filepath):
        """Yields examples."""
        # TODO(oscarmc4_cleaned_hf_dataset_tfds): Yields (key, example)
        # tuples from the dataset
        with gzip.open(filepath, "rb") as f:
            line = f.readline()
            idx = 0
            while line:
                item = json.loads(line)
                yield idx, {
                    "id": item["id"],
                    "text": item["text"],
                    "corpus": item["corpus"],
                }
                idx += 1
                line = f.readline()
