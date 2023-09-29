"""dergipark dataset.
Original tfds builder folder in /media/disk/datasets/bounllm/tfds/dergipark"""

import tensorflow_datasets as tfds
import gzip
import numpy as np
import json
from pathlib import Path


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for dergipark dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = (
      "Put dergipark articles ('_no_inline_citations.txt') in the manual_dir / 'no_inline_txt' "
  )

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(dergipark): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "id": tfds.features.Scalar(dtype=np.int64),
                    "text": tfds.features.Text(),
                    "corpus": tfds.features.Text(),
                    "article": tfds.features.Text(),
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
    # TODO(dergipark): Downloads the data and defines the splits
    filepath = Path(dl_manager.manual_dir) 

    # TODO(dergipark): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        "train": self._generate_examples(filepath, "train"),
        "validation": self._generate_examples(filepath, "val"),
    }

  def _generate_examples(self, path, split):
    """Yields examples."""
    # TODO(dergipark): Yields (key, example) tuples from the dataset

    file_list_path = path / (split + ".txt")
    with open(str(file_list_path)) as f:
        files = [l.strip() for l in f.readlines()]
    
    for idx, file in enumerate(files):
        file_path = path / "no_inline_txt" / file
        with open(file_path) as f:
            line = f.read().strip()
        yield idx, {
            "id": idx,
            "text": line,
            "corpus": "dergipark",
            "article": file,
        }

# tfds build --manual_dir /media/disk/datasets/bounllm/dergipark/dergipark-090920230005 --data_dir /media/disk/datasets/bounllm/tfds/datasets/dergipark