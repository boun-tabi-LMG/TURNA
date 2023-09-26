import tensorflow_datasets as tfds
from pathlib import Path
import numpy as np

class BilkentCreativeWritingsBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Bilkent Creative Writings dataset."""
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {'1.0.0': 'Initial release.'}
    MANUAL_DOWNLOAD_INSTRUCTIONS = "Put writings in the manual_dir and train.txt and val.txt in the parent directory"
    CORPUS_NAME = "bilkent-creative-writings"
    FILE_LIST_PARENT = True

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return self.dataset_info_from_configs(
                features=tfds.features.FeaturesDict(
                    {
                        "id": tfds.features.Scalar(dtype=np.int64),
                        "text": tfds.features.Text(),
                        "corpus": tfds.features.Text(),
                        "article": tfds.features.Text(),
                    }
                ),
                supervised_keys=None,  # Set to `None` to disable
                homepage="https://dataset-homepage/",
            )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        filepath = Path(dl_manager.manual_dir)
        return {
            "train": self._generate_examples(filepath, "train"),
            "validation": self._generate_examples(filepath, "val"),
        }

    def _generate_examples(self, path, split):
        """Yields examples."""
        file_list_path = path.parent if self.FILE_LIST_PARENT else path
        file_list_path = file_list_path / f"{split}.txt"
        with open(file_list_path) as f:
            files = [l.strip() for l in f.readlines()]

        for idx, file in enumerate(files):
            file_path = path / file
            with open(file_path) as f:
                line = f.read().strip()
            yield idx, {
                "id": idx,
                "text": line,
                "corpus": self.CORPUS_NAME,
                "article": file,
            }

# tfds build  --manual_dir /media/disk/datasets/bounllm/bilkent-creative-writings/texts_clean/ --data_dir /media/disk/datasets/bounllm/tfds/datasets/bilkent_creative_writings