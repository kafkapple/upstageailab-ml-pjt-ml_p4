import sys

from src.config import Config
from src.utils.data_utils import split_dataset
config = Config()

split_files = split_dataset(
    file_path=config.dataset[config.project['dataset_name']]['val_data_path'],
    n_splits=5,
    seed=42
)
print(split_files)

