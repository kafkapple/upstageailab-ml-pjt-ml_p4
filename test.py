import sys

from src.config import Config
from src.utils.data_utils import split_dataset
config = Config()

def split_dataset():
    split_files = split_dataset(
        file_path=config.dataset[config.project['dataset_name']]['val_data_path'],
    n_splits=5,
    seed=42
    )
    print(split_files)


def restore_model_registry():
    from src.utils.mlflow_utils import MLflowModelManager

        # MLflow model registry 복구
    model_manager = MLflowModelManager(config)
    model_manager.restore_model_registry()


restore_model_registry()