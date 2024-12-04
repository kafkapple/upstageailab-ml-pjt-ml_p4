import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from types import SimpleNamespace

@dataclass
class MLflowConfig:
    """MLflow 관련 설정"""
    tracking_uri: str
    experiment_name: str
    model_registry_metric_threshold: float
    mlrun_path: Path
    backend_store_uri: Path
    model_info_path: Path
    artifact_location: Path
    server_config: Dict[str, Any]

class Config:
    def __init__(self, config_path: str = "config/config.yaml"):
        """설정 초기화"""
        # 프로젝트 루트 디렉토리 찾기
        self.project_root = self._find_project_root()
        self.config_path = self.project_root / config_path
        
        # 설정 파일 로드
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        # 기본 경로 설정
        self.paths = {
            'data': self.project_root / 'data',
            'raw_data': self.project_root / 'data' / 'raw',
            'processed_data': self.project_root / 'data' / 'processed',
            'models': self.project_root / 'models',
            'logs': self.project_root / 'logs'
        }
        
        # 데이터셋별 경로 설정
        for dataset_name, dataset_config in self._config['dataset'].items():
            dataset_dir = self.paths['raw_data'] / dataset_name
            dataset_config['train_data_path'] = dataset_dir / dataset_config['train_data_path']
            dataset_config['val_data_path'] = dataset_dir / dataset_config['val_data_path']
        
        # MLflow 관련 설정
        self.mlflow = MLflowConfig(
            tracking_uri=self._config["mlflow"]["tracking_uri"],
            experiment_name=self._config["mlflow"]["experiment_name"],
            model_registry_metric_threshold=self._config["mlflow"]["model_registry_metric_threshold"],
            mlrun_path=self.project_root / self._config["mlflow"]["mlrun_path"],
            backend_store_uri=self.project_root / self._config["mlflow"]["backend_store_uri"],
            model_info_path=self.project_root / self._config["mlflow"]["model_info_path"],
            artifact_location=self.project_root / self._config["mlflow"]["artifact_location"],
            server_config=self._config["mlflow"]["server_config"]
        )
        
        # 프로젝트 설정
        self.project = self._config["project"]
        
        # 데이터셋 설정
        self.dataset = self._config["dataset"]  # 전체 데이터셋 설정
        self.data = self._config["dataset"][self.project["dataset_name"]]  # 현재 선택된 데이터셋
        
        # 모델 설정
        self.models = self._config["models"]  # 전체 모델 설정
        self.model_config = self._config["models"][self.project["model_name"]]  # 현재 선택된 모델
        
        # 공통 설정
        self.common = self._config["common"]
        
        # 체크포인트 설정
        self.checkpoint = {
            'dirpath': self.paths['logs'] / 'checkpoints',
            'filename': self._config['common']['checkpoint']['filename'],
            'monitor': self._config['common']['checkpoint']['monitor'],
            'mode': self._config['common']['checkpoint']['mode'],
            'save_top_k': self._config['common']['checkpoint']['save_top_k'],
            'save_last': self._config['common']['checkpoint']['save_last']
        }
        
        # HPO 설정
        self.hpo = self._config["hpo"]
        
        # 필요한 디렉토리 생성
        self._create_directories()
    
    def _find_project_root(self) -> Path:
        """프로젝트 루트 디렉토리 찾기"""
        current_dir = Path(__file__).resolve().parent
        while current_dir.name:
            if (current_dir / 'src').exists() or (current_dir / 'config' / 'config.yaml').exists():
                return current_dir
            current_dir = current_dir.parent
        raise RuntimeError("Project root directory not found")
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        # MLflow 디렉토리
        self.mlflow.mlrun_path.mkdir(parents=True, exist_ok=True)
        self.mlflow.backend_store_uri.mkdir(parents=True, exist_ok=True)
        self.mlflow.artifact_location.mkdir(parents=True, exist_ok=True)
        Path(self.mlflow.model_info_path).parent.mkdir(parents=True, exist_ok=True)

