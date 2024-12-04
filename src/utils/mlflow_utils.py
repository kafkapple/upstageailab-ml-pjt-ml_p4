import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import pandas as pd
import torch
import mlflow
import shutil
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
import sys
import os

from src.config import Config

class ModelAlias(Enum):
    """모델 별칭 정의"""
    NONE = "latest"
    STAGING = "candidate"
    PRODUCTION = "champion"
    ARCHIVED = "archived"

def cleanup_old_runs(config, days_to_keep=7):
    """오래된 MLflow 실행 정리"""
    try:
        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
        
        if experiment is None:
            print(f"No experiment found with name: {config.mlflow.experiment_name}")
            return
            
        # 실험의 모든 실행 가져오기
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        # 기준 시간 계산
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # 오래된 실행 삭제
        for run in runs:
            run_date = datetime.fromtimestamp(run.info.start_time / 1000.0)
            if run_date < cutoff_date:
                client.delete_run(run.info.run_id)
                print(f"Deleted run: {run.info.run_id} from {run_date}")
                
    except Exception as e:
        print(f"Error cleaning up old runs: {str(e)}")

def cleanup_artifacts(config, metrics: Dict[str, float], run_id: str):
    """MLflow 아티팩트 정리
    
    Args:
        config: 설정 객체
        metrics: 평가 지표
        run_id: MLflow 실행 ID
    """
    try:
        # 성능이 좋지 않은 실행의 아티팩트 삭제
        if metrics.get('val_f1', 0) < config.mlflow.model_registry_metric_threshold:
            print(f"\nRemoving artifacts for run {run_id} due to low performance...")
            artifact_path = Path(config.mlflow.artifact_location) / run_id
            if artifact_path.exists():
                shutil.rmtree(str(artifact_path))
                print(f"Removed artifacts at: {artifact_path}")
    except Exception as e:
        print(f"Error cleaning up artifacts: {str(e)}")

def setup_mlflow_server(config: Config, reset_experiments: bool = False):
    """MLflow 서버 설정
    
    Args:
        config: 설정 객체
        reset_experiments: MLflow 실험 데이터 초기화 여부 (기본값: False)
    """
    # 절대 경로 사용
    root_dir = Path.cwd().resolve()
    
    # MLflow 디렉토리 초기화
    mlruns_dir = root_dir / 'mlruns'
    mlartifacts_dir = root_dir / 'mlartifacts'
    trash_dir = mlruns_dir / ".trash"
    
    print(f"\nDebug: Setting up MLflow directories:")
    print(f"Root dir: {root_dir}")
    print(f"MLruns dir: {mlruns_dir}")
    print(f"Artifacts dir: {mlartifacts_dir}")
    
    # 실험 데이터 초기화 (선택적)
    if reset_experiments:
        print("Resetting MLflow experiments...")
        if mlruns_dir.exists():
            shutil.rmtree(mlruns_dir)
        if mlartifacts_dir.exists():
            shutil.rmtree(mlartifacts_dir)
    
    # 디렉토리 생성
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlartifacts_dir.mkdir(parents=True, exist_ok=True)
    trash_dir.mkdir(parents=True, exist_ok=True)
    
    # MLflow 설정
    registry_uri = f"file://{mlruns_dir}"
    os.environ['MLFLOW_TRACKING_URI'] = config.mlflow.tracking_uri
    os.environ['MLFLOW_REGISTRY_URI'] = registry_uri
    
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    
    print(f"\nDebug: MLflow Configuration:")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Registry URI: {mlflow.get_registry_uri()}")
    
    # 모델 레지스트리 초기화 (선택적)
    if reset_experiments:
        client = MlflowClient()
        try:
            registered_models = client.search_registered_models()
            for model in registered_models:
                client.delete_registered_model(model.name)
        except:
            pass
    
    print(f"Debug: MLflow environment initialized")
    print(f"Debug: MLflow metadata directory: {mlruns_dir}")
    print(f"Debug: MLflow artifacts directory: {mlartifacts_dir}")

def initialize_mlflow(config: Config) -> str:
    """MLflow 초기화 및 설정
    
    Args:
        config: 설정 객체
        
    Returns:
        str: experiment_id
    """
    # MLflow 실험 설정
    experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None:
        
        experiment_id = mlflow.create_experiment(
            name=config.mlflow.experiment_name,
            artifact_location=str(config.mlflow.artifact_location)
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    print(f"Debug: MLflow initialized:")
    print(f"Debug: Experiment name: {config.mlflow.experiment_name}")
    print(f"Debug: Experiment ID: {experiment_id}")
    print(f"Debug: Artifact location: {config.mlflow.artifact_location}")
    
    return experiment_id

class MLflowModelManager:
    def __init__(self, config: Config):
        """MLflow 모델 관리자 초기화
        
        Args:
            config: 설정 객체
        """
        self.config = config
        self.model_info_path = Path(config.mlflow.model_info_path)
        
        # MLflow 클라이언트 설정
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        
        print(f"Debug: MLflow Model Manager initialized")
        print(f"Debug: Model info path: {self.model_info_path}")
        print(f"Debug: Tracking URI: {config.mlflow.tracking_uri}")
        
    def register_model(self, model_name: str, run_id: str, model_uri: str = 'model') -> ModelVersion:
        """MLflow에 모델을 등록하고 버전 정보를 반환"""
        # MLflow에 델 등록
        model_uri = f"runs:/{run_id}/{model_uri}"
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            print(f"Registered model '{model_name}' version {model_version.version}")
            
            # 모델 버전 정보 반환
            return model_version
            
        except Exception as e:
            print(f"Error registering model: {str(e)}")
            raise
    
    def promote_to_staging(self, model_name: str, run_id: str, model_uri: str = 'model') -> ModelVersion:
        """모델을 Staging(candidate) 단계로 승격"""
        try:
            print(f"\n=== Debug: Promoting model to Candidate ===")
            print(f"Debug: Model name: {model_name}")
            print(f"Debug: Run ID: {run_id}")
            
            # 1. 먼저 모델 등록 (없는 경우)
            versions = self.client.search_model_versions(f"name='{model_name}'")
            model_version = next(
                (v for v in versions if v.run_id == run_id), 
                None
            )
            
            if not model_version:
                print(f"Debug: Registering new model version for run_id: {run_id}")
                # MLflow에 모델 등록
                full_uri = f"runs:/{run_id}/{model_uri}"
                model_version = mlflow.register_model(full_uri, model_name)
                print(f"Debug: Registered as version: {model_version.version}")
            else:
                print(f"Debug: Found existing version: {model_version.version}")
            
            # 2. 기존 alias 제거
            try:
                current_candidate = self.client.get_model_version_by_alias(
                    name=model_name,
                    alias=ModelAlias.STAGING.value
                )
                if current_candidate:
                    print(f"Debug: Removing candidate alias from version: {current_candidate.version}")
                    self.client.delete_registered_model_alias(
                        name=model_name,
                        alias=ModelAlias.STAGING.value
                    )
            except Exception as e:
                print(f"Debug: No current candidate found: {str(e)}")
            
            # 3. 새로운 candidate 설정
            print(f"Debug: Setting new candidate version: {model_version.version}")
            self.client.set_registered_model_alias(
                name=model_name,
                alias=ModelAlias.STAGING.value,
                version=model_version.version
            )
            
            # 4. 동기화
            self.sync_model_info()
            
            print(f"Debug: Model {model_name} version {model_version.version} promoted to Candidate")
            return model_version
            
        except Exception as e:
            print(f"Error promoting model to staging: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def promote_to_production(self, model_name: str, version: str) -> None:
        """모델을 Champion으로 승격"""
        try:
            print(f"\n=== Debug: Promoting model to Champion ===")
            self._set_model_alias(model_name, version, ModelAlias.PRODUCTION.value)
        except Exception as e:
            print(f"Error promoting model to champion: {str(e)}")
            raise
    
    def archive_model(self, model_name: str, version: str) -> None:
        """모델을 Archive 단계로 이동"""
        try:
            print(f"\n=== Debug: Archiving model ===")
            self._set_model_alias(model_name, version, ModelAlias.ARCHIVED.value)
        except Exception as e:
            print(f"Error archiving model: {str(e)}")
            raise
    
    def get_latest_versions(self, model_name: str, aliases: Optional[List[str]] = None) -> List[ModelVersion]:
        """특정 alias의 최신 모델 버전들을 조회"""
        try:
            if aliases is None:
                aliases = [alias.value for alias in ModelAlias]
            
            versions = []
            for alias in aliases:
                try:
                    version = self.client.get_model_version_by_alias(
                        name=model_name,
                        alias=alias
                    )
                    if version:
                        versions.append(version)
                except Exception:
                    continue
            return versions
        except Exception as e:
            print(f"Error getting latest versions: {str(e)}")
            return []
    
    def save_model_info(self, run_id: str, metrics: Dict[str, float], params: Dict[str, Any], version: str) -> None:
        """모델 정보를 JSON 파일로 저장"""
        try:
            # 항상 새로운 모델은 Staging(candidate)으로 시작
            current_alias = ModelAlias.STAGING.value
            
            # 모델 설정 가져오기
            model_config = self.config.models[self.config.project['model_name']]
            
            # 중복 제거된 파라미터 구성
            model_params = {
                'model_name': self.config.project['model_name'],
                'dataset_name': self.config.project['dataset_name'],
                'pretrained_model': model_config['pretrained_model'],
                'training': model_config['training']  # 학습 관련 파라미터는 training 하위로 이동
            }
            
            # 추가 파라미터가 있다면 training 하위로 이동
            if params:
                model_params['training'].update({
                    k: v for k, v in params.items() 
                    if k not in ['model_name', 'dataset_name', 'pretrained_model']
                })
            
            model_info = {
                "experiment_name": mlflow.get_experiment(mlflow.get_run(run_id).info.experiment_id).name,
                "experiment_id": mlflow.get_run(run_id).info.experiment_id,
                "run_id": run_id,
                "run_name": f"{self.config.project['model_name']}_{self.config.project['dataset_name']}",
                "metrics": metrics,
                "params": model_params,  # 중복 제거된 파라미터
                "stage": current_alias,
                "version": version,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            model_infos = self.load_model_info()
            model_infos.append(model_info)
            
            with open(self.model_info_path, 'w', encoding='utf-8') as f:
                json.dump(model_infos, f, indent=2, ensure_ascii=False)
                
            print(f"Model info saved successfully as {current_alias}")
            
        except Exception as e:
            print(f"Error saving model info: {str(e)}")
            raise
    
    def load_model_info(self) -> List[Dict]:
        """저장된 모델 정보를 JSON 파일에서 로드"""
        try:
            if self.model_info_path.exists():
                with open(self.model_info_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading model info: {str(e)}")
            return []
    
    def display_models(self) -> None:
        """저장된 모델 정보를 테이블 형태로 출력"""
        model_infos = self.load_model_info()
        if not model_infos:
            print("No models found in registry.")
            return
        
        df = pd.DataFrame(model_infos)
        display_columns = [
            "experiment_name",
            "run_name",
            "run_id",
            "metrics",
            "stage",
            "timestamp"
        ]
        df = df[display_columns]
        
        # metrics 컬럼을 보기 좋게 포맷팅
        df['metrics'] = df['metrics'].apply(lambda x: {k: f"{v:.4f}" for k, v in x.items()})
        
        # 인덱스 이름 설정 및 1부터 시작하도록 변경
        df.index = range(1, len(df) + 1)
        df.index.name = 'model_index'
        
        print("\nRegistered Models:")
        print(df.to_string())
    
    def manage_model(self, model_name: str):
        """대화형 모델 관리"""
        while True:
            # 모델 목록 표시
            print("\n=== Model Management ===")
            model_infos = self.load_model_info()
            
            if not model_infos:
                print("No models found in registry.")
                return
            
            # DataFrame 생성 및 포맷팅
            df = pd.DataFrame(model_infos)
            display_columns = [
                "version",
                "stage",
                "metrics",
                "timestamp"
            ]
            df = df[display_columns]
            
            # metrics 컬럼을 보기 좋게 포맷팅
            df['metrics'] = df['metrics'].apply(lambda x: f"F1: {x.get('val_f1', 0):.4f}")
            
            # 인덱스 1부터 시작
            df.index = range(1, len(df) + 1)
            df.index.name = 'index'
            
            print("\nRegistered Models:")
            print(df.to_string())
            
            # 메뉴 표시
            print("\nOptions:")
            print("1. Promote model to Champion")
            print("2. Demote model to Candidate")
            print("3. Archive model")
            print("4. Sync model registry")
            print("q. Quit")
            
            choice = input("\nSelect an option: ").strip().lower()
            
            if choice == 'q':
                break
            
            try:
                if choice in ['1', '2', '3']:
                    idx = int(input("Enter model index: ")) - 1
                    if 0 <= idx < len(model_infos):
                        version = model_infos[idx]['version']
                        
                        if choice == '1':
                            self.promote_to_production(model_name, version)
                            print(f"Model index {idx+1} promoted to Champion")
                        elif choice == '2':
                            self.promote_to_staging(model_name, version)
                            print(f"Model index {idx+1} demoted to Candidate")
                        elif choice == '3':
                            self.archive_model(model_name, version)
                            print(f"Model index {idx+1} archived")
                        
                        self.sync_model_info()
                    else:
                        print(f"Invalid index. Please enter a number between 1 and {len(model_infos)}")
                    
                elif choice == '4':
                    self.sync_model_info()
                    print("Model registry synchronized")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def _list_all_models(self, model_name: str):
        """모든 모델 버전 표시"""
        print("\n=== Model Versions ===")
        
        # MLflow에서 모델 버전 정보 가져오기
        versions = self.client.search_model_versions(f"name='{model_name}'")
        model_infos = self.load_model_info()
        
        # 각 버전의 상태 표시
        for version in versions:
            # 현재 alias 확인
            try:
                if self.client.get_model_version_by_alias(model_name, ModelAlias.PRODUCTION.value).version == version.version:
                    status = "Champion"
                elif self.client.get_model_version_by_alias(model_name, ModelAlias.STAGING.value).version == version.version:
                    status = "Candidate"
                elif self.client.get_model_version_by_alias(model_name, ModelAlias.ARCHIVED.value).version == version.version:
                    status = "Archived"
                else:
                    status = "None"
            except:
                status = "None"
            
            # 모델 정보 찾기
            model_info = next((info for info in model_infos if info['version'] == version.version), None)
            metrics = model_info['metrics'] if model_info else {}
            
            print(f"\nVersion: {version.version}")
            print(f"Status: {status}")
            print(f"Run ID: {version.run_id}")
            if metrics:
                print(f"F1 Score: {metrics.get('val_f1', 'N/A'):.4f}")
            print(f"Created: {version.creation_timestamp}")
    
    def get_production_model_path(self, model_name: str = 'default') -> Optional[str]:
        """프로덕션 모델의 저장 경로 반환"""
        try:
            print("\nDebug: Finding production model path...")
            
            # 프로덕션 모델 정보 가져오기
            production_models = self.get_production_models()
            if not production_models:
                print("Debug: No production models found.")
                return None
                
            print(f"Debug: Found {len(production_models)} production models")
            
            # 가장 최근의 프로덕션 모델 선택
            latest_model = sorted(
                production_models,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )[0]
            
            print(f"Debug: Selected latest model: {latest_model['run_name']}")
            print(f"Debug: Run ID: {latest_model['run_id']}")
            print(f"Debug: Experiment ID: {latest_model['experiment_id']}")
            
            # MLflow에서 모델 경로 가져오기
            experiment_id = latest_model['experiment_id']
            run_id = latest_model['run_id']
            
            # 두 가지 가능한 경로 확인
            mlruns_path = self.config.project_root / 'mlruns' / experiment_id / run_id / 'artifacts' / 'model'
           # mlartifacts_path = self.config.base_path / 'mlartifacts' / experiment_id / run_id / 'artifacts' / 'model'
            
            print(f"Debug: Checking mlruns path: {mlruns_path}")
            print(f"Debug: mlruns path exists: {os.path.exists(mlruns_path)}")
            #
            # 존재하는 경로 반환
            if os.path.exists(mlruns_path):
                return mlruns_path
            # elif os.path.exists(mlartifacts_path):
            #     return mlartifacts_path
            else:
                print("Model path not found in either mlruns or mlartifacts directories")
                return None
                
        except Exception as e:
            import traceback
            print(f"Error getting production model path: {str(e)}")
            traceback.print_exc()
            return None
    
    def load_production_model(self, model_name: str):
        """프로덕션 모델 로드"""
        try:
            # 프로덕션 모델 정보 가져오기
            production_models = self.get_production_models()
            if not production_models:
                print("No production models found.")
                return None
                
            # 가장 최근의 프로덕션 모델 선택
            latest_model = sorted(
                production_models,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )[0]
            
            # 모델 파일 경로 확인
            model_path = self.get_production_model_path(model_name)
            if not model_path:
                print(f"Model path not found for: {model_name}")
                return None
                
            print(f"\nLoading model from: {model_path}")
            
            # config.json 로드
            config_path = Path(model_path) / "config.json"
            if not config_path.exists():
                print(f"Config file not found at: {config_path}")
                return None
                
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            print(f"Loaded config: {config}")
            
            # 기본값 설정
            default_config = {
                'num_unfreeze_layers': -1,  # 기본값 설정
                'learning_rate': 2e-5 if 'ELECTRA' in config['model_type'] else 5e-6,
                'optimizer': 'AdamW',
                'lr_scheduler': 'cosine' if 'ELECTRA' in config['model_type'] else 'exp',
                'precision': 16,
                'batch_size': 32,
                'accumulate_grad_batches': 2
            }
            
            # config에 없는 키는 기본값으로 설정
            for key, value in default_config.items():
                if key not in config:
                    print(f"Warning: {key} not found in config.json, using default value: {value}")
                    config[key] = value
            
            # 모델 타입에 따라 적절한 클래스 초기화
            if config['model_type'] == 'KcBERT':
                from src.models.kcbert_model import KcBERT
                model = KcBERT(
                    pretrained_model=config['pretrained_model'],
                    num_labels=config['num_labels'],
                    num_unfreeze_layers=config['num_unfreeze_layers']
                )
            elif config['model_type'] == 'KcELECTRA':
                from src.models.kcelectra_model import KcELECTRA
                model = KcELECTRA(
                    pretrained_model=config['pretrained_model'],
                    num_labels=config['num_labels'],
                    num_unfreeze_layers=config['num_unfreeze_layers']
                )
            else:
                raise ValueError(f"Unknown model type: {config['model_type']}")
            
            # 모델 가중치 로드
            model_pt_path = Path(model_path) / "model.pt"
            if not model_pt_path.exists():
                print(f"Model weights file not found at: {model_pt_path}")
                print(f"Checking directory contents:")
                print(f"Directory exists: {model_path.exists()}")
                if model_path.exists():
                    print(f"Files in directory:")
                    for file in model_path.iterdir():
                        print(f"  - {file}")
                return None
                
            # CPU 환경에서도 동작하도록 map_location 추가
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            state_dict = torch.load(model_pt_path, map_location=torch.device(device))
            model.load_state_dict(state_dict)
            model.eval()
            
            return model
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_best_model_info(self, metric: str = "val_f1") -> Optional[Dict]:
        """최고 성능의 모델 정보 반환"""
        try:
            model_infos = self.load_model_info()
            if not model_infos:
                return None
            
            sorted_models = sorted(
                model_infos,
                key=lambda x: x['metrics'].get(metric, 0),
                reverse=True
            )
            return sorted_models[0]
            
        except Exception as e:
            print(f"Error getting best model info: {str(e)}")
            return None
    
    def check_production_model_exists(self, model_name: str) -> bool:
        """Production(champion) 단계의 모델이 존재하는지 확인"""
        try:
            version = self.client.get_model_version_by_alias(
                name=model_name,
                alias=ModelAlias.PRODUCTION.value
            )
            return version is not None
        except Exception:
            return False
    
    def get_latest_model_info(self) -> Optional[Dict]:
        """가장 최근의 모델 정보 반환"""
        try:
            model_infos = self.load_model_info()
            return model_infos[-1] if model_infos else None
        except Exception as e:
            print(f"Error getting latest model info: {str(e)}")
            return None
    
    def get_production_models(self) -> List[Dict]:
        """Production 단계의 모든 모델 정보 반환"""
        try:
            model_infos = self.load_model_info()
            production_models = [
                info for info in model_infos 
                if info.get('stage') == ModelAlias.PRODUCTION.value
            ]
            return production_models
        except Exception as e:
            print(f"Error getting production models: {str(e)}")
            return []
    
    def select_production_model(self) -> Optional[Dict]:
        """Production 모델 중 하나를 선택"""
        production_models = self.get_production_models()
        
        if not production_models:
            print("No production models found.")
            return None
        
        if len(production_models) == 1:
            return production_models[0]
        
        print("\n=== Production Models ===")
        df = pd.DataFrame(production_models)
        df.index = range(1, len(df) + 1)
        df.index.name = 'model_index'
        print(df.to_string())
        
        while True:
            try:
                choice = input("\nSelect model index (or 'q' to quit): ")
                if choice.lower() == 'q':
                    return None
                
                idx = int(choice) - 1
                if 0 <= idx < len(production_models):
                    return production_models[idx]
                else:
                    print(f"Invalid index. Please enter a number between 1 and {len(production_models)}")
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def load_production_model_info(self) -> Optional[Dict]:
        """Production 모델 정보 로드 (UI 용)"""
        try:
            production_models = self.get_production_models()
            
            if not production_models:
                return None
            
            if len(production_models) == 1:
                return production_models[0]
            
            # 가장 최근의 production 모델 반환
            return sorted(
                production_models,
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )[0]
        except Exception as e:
            print(f"Error loading production model info: {str(e)}")
            return None
    
    def sync_model_info(self):
        """MLflow alias와 JSON 파일의 stage 정보 동기화"""
        try:
            print("\nDebug: Syncing model info...")
            model_infos = self.load_model_info()
            model_name = self.config.project['model_name']
            
            # MLflow에서 모든 버전 정보 가져오기
            versions = self.client.search_model_versions(f"name='{model_name}'")
            print(f"Debug: Found {len(versions)} model versions")
            
            # 현재 alias 상태 확인
            alias_map = {}
            for alias in [ModelAlias.PRODUCTION.value, ModelAlias.STAGING.value, ModelAlias.ARCHIVED.value]:
                try:
                    version = self.client.get_model_version_by_alias(model_name, alias)
                    if version:
                        alias_map[version.version] = alias
                        print(f"Debug: Version {version.version} has alias: {alias}")
                except Exception as e:
                    print(f"Debug: No version found for alias {alias}: {str(e)}")
            
            # JSON 파일 업데이트
            updated = False
            for info in model_infos:
                current_stage = info.get('stage')
                new_stage = alias_map.get(info['version'], ModelAlias.NONE.value)
                
                if current_stage != new_stage:
                    print(f"Debug: Updating version {info['version']} stage from {current_stage} to {new_stage}")
                    info['stage'] = new_stage
                    updated = True
            
            if updated:
                # 저장
                with open(self.model_info_path, 'w', encoding='utf-8') as f:
                    json.dump(model_infos, f, indent=2, ensure_ascii=False)
                print("Debug: Model info file updated")
                
                # UI 새로고침을 위한 플래그 설정
                if 'st' in sys.modules:
                    import streamlit as st
                    st.session_state.model_state_changed = True
                    print("Debug: Set UI refresh flag")
            
            print("Model info synchronized with MLflow aliases")
            
        except Exception as e:
            print(f"Error synchronizing model info: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _set_model_alias(self, model_name: str, version: str, new_alias: str):
        """모델 alias 설정 (다른 alias 제거)"""
        try:
            print(f"\nDebug: Setting alias {new_alias} for version {version}")
            
            # 1. 해당 버전의 현재 alias 확인 및 제거
            for alias in [a.value for a in ModelAlias]:
                try:
                    current = self.client.get_model_version_by_alias(model_name, alias)
                    if current and current.version == version:
                        print(f"Debug: Removing existing alias {alias} from version {version}")
                        self.client.delete_registered_model_alias(model_name, alias)
                except: pass
            
            # 2. 새로운 alias 설정
            self.client.set_registered_model_alias(
                name=model_name,
                alias=new_alias,
                version=version
            )
            print(f"Debug: Set new alias {new_alias} for version {version}")
            
            # 3. 동기화
            self.sync_model_info()
            
        except Exception as e:
            print(f"Error setting model alias: {str(e)}")
            raise

class ModelInference:
    def __init__(self, config):
        self.config = config
        self.model_manager = MLflowModelManager(config)
        self.model = None
        self.tokenizer = None
    
    def load_production_model(self):
        """프로덕션 모델 로드"""
        model_name = self.config.project['model_name']
        
        # Production 모델 체크
        if not self.model_manager.check_production_model_exists(model_name):
            print(f"No production model found for {model_name}")
            print("Attempting to load latest model and promote to production...")
            
            # 최신 모델 정보 가져오기
            latest_model = self.model_manager.get_latest_model_info()
            if latest_model:
                print(f"Debug: Latest model info: {latest_model}")
                # Staging을 거쳐 Production으로 승격
                model_version = self.model_manager.promote_to_staging(
                    model_name, 
                    latest_model['run_id']
                )
                self.model_manager.promote_to_production(
                    model_name, 
                    model_version.version
                )
                print(f"Model {model_name} promoted to production.")
                
                # MLflow 업데이트를 위한 짧은 대기
                import time
                time.sleep(1)
                
                # 모델 로드 재시도
                return self.load_production_model()
            else:
                print("No models found in registry.")
                return None
        
        # Production 델 로드
        model = self.model_manager.load_production_model(model_name)
        if model is None:
            print("Failed to load production model.")
            return None
        
        self.model = model
        return model
    
    def predict(self, texts: List[str]) -> List[int]:
        """텍스트 감성 분석"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model and tokenizer must be loaded first")
        
        import torch
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.training_config['max_length'],
            return_tensors="pt"
        )
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        return predictions.tolist()

# 사용 예시
if __name__ == "__main__":
    config = Config()
    
    # MLflow 모델 관리 초기화
    model_manager = MLflowModelManager(config)
    model_manager.manage_model(config.project['model_name'])
    
    # # 현재 등록된 모델 표시
    # print("\nRegistered Models:")
    # model_manager.display_models()
    
