import json
import os
from pathlib import Path
from typing import Dict, List, Optional

class ConfigSynchronizer:
    """Synchronize model configurations between registry and artifact configs"""
    
    def __init__(self, registry_path: str):
        """
        # Parameters
        registry_path: Path to model registry JSON file
        """
        self.registry_path = Path(registry_path)
        self._ensure_registry_exists()
    
    def _ensure_registry_exists(self):
        """레지스트리 파일이 존재하지 않으면 생성"""
        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def _load_registry(self) -> List[Dict]:
        """Load model registry data"""
        with open(self.registry_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_artifact_config(self, exp_id: str, run_id: str) -> Optional[Dict]:
        """Load individual model artifact config"""
        config_path = Path(f"mlruns/{exp_id}/{run_id}/artifacts/model/config.json")
        if not config_path.exists():
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_artifact_config(self, exp_id: str, run_id: str, config: Dict) -> None:
        """Save updated config to artifact path"""
        config_path = Path(f"mlruns/{exp_id}/{run_id}/artifacts/model/config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def sync_configs(self) -> None:
        """Synchronize all model configs with registry information"""
        print("\n=== Synchronizing configs ===")
        registry_data = self._load_registry()
        
        for model_info in registry_data:
            print(f"Synchronizing config for run {model_info['run_id']}")
            exp_id = model_info['experiment_id']
            run_id = model_info['run_id']
            
            # Load artifact config
            artifact_config = self._load_artifact_config(exp_id, run_id)
            if artifact_config is None:
                print(f"Warning: Config not found for run {run_id}")
                continue
                
            # Parameters to sync from registry to artifact config
            params_to_sync = [
                'learning_rate',
                'optimizer',
                'lr_scheduler',
                'precision',
                'per_device_train_batch_size',  # batch_size
                'accumulate_grad_batches'
            ]
            
            # Update artifact config with registry params
            updated = False
            for param in params_to_sync:
                print(f"Syncing param: {param}")
                if param not in artifact_config and param in model_info['params']:
                    artifact_config[param] = model_info['params'][param]
                    updated = True
                    
            if updated:
                print(f"Updating config for run {run_id}")
                self._save_artifact_config(exp_id, run_id, artifact_config)
            else:
                print(f"No updates needed for run {run_id}") 
    
    def sync_config(self, run_id: str, metrics: Dict[str, float], run_name: str):
        """설정 동기화
        
        Args:
            run_id: MLflow run ID
            metrics: 학습 메트릭
            run_name: 실행 이름
        """
        try:
            # 기존 레지스트리 로드
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            
            # 새로운 실행 정보 추가
            run_info = {
                'run_id': run_id,
                'metrics': metrics,
                'run_name': run_name,
                'stage': 'Production'  # 기본값
            }
            
            # 기존 Production 모델을 Archived로 변경
            for model in registry:
                if model['stage'] == 'Production':
                    model['stage'] = 'Archived'
            
            # 새로운 모델 정보 추가
            registry.append(run_info)
            
            # 레지스트리 저장
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(registry, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error syncing config: {e}")
            raise