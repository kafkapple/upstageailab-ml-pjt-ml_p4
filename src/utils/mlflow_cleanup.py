import os
import shutil
from pathlib import Path
import yaml
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLflowCleanup:
    """MLflow 실험 디렉토리 정리를 위한 유틸리티 클래스"""
    
    def __init__(self, mlruns_dir: str):
        """
        Args:
            mlruns_dir (str): MLflow 실험 디렉토리 경로
        """
        self.mlruns_dir = Path(mlruns_dir)
        if not self.mlruns_dir.exists():
            raise ValueError(f"MLflow 실험 디렉토리가 존재하지 않습니다: {mlruns_dir}")
    
    def find_malformed_experiments(self) -> List[Path]:
        """meta.yaml 파일이 없는 손상된 실험 디렉토리 찾기
        
        Returns:
            List[Path]: 손상된 실험 디렉토리 경로 목록
        """
        malformed = []
        for exp_dir in self.mlruns_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name != '.trash':
                meta_yaml = exp_dir / 'meta.yaml'
                if not meta_yaml.exists():
                    malformed.append(exp_dir)
        return malformed
    
    def backup_experiments(self, experiments: List[Path], backup_dir: Optional[str] = None) -> Path:
        """실험 디렉토리 백업
        
        Args:
            experiments (List[Path]): 백업할 실험 디렉토리 목록
            backup_dir (Optional[str]): 백업 디렉토리 경로. 기본값은 mlruns 옆의 mlruns_backup
            
        Returns:
            Path: 백업 디렉토리 경로
        """
        if backup_dir is None:
            backup_dir = self.mlruns_dir.parent / 'mlruns_backup'
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        for exp_dir in experiments:
            backup_exp_dir = backup_path / exp_dir.name
            if exp_dir.exists():
                shutil.copytree(exp_dir, backup_exp_dir, dirs_exist_ok=True)
                logger.info(f"실험 백업 완료: {exp_dir.name} -> {backup_exp_dir}")
        
        return backup_path
    
    def remove_malformed_experiments(self, experiments: List[Path]):
        """손상된 실험 디렉토리 제거
        
        Args:
            experiments (List[Path]): 제거할 실험 디렉토리 목록
        """
        for exp_dir in experiments:
            if exp_dir.exists():
                shutil.rmtree(exp_dir)
                logger.info(f"손상된 실험 제거 완료: {exp_dir.name}")
    
    def create_default_meta_yaml(self, experiment_id: str) -> dict:
        """기본 meta.yaml 파일 내용 생성
        
        Args:
            experiment_id (str): 실험 ID
            
        Returns:
            dict: meta.yaml 파일 내용
        """
        return {
            'artifact_location': f'file:///{self.mlruns_dir.absolute()}/{experiment_id}',
            'experiment_id': experiment_id,
            'lifecycle_stage': 'active',
            'name': f'recovered_experiment_{experiment_id}',
            'tags': {}
        }
    
    def cleanup(self, backup: bool = True, backup_dir: Optional[str] = None):
        """MLflow 실험 디렉토리 정리 실행
        
        Args:
            backup (bool): 백업 수행 여부
            backup_dir (Optional[str]): 백업 디렉토리 경로
        """
        malformed = self.find_malformed_experiments()
        if not malformed:
            logger.info("손상된 실험이 없습니다.")
            return
        
        logger.info(f"발견된 손상 실험 수: {len(malformed)}")
        for exp_dir in malformed:
            logger.info(f"손상된 실험 ID: {exp_dir.name}")
        
        if backup:
            backup_path = self.backup_experiments(malformed, backup_dir)
            logger.info(f"백업 완료: {backup_path}")
        
        self.remove_malformed_experiments(malformed)
        logger.info("정리 완료")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MLflow 실험 디렉토리 정리 도구')
    parser.add_argument('--mlruns-dir', type=str, default='mlruns',
                      help='MLflow 실험 디렉토리 경로 (기본값: mlruns)')
    parser.add_argument('--no-backup', action='store_true',
                      help='백업 비활성화')
    parser.add_argument('--backup-dir', type=str,
                      help='백업 디렉토리 경로 (기본값: mlruns_backup)')
    
    args = parser.parse_args()
    
    try:
        cleanup = MLflowCleanup(args.mlruns_dir)
        cleanup.cleanup(backup=not args.no_backup, backup_dir=args.backup_dir)
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise 