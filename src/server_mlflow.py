import os
from pathlib import Path
import subprocess
from config import Config

def start_mlflow_server(config: Config):
    """MLflow 서버 시작"""
    
    # MLflow 파일 URI 허용 설정
    os.environ['MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE'] = 'true'
    
    # MLflow 저장 경로 설정 (Windows 경로를 Unix 스타일로 변환)
    mlruns_path = str(config.project_root / config.mlflow.mlrun_path).replace("\\", "/")
    artifact_root = str(config.project_root / config.mlflow.artifact_location).replace("\\", "/")

    # 디렉토리 생성
    os.makedirs(mlruns_path, exist_ok=True)
    os.makedirs(artifact_root, exist_ok=True)
    
    # MLflow 서버 실행 명령
    # http://127.0.0.1:5050 -> host=127.0.0.1, port=5050
    uri_parts = config.mlflow.tracking_uri.replace("http://", "").split(":")
    mlflow_host = uri_parts[0]  # 127.0.0.1
    mlflow_port = uri_parts[1]  # 5050
    
    # 경로를 Unix 스타일로 통일
    mlflow_backend_store_uri = f"file:///{mlruns_path}"
    mlflow_default_artifact_root = f"file:///{artifact_root}"
    
    print(f"Host: {mlflow_host}")
    print(f"Port: {mlflow_port}")
    print(f"Backend Store URI: {mlflow_backend_store_uri}")
    print(f"Default Artifact Root: {mlflow_default_artifact_root}")
    
    cmd = [
        "mlflow", "server",
        "--backend-store-uri", mlflow_backend_store_uri,
        "--default-artifact-root", mlflow_default_artifact_root,
        "--host", mlflow_host,
        "--port", mlflow_port
    ]
    
    print(f"\nStarting MLflow server...")
    print(f"Tracking URI: {config.mlflow.tracking_uri}")
    print(f"Experiment data: {mlruns_path}")
    print(f"Default artifacts: {artifact_root}")
    print(f"Run artifacts will be stored at: {artifact_root}/exp_id/run_id/artifacts\n")
    
    # 서버 실행 (포그라운드 모드)
    subprocess.run(cmd)
    # subprocess.Popen(cmd) # background 모드

def update_artifact_location(config: Config, experiment_id: str, run_id: str):
    """아티팩트 저장 위치 설정"""
    import mlflow
    import shutil
    
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    
    # 원하는 아티팩트 경로 설정
    target_artifact_path = config.project_root / config.mlflow.artifact_location / experiment_id / run_id / "artifacts"
    os.makedirs(target_artifact_path, exist_ok=True)
    
    # 현재 실험 가져오기
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None:
        print(f"Error: Experiment {config.mlflow.experiment_name} not found")
        return None, None
    
    print(f"Using experiment: {config.mlflow.experiment_name} (ID: {experiment.experiment_id})")
    
    # 기존 아티팩트 경로 (임시)
    temp_artifact_path = config.project_root / config.mlflow.artifact_location / "temp_artifacts"
    os.makedirs(temp_artifact_path, exist_ok=True)
    
    # 아티팩트를 임시 경로에 저장하고 원하는 경로로 이동
    def log_artifact_with_path(local_path: str, artifact_path: str = None):
        """아티팩트를 지정된 경로에 저장"""
        # 먼저 임시 경로에 저장
        client.log_artifact(run_id, local_path, artifact_path)
        
        # 실제 원하는 경로로 파일 이동
        artifact_name = os.path.basename(local_path)
        temp_path = temp_artifact_path / artifact_name
        final_path = target_artifact_path / artifact_name
        
        if os.path.exists(temp_path):
            shutil.move(str(temp_path), str(final_path))
            print(f"Moved artifact to: {final_path}")
    
    return experiment_id, log_artifact_with_path

if __name__ == "__main__":
    config = Config()
    start_mlflow_server(config) 

# MLflow UI 실행 옵션:
# mlflow ui --host 0.0.0.0 --port 5050 &
# 
# 주요 션:
# --host: 호스트 주소 (0.0.0.0은 모든 IP에서 접근 허용)
# --port: UI 서버 포트 지정
# --backend-store-uri: MLflow 실험 데이터 저장 위치
# --default-artifact-root: 모델 아티팩트 저장 위치 (기본값)
# --workers: Gunicorn worker 프로세스 수
# --gunicorn-opts: Gunicorn 서버 추가 옵션 