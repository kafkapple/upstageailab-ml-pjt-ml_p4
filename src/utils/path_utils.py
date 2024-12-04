import os
from pathlib import Path
from typing import Union, Optional
import re

def convert_to_mlflow_uri(path: Union[str, Path], make_absolute: bool = True, filename: Optional[str] = None) -> str:
    """Windows 경로를 MLflow URI 형식으로 변환
    
    Args:
        path (Union[str, Path]): 변환할 경로
        make_absolute (bool, optional): 절대 경로로 변환할지 여부. Defaults to True.
        filename (Optional[str], optional): 추가할 파일 이름. Defaults to None.
    
    Returns:
        str: MLflow URI 형식의 경로
    
    Examples:
        >>> convert_to_mlflow_uri("E:\\path\\to\\dir")
        'file:///E:/path/to/dir'
        >>> convert_to_mlflow_uri("E:\\path\\to\\dir", filename="model.pt")
        'file:///E:/path/to/dir/model.pt'
        >>> convert_to_mlflow_uri("relative/path", make_absolute=False)
        'file:///relative/path'
    """
    # Path 객체로 변환
    path = Path(path)
    
    # 파일 이름이 있으면 경로에 추가
    if filename:
        path = path / filename
    
    # 절대 경로로 변환 (옵션)
    if make_absolute:
        path = path.absolute()
    
    # 문자열로 변환하고 경로 구분자 정규화
    path_str = str(path).replace('\\', '/')
    
    # Windows 경로 처리
    if os.name == 'nt':
        # 드라이브 문자가 있는 경우 (예: C:, D: 등)
        if re.match(r'^[a-zA-Z]:', path_str):
            # 드라이브 문자는 대문자로 유지
            drive = path_str[0].upper()
            rest = path_str[2:]
            # 시작 슬래시 제거
            if rest.startswith('/'):
                rest = rest[1:]
            return f"file:///{drive}:/{rest}"
    
    # 일반 경로 처리
    if path_str.startswith('/'):
        return f"file://{path_str}"
    else:
        return f"file:///{path_str}"

def join_mlflow_path(base_uri: str, *parts: str) -> str:
    """MLflow URI에 추가 경로 결합
    
    Args:
        base_uri (str): 기본 MLflow URI (예: file:///E:/path/to/dir)
        *parts (str): 추가할 경로 부분들
    
    Returns:
        str: 결합된 MLflow URI
        
    Examples:
        >>> join_mlflow_path("file:///E:/path/to/dir", "model", "weights.pt")
        'file:///E:/path/to/dir/model/weights.pt'
    """
    # URI 스키마와 경로 분리
    if '://' in base_uri:
        scheme, path = base_uri.split('://', 1)
    else:
        scheme, path = 'file', base_uri
    
    # 경로에서 선행 슬래시 제거
    path = path.lstrip('/')
    
    # Windows 드라이브 문자 처리
    if os.name == 'nt' and re.match(r'^[a-zA-Z]:', path):
        # 드라이브 문자는 그대로 유지
        drive = path[:2]
        path = path[2:].lstrip('/')
        path = f"{drive}/{path}"
    
    # 경로 결합
    full_path = Path(path)
    for part in parts:
        full_path = full_path / part
    
    # 경로 정규화
    normalized_path = str(full_path).replace('\\', '/')
    
    # Windows 드라이브 문자가 있는 경우 처리
    if os.name == 'nt' and re.match(r'^[a-zA-Z]:', normalized_path):
        return f"{scheme}:///{normalized_path}"
    
    return f"{scheme}:///{normalized_path}"

def get_mlflow_paths(config) -> dict:
    """MLflow 관련 모든 경로를 URI 형식으로 변환
    
    Args:
        config: 설정 객체
    
    Returns:
        dict: MLflow 경로 정보
        {
            'tracking_uri': str,
            'artifact_root': str,
            'backend_store_uri': str,
            'model_registry_path': str,
            'model_path': str,
        }
    """
    # 기본 경로 설정
    mlruns_dir = ensure_mlflow_path(config.mlflow.mlrun_path)
    artifact_dir = ensure_mlflow_path(config.mlflow.artifact_root)
    model_registry = ensure_mlflow_path(config.mlflow.model_registry_path, is_dir=False)
    
    # 모델 관련 경로
    model_dir = ensure_mlflow_path(Path(config.models[config.project['model_name']]['model_dir']))
    
    return {
        'tracking_uri': config.mlflow.tracking_uri,
        'artifact_root': convert_to_mlflow_uri(artifact_dir),
        'backend_store_uri': convert_to_mlflow_uri(mlruns_dir),
        'model_registry_path': str(model_registry),
        'model_path': convert_to_mlflow_uri(model_dir),
    }

def ensure_mlflow_path(path: Union[str, Path], is_dir: bool = True) -> Path:
    """MLflow 경로 생성 및 확인
    
    Args:
        path (Union[str, Path]): 확인할 경로
        is_dir (bool, optional): 디렉토리 여부. Defaults to True.
    
    Returns:
        Path: 생성된 경로 객체
    """
    path = Path(path)
    if is_dir:
        path.mkdir(parents=True, exist_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path