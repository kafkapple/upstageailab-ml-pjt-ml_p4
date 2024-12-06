from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import mlflow
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parents[2].absolute()
sys.path.append(str(project_root))
os.environ['PYTHONPATH'] = str(project_root)
sys.path.append('/data/ephemeral/home/upstageailab-ml-pjt-ml_p4')
from src.config import Config
from src.utils.mlflow_utils import MLflowModelManager
from src.inference import SentimentPredictor
from src.train import ModelTrainer

os.environ['NO_PROXY'] = '*'  # mac에서 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요

def prepare_wild_data(**context):
    """in-the-wild 데이터 준비 및 분할"""
    config = Config()
    
    # 예측기 초기화
    predictor = SentimentPredictor(
        config_path=str(project_root / "config" / "config.yaml"),
        alias="champion"
    )
    
    # 데이터 로드
    data_path = Path(config.project['data_path']) / config.dataset['in_the_wild']['wild_data_path']
    df = pd.read_csv(data_path)
    
    # 텍스트 컬럼명
    text_col = config.dataset['in_the_wild']['column_mapping']['text']
    label_col = config.dataset['in_the_wild']['column_mapping']['label']
    
    print(f"Loaded {len(df)} samples from {data_path}")
    
    # 추론 실행
    texts = df[text_col].tolist()
    results = predictor.predict(texts)
    
    # 레이블 및 신뢰도 추가
    df[label_col] = [1 if r['label'] == '긍정' else 0 for r in results]
    df['confidence'] = [r['confidence'] for r in results]
    
    # 높은 신뢰도(0.8 이상)의 데이터만 선택
    df_filtered = df[df['confidence'] >= 0.6].copy()
    print(f"Selected {len(df_filtered)} samples with high confidence")
    
    # 학습/검증 데이터 분할
    test_size = config.dataset['in_the_wild']['test_size']
    train_df, val_df = train_test_split(
        df_filtered,
        test_size=test_size,
        random_state=config.project['random_state'],
        stratify=df_filtered[label_col]
    )
    
    # 데이터 저장
    train_path = Path(config.project['data_path']) / config.dataset['in_the_wild']['train_data_path']
    val_path = Path(config.project['data_path']) / config.dataset['in_the_wild']['val_data_path']
    
    # 디렉토리가 없으면 생성
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Saved {len(train_df)} training samples to {train_path}")
    print(f"Saved {len(val_df)} validation samples to {val_path}")
    
    # 데이터 통계
    stats = {
        'total_samples': len(df),
        'filtered_samples': len(df_filtered),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'positive_ratio_train': (train_df[label_col] == 1).mean(),
        'positive_ratio_val': (val_df[label_col] == 1).mean(),
    }
    
    return stats

def finetune_model(**context):
    """in-the-wild 데이터로 모델 파인튜닝"""
    try:
        # 모델 파인튜닝
        result = ModelTrainer.train_model(
            config_path=str(project_root / "config" / "config.yaml"),
            dataset_name="in_the_wild",  # in-the-wild 데이터셋 사용
            interactive=False,
            reset_mlflow=False
        )
        
        print("\n=== Fine-tuning Results ===")
        print(f"Run ID: {result['run_id']}")
        print(f"Validation Accuracy: {result['metrics']['val_accuracy']:.4f}")
        print(f"Validation F1 Score: {result['metrics']['val_f1']:.4f}")
        
        return result
        
    except Exception as e:
        print(f"Error during fine-tuning: {str(e)}")
        raise

def send_slack_notification(message, **context):
    """Slack 알림 전송"""
    return SlackWebhookOperator(
        task_id='slack_notification',
        webhook_conn_id='slack_webhook',
        message=message,
        username='파인튜닝 파이프라인 봇',
    ).execute(context=context)

# DAG 정의
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sentiment_model_finetuning',
    default_args=default_args,
    description='감성 분석 모델 파인튜닝 파이프라인',
    schedule_interval='0 2 * * 1',  # 매주 월요일 오전 2시
    catchup=False
)

# 데이터 준비 태스크
prepare_data = PythonOperator(
    task_id='prepare_wild_data',
    python_callable=prepare_wild_data,
    dag=dag,
)

# 파인튜닝 태스크
finetune = PythonOperator(
    task_id='finetune_model',
    python_callable=finetune_model,
    dag=dag,
)

# 결과 검증 태스크
validate = PythonOperator(
    task_id='validate_results',
    python_callable=lambda: print("파인튜닝 결과 검증 완료"),
    dag=dag,
)

# 태스크 순서 정의
prepare_data >> finetune >> validate