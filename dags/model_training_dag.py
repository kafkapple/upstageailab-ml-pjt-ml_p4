from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import numpy as np
from slack_sdk import WebClient
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Slack 설정
SLACK_TOKEN = os.getenv("SLACK_TOKEN")
SLACK_CHANNEL = "#model-training"  # 실제 채널명으로 수정 필요

def split_dataset():
    """데이터셋을 n개로 분할하고 각각 저장"""
    # 데이터 로드 (예: NSMC 데이터셋)
    data = pd.read_csv("/path/to/nsmc_dataset.csv")  # 실제 데이터 경로로 수정 필요
    
    # 데이터를 n개로 분할
    n_splits = 5  # 원하는 분할 수로 수정
    split_size = len(data) // n_splits
    
    for i in range(n_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size if i < n_splits - 1 else len(data)
        
        split_data = data.iloc[start_idx:end_idx]
        split_data.to_csv(f"/path/to/splits/split_{i}.csv", index=False)
    
    return n_splits

def send_slack_message(message, channel=SLACK_CHANNEL):
    """Slack 메시지 전송"""
    client = WebClient(token=SLACK_TOKEN)
    try:
        response = client.chat_postMessage(
            channel=channel,
            text=message
        )
        return response
    except Exception as e:
        print(f"Error sending slack message: {str(e)}")
        return None

def train_and_notify(split_index, **context):
    """모델 학습 및 결과 알림"""
    # 학습 시작 알림
    start_message = f"🚀 모델 학습 시작 (Split {split_index})"
    send_slack_message(start_message)
    
    try:
        # 모델 학습
        config_path = "config/config.yaml"
        dataset_path = f"/path/to/splits/split_{split_index}.csv"
        
        result = train_model(
            config_path=config_path,
            dataset_name=dataset_path,
            interactive=False
        )
        
        # 결과 메시지 생성
        metrics = result['metrics']
        message = f"""
        ✅ 모델 학습 완료 (Split {split_index})
        
        📊 성능 지표:
        - Validation Accuracy: {metrics['val_accuracy']:.4f}
        - Validation F1 Score: {metrics['val_f1']:.4f}
        - Validation Precision: {metrics['val_precision']:.4f}
        - Validation Recall: {metrics['val_recall']:.4f}
        
        🔍 Run ID: {result['run_id']}
        """
        
    except Exception as e:
        message = f"""
        ❌ 모델 학습 실패 (Split {split_index})
        오류: {str(e)}
        """
    
    # 결과 알림
    send_slack_message(message)

# DAG 정의
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['your-email@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='주기적 모델 학습 파이프라인',
    schedule_interval='0 2 * * *',  # 매일 오전 2시에 실행
    catchup=False
)

# 데이터셋 분할 태스크
split_task = PythonOperator(
    task_id='split_dataset',
    python_callable=split_dataset,
    dag=dag,
)

# 각 분할에 대한 학습 태스크 생성
training_tasks = []
for i in range(5):  # n_splits와 동일한 수로 설정
    train_task = PythonOperator(
        task_id=f'train_model_split_{i}',
        python_callable=train_and_notify,
        op_kwargs={'split_index': i},
        dag=dag,
    )
    training_tasks.append(train_task)

# 태스크 의존성 설정
split_task >> training_tasks 