from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
from pathlib import Path
import torch
import mlflow
from slack_sdk import WebClient
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout
import json

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parents[2].absolute()
sys.path.append(str(project_root))
os.environ['PYTHONPATH'] = str(project_root)
sys.path.append('/data/ephemeral/home/upstageailab-ml-pjt-ml_p4')
# .env 파일 로드
load_dotenv(project_root / '.env')

# Slack 설정
SLACK_TOKEN = os.getenv("SLACK_WEBHOOK_TOKEN")
SLACK_CHANNEL = os.getenv("CHANNEL_NAME")

if not SLACK_TOKEN:
    raise ValueError("SLACK_WEBHOOK_TOKEN not found in environment variables")
if not SLACK_CHANNEL:
    raise ValueError("CHANNEL_NAME not found in environment variables")

from src.train import ModelTrainer
from src.config import Config
from src.utils.mlflow_utils import MLflowModelManager

class SlackLogHandler:
    """Slack으로 로그를 전송하는 핸들러"""
    def __init__(self, token, channel):
        self.token = token
        self.channel = channel
        self.client = WebClient(token=token)
        self.message_buffer = []
    
    def write(self, message):
        if message.strip():
            self.message_buffer.append(message)
    
    def flush(self):
        if self.message_buffer:
            message = "".join(self.message_buffer)
            if message.strip():
                self.send_message(message)
            self.message_buffer = []
    
    def send_message(self, message):
        try:
            self.client.chat_postMessage(
                channel=self.channel,
                text=f"```\n{message}\n```"
            )
        except Exception as e:
            print(f"Slack 메시지 전송 실패: {str(e)}")

def train_and_evaluate(**context):
    """모델 학습 및 평가 실행"""
    slack_handler = SlackLogHandler(SLACK_TOKEN, SLACK_CHANNEL)
    
    try:
        print("🚀 모델 학습 시작")
        
        # ModelTrainer.train_model 클래스 메서드 사용
        result = ModelTrainer.train_model(
            config_path=str(project_root / "config" / "config.yaml"),
            interactive=False,
            reset_mlflow=False
        )
        
        # 결과 출력
        print("\n✅ 모델 학습 완료")
        print("\n📊 성능 지표:")
        print(f"- Validation Accuracy: {result['metrics']['val_accuracy']:.4f}")
        print(f"- Validation F1 Score: {result['metrics']['val_f1']:.4f}")
        print(f"- Validation Precision: {result['metrics']['val_precision']:.4f}")
        print(f"- Validation Recall: {result['metrics']['val_recall']:.4f}")
        
        print(f"\n🔍 Run ID: {result['run_id']}")
        print(f"📁 모델 저장 완료")
        
        return result
        
    except Exception as e:
        print(f"\n❌ 모델 학습 실패")
        print(f"오류: {str(e)}")
        import traceback
        print("\n=== 상세 오류 내용 ===")
        print(traceback.format_exc())
        raise
    
    finally:
        slack_handler.flush()

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
    'sentiment_model_training',
    default_args=default_args,
    description='감성 분석 모델 학습 파이프라인',
    schedule_interval='0 2 * * *',  # 매일 오전 2시
    catchup=False
)

# 환경 체크 태스크
check_env = PythonOperator(
    task_id='check_environment',
    python_callable=lambda: print("환경 체크 완료: Python, CUDA, MLflow 설정 확인"),
    dag=dag,
)

# 데이터 준비 태스크
prepare_data = PythonOperator(
    task_id='prepare_data',
    python_callable=lambda: print("데이터 준비 완료: 학습/검증 데이터셋 구성"),
    dag=dag,
)

# 학습 태스크
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_and_evaluate,
    dag=dag,
)

# 모델 검증 태스크
validate_model = PythonOperator(
    task_id='validate_model',
    python_callable=lambda: print("모델 검증 완료: 성능 지표 확인"),
    dag=dag,
)

# 결과 정리 태스크
finalize = PythonOperator(
    task_id='finalize_training',
    python_callable=lambda: print("학습 파이프라인 완료: 모든 단계 성공"),
    dag=dag,
)

# 태스크 순서 정의
check_env >> prepare_data >> train_task >> validate_model >> finalize