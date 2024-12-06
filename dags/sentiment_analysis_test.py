from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
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

from src.inference import SentimentPredictor
from src.config import Config
from src.utils.mlflow_utils import MLflowModelManager

def send_slack_notification(message, **context):
    """Slack 알림 전송"""
    return SlackWebhookOperator(
        task_id='slack_notification',
        webhook_conn_id='slack_webhook',
        message=message,
        username='감성분석 테스트 봇',
    ).execute(context=context)

def notify_start(**context):
    """테스트 시작 알림"""
    message = "🚀 감성 분석 모델 테스트를 시작합니다..."
    return send_slack_notification(message, **context)

def load_model_with_notification(**context):
    """모델 로드 및 알림"""
    try:
        predictor = SentimentPredictor(
            config_path=str(project_root / "config" / "config.yaml"),
            alias="champion"
        )
        message = "✅ 모델 로드 완료\n• 모델: KcBERT\n• 버전: champion"
        return send_slack_notification(message, **context)
    except Exception as e:
        error_message = f"❌ 모델 로드 실패\n오류: {str(e)}"
        send_slack_notification(error_message, **context)
        raise

def format_result(result):
    """추론 결과를 포맷팅"""
    text = result['text']
    label = result['label']
    confidence = result['confidence']
    
    if 'probs' in result:
        probs = result['probs']
        return f"📝 텍스트: {text}\n🏷️ 예측: {label} (확률: {confidence:.2%})\n📊 확률분포: 긍정={probs['긍정']:.2%}, 부정={probs['부정']:.2%}\n"
    else:
        return f"📝 텍스트: {text}\n🏷️ 예측: {label} (확률: {confidence:.2%})\n"

def run_single_inference(**context):
    """단일 추론 테스트"""
    try:
        predictor = SentimentPredictor(
            config_path=str(project_root / "config" / "config.yaml"),
            alias="champion"
        )
        text = "정말 재미있는 영화였어요!"
        result = predictor.predict(text)
        
        message = "🎯 단일 추론 테스트 결과:\n\n" + format_result(result)
        return send_slack_notification(message, **context)
    except Exception as e:
        error_message = f"❌ 단일 추론 테스트 실패\n오류: {str(e)}"
        send_slack_notification(error_message, **context)
        raise

def run_batch_inference(**context):
    """배치 추론 테스트"""
    try:
        predictor = SentimentPredictor(
            config_path=str(project_root / "config" / "config.yaml"),
            alias="champion"
        )
        texts = [
            "다시 보고 싶은 영화예요!",
            "시간 낭비였네요...",
            "배우들의 연기가 훌륭했습니다.",
            "스토리가 너무 뻔해요."
        ]
        results = predictor.predict(texts)
        
        message = "📊 배치 추론 테스트 결과:\n\n" + "\n".join(format_result(r) for r in results)
        return send_slack_notification(message, **context)
    except Exception as e:
        error_message = f"❌ 배치 추론 테스트 실패\n오류: {str(e)}"
        send_slack_notification(error_message, **context)
        raise

def validate_results_with_notification(**context):
    """결과 검증 및 최종 알림"""
    message = "✅ 테스트 완료!\n\n📋 요약:\n• 모델 로드: 성공\n• 단일 추론: 성공\n• 배치 추론: 성공\n• 전체 테스트: 성공"
    return send_slack_notification(message, **context)

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
    'sentiment_analysis_test',
    default_args=default_args,
    description='감성 분석 모델 테스트 DAG',
    schedule_interval=None,  # 수동 실행
    catchup=False
)

# 시작 알림
start_notification = PythonOperator(
    task_id='notify_start',
    python_callable=notify_start,
    dag=dag,
)

# 모델 로드 태스크
load_model = PythonOperator(
    task_id='load_model',
    python_callable=load_model_with_notification,
    dag=dag,
)

# 단일 추론 테스트
single_inference = PythonOperator(
    task_id='single_inference_test',
    python_callable=run_single_inference,
    dag=dag,
)

# 배치 추론 테스트
batch_inference = PythonOperator(
    task_id='batch_inference_test',
    python_callable=run_batch_inference,
    dag=dag,
)

# 결과 검증 태스크
validate_results = PythonOperator(
    task_id='validate_results',
    python_callable=validate_results_with_notification,
    dag=dag,
)

# 태스크 순서 정의
start_notification >> load_model >> single_inference >> batch_inference >> validate_results
