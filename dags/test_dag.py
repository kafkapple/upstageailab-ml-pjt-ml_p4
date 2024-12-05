from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from slack_sdk import WebClient
from transformers import pipeline
import pandas as pd
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
# Slack 설정
SLACK_TOKEN = os.getenv('SLACK_TOKEN')
CHANNEL_NAME = os.getenv('CHANNEL_NAME')
SLACK_USERNAME = os.getenv('SLACK_USERNAME')
slack_client = WebClient(token=SLACK_TOKEN)

# 감성 분석 모델 로드
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_channel_id(channel_name):
    """채널 이름으로 채널 ID를 찾습니다"""
    try:
        result = slack_client.conversations_list()
        for channel in result['channels']:
            if channel['name'] == channel_name:
                return channel['id']
        raise ValueError(f"Channel {channel_name} not found")
    except Exception as e:
        print(f"Error finding channel: {e}")
        raise

def get_slack_messages():
    """Slack 채널의 최근 메시지를 가져옵니다"""
    try:
        channel_id = get_channel_id(CHANNEL_NAME)
        result = slack_client.conversations_history(
            channel=channel_id,
            limit=100  # 최근 100개 메시지
        )
        messages = [
            {
                'text': msg['text'],
                'ts': msg['ts']
            } 
            for msg in result['messages']
            if 'text' in msg and msg['text'].strip()
        ]
        return messages
    except Exception as e:
        print(f"Error fetching messages: {e}")
        return []

def analyze_sentiment(messages):
    """메시지의 감성을 분석합니다"""
    results = []
    for msg in messages:
        try:
            sentiment = sentiment_analyzer(msg['text'])[0]
            results.append({
                'text': msg['text'],
                'ts': msg['ts'],
                'sentiment': sentiment['label'],
                'score': sentiment['score']
            })
        except Exception as e:
            print(f"Error analyzing message: {e}")
    return results

def post_results(results):
    """분석 결과를 Slack 채널에 게시합니다"""
    try:
        channel_id = get_channel_id(CHANNEL_NAME)
        summary = pd.DataFrame(results)
        positive_count = len(summary[summary['sentiment'] == 'POSITIVE'])
        negative_count = len(summary[summary['sentiment'] == 'NEGATIVE'])
        
        message = f"""
        *최근 대화 감성 분석 결과*
        - 긍정적인 메시지: {positive_count}
        - 부정적인 메시지: {negative_count}
        - 전체 분석된 메시지: {len(results)}
        """
        
        slack_client.chat_postMessage(
            channel=channel_id,
            text=message
        )
    except Exception as e:
        print(f"Error posting results: {e}")

# DAG 정의
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'slack_sentiment_analysis',
    default_args=default_args,
    description='Slack 채널 메시지 감성 분석',
    schedule_interval='*/30 * * * *',  # 30분마다 실행
    catchup=False
) as dag:

    # Task 정의
    get_messages_task = PythonOperator(
        task_id='get_messages',
        python_callable=get_slack_messages,
    )

    analyze_sentiment_task = PythonOperator(
        task_id='analyze_sentiment',
        python_callable=analyze_sentiment,
        op_kwargs={'messages': "{{ task_instance.xcom_pull(task_ids='get_messages') }}"},
    )

    post_results_task = PythonOperator(
        task_id='post_results',
        python_callable=post_results,
        op_kwargs={'results': "{{ task_instance.xcom_pull(task_ids='analyze_sentiment') }}"},
    )

    # Task 순서 정의
    get_messages_task >> analyze_sentiment_task >> post_results_task