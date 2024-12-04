from airflow.models import Variable

# Airflow UI에서 Variables에 추가해야 할 설정
SLACK_CONFIG = {
    "SLACK_TOKEN": "your-slack-bot-token",
    "SLACK_CHANNEL": "#model-training"
}

# Variables 설정 방법
for key, value in SLACK_CONFIG.items():
    Variable.set(key, value) 