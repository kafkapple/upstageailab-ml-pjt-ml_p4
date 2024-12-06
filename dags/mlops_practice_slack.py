from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import joblib
import os
import sys

sys.path.append('/data/ephemeral/home/upstageailab-ml-pjt-ml_p4')

os.environ['NO_PROXY'] = '*' # mac에서 airflow로 외부 요청할 때 이슈가 있음. 하여 해당 코드 추가 필요
# https://stackoverflow.com/questions/76546457/airflow-job-unable-to-send-requests-to-the-internet


default_args = {
    'owner': 'admin',
    'start_date': datetime(2023, 9, 22),
    'retries': 1,
}

# 1. 데이터 준비 함수
def prepare_data(**context):
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XCom을 사용하여 데이터를 전달
    context['ti'].xcom_push(key='X_train', value=X_train.to_json())
    context['ti'].xcom_push(key='X_test', value=X_test.to_json())
    context['ti'].xcom_push(key='y_train', value=y_train.to_json(orient='records'))
    context['ti'].xcom_push(key='y_test', value=y_test.to_json(orient='records'))

# 2. 모델 학습 함수
def train_model(model_name, **context):
    ti = context['ti']
    X_train = pd.read_json(ti.xcom_pull(key='X_train'))
    y_train = pd.read_json(ti.xcom_pull(key='y_train'), typ='series')

    if model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == 'SVM':
        model = SVC()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)

    # 모델을 파일로 저장
    model_path = f'/tmp/{model_name}_model.pkl'
    joblib.dump(model, model_path)

    context['ti'].xcom_push(key=f'model_path_{model_name}', value=model_path)

# 3. 모델 평가 함수
def evaluate_model(model_name, **context):
    ti = context['ti']
    model_path = ti.xcom_pull(key=f'model_path_{model_name}')
    model = joblib.load(model_path)

    X_test = pd.read_json(ti.xcom_pull(key='X_test'))
    y_test = pd.read_json(ti.xcom_pull(key='y_test'), typ='series')

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Model Accuracy: {accuracy}")

    context['ti'].xcom_push(key=f'performance_{model_name}', value=accuracy)

# 4. 최고 성능 모델 선택
def select_best_model(**context):
    ti = context['ti']

    rf_performance = ti.xcom_pull(key='performance_RandomForest')
    gb_performance = ti.xcom_pull(key='performance_GradientBoosting')
    svm_performance = ti.xcom_pull(key='performance_SVM')

    performances = {
        'RandomForest': rf_performance,
        'GradientBoosting': gb_performance,
        'SVM': svm_performance
    }

    best_model = max(performances, key=performances.get)
    best_performance = performances[best_model]

    print(f"Best Model: {best_model} with accuracy {best_performance}")
    context['ti'].xcom_push(key='best_model', value=best_model)
def send_slack_basic(**context):
    """학습 시작 알림을 보내는 함수"""
    message = "🚀 감성 분석 모델 학습을 시작합니다..."
    
    # SlackWebhookOperator를 직접 실행하지 않고 반환
    return SlackWebhookOperator(
        task_id='send_slack_start_notification',
        webhook_conn_id='slack_webhook',
        message=message,
        username='ML Pipeline Bot',
    ).execute(context=context)

# 5. Slack 메시지 전송 함수
def send_slack_notification(**context):
    ti = context['ti']
    best_model = ti.xcom_pull(key='best_model')
    rf_performance = ti.xcom_pull(key='performance_RandomForest')
    gb_performance = ti.xcom_pull(key='performance_GradientBoosting')
    svm_performance = ti.xcom_pull(key='performance_SVM')
    
    message = (f"🎯 모델 학습 결과:\n\n"
               f"*최고 성능 모델: {best_model}*\n\n"
               f"📊 성능 비교:\n"
               f"• RandomForest: {rf_performance:.4f}\n"
               f"• GradientBoosting: {gb_performance:.4f}\n"
               f"• SVM: {svm_performance:.4f}")
    
    return SlackWebhookOperator(
        task_id='send_slack_final_notification',
        webhook_conn_id='slack_webhook',
        message=message,
        username='ML Pipeline Bot',
    ).execute(context=context)

# DAG 정의
dag = DAG(
    'iris_ml_training_pipeline_multiple_models_slack',
    default_args=default_args,
    description='A machine learning pipeline using multiple models on Iris dataset',
    schedule_interval='@daily',
    catchup=False
)

send_slack_basic_task = PythonOperator(
    task_id='send_slack_basic',
    python_callable=send_slack_basic,
    provide_context=True,
    dag=dag,
)

# Task 정의
prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_data,
    provide_context=True,
    dag=dag,
)

train_rf_task = PythonOperator(
    task_id='train_random_forest',
    python_callable=train_model,
    op_kwargs={'model_name': 'RandomForest'},
    provide_context=True,
    dag=dag,
)

train_gb_task = PythonOperator(
    task_id='train_gradient_boosting',
    python_callable=train_model,
    op_kwargs={'model_name': 'GradientBoosting'},
    provide_context=True,
    dag=dag,
)

train_svm_task = PythonOperator(
    task_id='train_svm',
    python_callable=train_model,
    op_kwargs={'model_name': 'SVM'},
    provide_context=True,
    dag=dag,
)

evaluate_rf_task = PythonOperator(
    task_id='evaluate_random_forest',
    python_callable=evaluate_model,
    op_kwargs={'model_name': 'RandomForest'},
    provide_context=True,
    dag=dag,
)

evaluate_gb_task = PythonOperator(
    task_id='evaluate_gradient_boosting',
    python_callable=evaluate_model,
    op_kwargs={'model_name': 'GradientBoosting'},
    provide_context=True,
    dag=dag,
)

evaluate_svm_task = PythonOperator(
    task_id='evaluate_svm',
    python_callable=evaluate_model,
    op_kwargs={'model_name': 'SVM'},
    provide_context=True,
    dag=dag,
)

select_best_model_task = PythonOperator(
    task_id='select_best_model',
    python_callable=select_best_model,
    provide_context=True,
    dag=dag,
)

# Slack 메시지 전송 Task
slack_notification_task = PythonOperator(
    task_id='send_slack_notification',
    python_callable=send_slack_notification,
    provide_context=True,
    dag=dag,
)

# Task 의존성 설정
send_slack_basic_task >> prepare_data_task >> [train_rf_task, train_gb_task, train_svm_task]
train_rf_task >> evaluate_rf_task
train_gb_task >> evaluate_gb_task
train_svm_task >> evaluate_svm_task
[evaluate_rf_task, evaluate_gb_task, evaluate_svm_task] >> select_best_model_task >> slack_notification_task

# export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/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'