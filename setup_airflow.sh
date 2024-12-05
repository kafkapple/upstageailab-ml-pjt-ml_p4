#!/bin/bash

# 색상 코드 설정
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 에러 발생 시 스크립트 중단
set -e

echo -e "${GREEN}=== Airflow 설정 시작 ===${NC}"

# 1. Airflow 홈 디렉토리 설정
AIRFLOW_HOME="/data/ephemeral/home/airflow"
export AIRFLOW_HOME
echo "export AIRFLOW_HOME=${AIRFLOW_HOME}" >> ~/.bashrc

# 2. PostgreSQL 설치 및 설정
echo -e "${YELLOW}PostgreSQL 설치 및 설정 중...${NC}"
apt-get install -y postgresql postgresql-contrib libpq-dev
service postgresql start

# PostgreSQL 데이터베이스 재설정
echo -e "${YELLOW}PostgreSQL 데이터베이스 재설정 중...${NC}"
su - postgres -c "psql -c \"DROP DATABASE IF EXISTS airflow;\""
su - postgres -c "psql -c \"DROP USER IF EXISTS airflow;\""
su - postgres -c "psql -c \"CREATE USER airflow WITH PASSWORD 'airflow';\""
su - postgres -c "psql -c \"CREATE DATABASE airflow OWNER airflow;\""
su - postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;\""

# 3. Python 패키지 설치
echo -e "${YELLOW}Python 패키지 설치 중...${NC}"
pip install psycopg2-binary

# 4. Airflow 설정 관리
echo -e "${YELLOW}Airflow 설정 관리 중...${NC}"
# dags 폴더 백업 (이미 존재하는 경우)
if [ -d "${AIRFLOW_HOME}/dags" ]; then
    echo -e "${YELLOW}기존 dags 폴더 백업 중...${NC}"
    mv ${AIRFLOW_HOME}/dags ${AIRFLOW_HOME}/dags_backup_$(date +%Y%m%d_%H%M%S)
fi

# 기존 Airflow 설정 제거 (dags 폴더 제외)
rm -rf ${AIRFLOW_HOME}/airflow.cfg ${AIRFLOW_HOME}/logs ${AIRFLOW_HOME}/plugins
mkdir -p ${AIRFLOW_HOME}

# 5. 필요한 디렉토리 생성
echo -e "${YELLOW}디렉토리 생성 중...${NC}"
mkdir -p ${AIRFLOW_HOME}/dags
mkdir -p ${AIRFLOW_HOME}/logs
mkdir -p ${AIRFLOW_HOME}/plugins

# 6. 환경 변수 설정
echo -e "${YELLOW}환경 변수 설정 중...${NC}"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="postgresql+psycopg2://airflow:airflow@localhost/airflow"
echo "export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@localhost/airflow" >> ~/.bashrc

# 7. Airflow 설정 파일 생성
echo -e "${YELLOW}Airflow 설정 파일 생성 중...${NC}"
cat > ${AIRFLOW_HOME}/airflow.cfg << EOL
[core]
dags_folder = ${AIRFLOW_HOME}/dags
load_examples = False
executor = LocalExecutor

[database]
sql_alchemy_conn = postgresql+psycopg2://airflow:airflow@localhost/airflow
sql_engine_encoding = utf-8

[webserver]
web_server_port = 8080
authenticate = True
auth_backend = airflow.auth.backend.session

[scheduler]
min_file_process_interval = 30
dag_file_processor_timeout = 600
EOL

# 8. 기존 Airflow 프로세스 종료
echo -e "${YELLOW}기존 Airflow 프로세스 종료 중...${NC}"
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true

# 9. 데이터베이스 초기화
echo -e "${YELLOW}Airflow 데이터베이스 초기화 중...${NC}"
airflow db init

# 10. Airflow 관리자 계정 생성
echo -e "${YELLOW}Airflow 관리자 계정 생성 중...${NC}"
airflow users create \
    --username admin \
    --firstname admin \
    --lastname admin \
    --role Admin \
    --email admin@example.com \
    --password admin

# 11. 권한 설정
echo -e "${YELLOW}권한 설정 중...${NC}"
chmod -R 777 ${AIRFLOW_HOME}

# 12. Slack Webhook 설정
echo -e "${YELLOW}Slack Webhook 설정 중...${NC}"
if [ -f "$AIRFLOW_HOME/connections/setup_slack.sh" ]; then
    echo "Setting up Slack Webhook connection..."
    source $AIRFLOW_HOME/connections/setup_slack.sh
    echo "Slack Webhook setup completed."
else
    echo "Slack Webhook 설정 파일을 찾을 수 없습니다."
fi

# 13. Airflow 서비스 시작
echo -e "${YELLOW}Airflow 서비스 시작 중...${NC}"
airflow webserver --port 8080 -D
sleep 5
airflow scheduler -D

echo -e "${GREEN}=== Airflow 설정 완료 ===${NC}"
echo -e "${GREEN}웹 인터페이스: http://localhost:8080${NC}"
echo -e "${GREEN}사용자명: admin${NC}"
echo -e "${GREEN}비밀번호: admin${NC}"

# 서비스 상태 확인
echo -e "${YELLOW}서비스 상태 확인 중...${NC}"
sleep 5
if pgrep -f "airflow webserver" > /dev/null
then
    echo -e "${GREEN}Webserver가 실행 중입니다.${NC}"
else
    echo -e "${RED}Webserver 실행 실패!${NC}"
fi

if pgrep -f "airflow scheduler" > /dev/null
then
    echo -e "${GREEN}Scheduler가 실행 중입니다.${NC}"
else
    echo -e "${RED}Scheduler 실행 실패!${NC}"
fi 