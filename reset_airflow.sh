#!/bin/bash

# 색상 코드 설정
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Airflow 제거 및 재설정 중...${NC}"

# 1. 기존 Airflow 프로세스 종료
echo -e "${YELLOW}기존 Airflow 프로세스 종료 중...${NC}"
pkill -f "airflow webserver" || true
pkill -f "airflow scheduler" || true

# 2. PostgreSQL 데이터베이스 제거
echo -e "${YELLOW}PostgreSQL 데이터베이스 제거 중...${NC}"
su - postgres -c "psql -c \"DROP DATABASE IF EXISTS airflow;\""
su - postgres -c "psql -c \"DROP USER IF EXISTS airflow;\""

# 3. Airflow 디렉토리 제거
echo -e "${YELLOW}Airflow 디렉토리 제거 중...${NC}"
rm -rf /data/ephemeral/home/airflow

# 4. Airflow 재설정
echo -e "${YELLOW}Airflow 재설정 중...${NC}"
./setup_airflow.sh

echo -e "${GREEN}Airflow 제거 및 재설정 완료.${NC}" 