#!/bin/bash

# 색상 코드 설정
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Airflow 서비스 시작 중...${NC}"

# Airflow 웹서버 시작
airflow webserver --port 8080 -D
sleep 5

# Airflow 스케줄러 시작
airflow scheduler -D

echo -e "${GREEN}Airflow 서비스 시작 완료.${NC}"

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