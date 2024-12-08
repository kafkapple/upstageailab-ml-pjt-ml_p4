# MLOps 프로젝트: 비정제 한국어 데이터 수집 및 감성 분석

## 1. 프로젝트 개요

### 1.1 프로젝트 정보

- **프로젝트 기간**: 2024년 11월 25일 ~ 12월 6일
- **목표**:
    - 웹 데이터 수집부터 모델 서빙까지의 **End-to-End MLOps 파이프라인 구현**
    - 실시간 긍정/부정을 예측하는 한국어 감성 분석 시스템 개발

### 1.2 Team ML 4 멤버 및 협업

- **박정준**:
    - **AI Engineer**
        - **:** NLP 모델 개발, 모델 관리 파이프라인 (MLflow)
- **김동완**:
    - **Data Engineer**
        - : Data management & scraping 데이터 수집 및 관리
- **김묘정**
    - **Front-end Engineer, QA**
        - : 프론트엔드 개발 (Streamlit), 모델 테스트
- **이다언**:
    - **Back-end Engineer**
        - : Fast API

### 1.3 MLOps 아키텍쳐 플로우차트
![MLOps 아키텍쳐 전체 플로우차트](https://github.com/UpstageAILab5/upstageailab-ml-pjt-ml_p4/blob/main/assets/images/flowchart_all.png)
![MLOps 아키텍쳐 플로우차트](https://github.com/UpstageAILab5/upstageailab-ml-pjt-ml_p4/blob/main/assets/images/flowchart_model.png)
---


## 2. 주요 기능

### 2.1 데이터 수집

- **웹 데이터 크롤링**: Selenium 및 Scrapy 사용
- **정제된 텍스트 처리**: 한글 텍스트만 추출

### 2.2 모델 학습 및 관리

- **모델 개발**:
    - Kc-BERT 및 Kc-ELECTRA 기반 Fine-tuning
    - NSMC 데이터셋 사용
- **MLflow를 활용한 관리**:
    - 실험 추적 및 모델 버전 관리

### 2.3 모델 서빙

- **FastAPI**: 실시간 감성 예측 API 제공
- **Airflow**: 배치 작업 관리

### 2.4 프론트엔드

- **Streamlit**: 사용자가 모델을 직접 테스트할 수 있는 인터페이스 제공

---

## 3. 디렉토리 구조 및 설정

```plaintext
project_root/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py             # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── base_model.py         # Model architecture definitions
│   │   └── kcbert_model.py       # KcBERT Model architecture definitions
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py
│   │   └── mlflow_utils.py      # MLflow integration utilities
│   └── data/
│       ├── __init__.py
│       └── base_dataset.py
│       └── nsmc_dataset.py     # nsmc dataset script
├── ──  train.py              # train module script
│   └── inference.py          # inference module script
├── configs/
│   ├── config.yaml          # Configuration files
│   └── model_registry.json  # Model registry files
├── mlruns/                  # mlflow artifacts files folder
├── init-scripts/
│   ├── init.sh              # Docker init file
├── dags/
│   └── dags.py             # dags script for airflow
├── app.py                   # streamlit web gui for model test & management
├── requirements.txt 
├── README.md
├── docker-compose.yml 
├── Dockerfile
└── .env                     # Environment variables for slack webhook - docker
```

### 📁 src
- 주요 모듈 코드 보관
  - **config.py**: 프로젝트 설정 관리
  - **data/**: 데이터 관련 코드
  - **models/**: 모델 아키텍처 정의
  - **utils/**: 유틸리티 함수 모음
### 📁 data
 - raw/: raw data
     - `data, models 폴더 및 파일이 없는 경우에도 [train.py](http://train.py) 실행시 저절로 데이터,모델 다운받아 실행`
 - processed/: processed data
### 📁 models
- Pretrained models
### 📁 configs
- YAML 기반 설정 파일
    - **데이터셋 종류**: 사용할 데이터셋 종류 설정 (기본값: NSMC - 네이버 영화 리뷰)
    - **모델 설정**: 사용할 모델 및 학습 파라미터 설정 (기본값: KcBERT)
    - **기타 파라미터**:
        - `dataset_sampling_rate`: 빠른 실험을 위한 데이터셋 샘플링 비율
        - `max_length`: 모델 입력의 최대 길이
        - `register_threshold`: 모델 등록을 위한 최소 기준
        - `unfrozen_layers`: 학습 시 언프리즈할 레이어 수
    
    - `requirements.txt`: 프로젝트 의존성
    - `.env`: 환경 변수
    - `README.md`: 프로젝트 문서
- JSON 기반 Model 관리 파일
### 📁 mlruns
- MLflow 실행 결과 metadata 저장 폴더
### 📁 mlartifacts
- MLflow 실행 결과 Model 등 아티팩트 저장 폴더
---
## 4. 기술 스택

### 4.1 데이터 수집 및 전처리

- **라이브러리**: Selenium, Scrapy
- **정제 작업**:
    - HTML 태그 및 특수 문자 제거
    - 연속 공백 및 이모지 제거

### 4.2 NLP 모델

- **모델**: Kc-BERT, Kc-ELECTRA
- **데이터셋**: NSMC (Naver Sentiment Movie Corpus)
- **모델 성능 평가 기준**: F1 Score ≥ 0.7

### 4.3 MLOps

- **MLflow**: 모델 버전 관리
    - 실험 및 모델 파라미터 추적
    - MLflow Model Registry, 별도의 model_registry.json 파일 정보를 싱크, 모델 버전 관리
    - **모델 단계**:
        - **Candidate**: 새로 훈련된 모델
        - **Champion**: 프로덕션 준비 완료된 모델
        - **Archived**: 사용 중단된 모델
- **Docker & Airflow**: 자동화 파이프라인 구축
    - Airflow 컨테이너 자동 생성
    - Slack webhook 연결 및 계정 자동 생성
      - 모델 등록 알림 및 모델 테스트 결과 알림

### 4.4 모델 서빙

- **FastAPI**: 실시간 추론 API
- **API 응답 예시**:
    
    ```json
    {
      "prediction": "positive",
      "confidence": 0.95
    }
    ```
---

## 5. 사용한 데이터셋 및 모델

### 5.1 NSMC 데이터셋

- **출처**: [GitHub - e9t/nsmc](https://github.com/e9t/nsmc)
- **내용**: 네이버 영화 리뷰 텍스트와 감성 레이블 (긍정: 1, 부정: 0)
- **특징**:
    - 총 20만 개의 데이터 (훈련: 15만 개, 테스트: 5만 개)
    - 구어체, 비속어, 오탈자 등이 포함된 정제되지 않은 데이터
- **사용 목적**: 한국어 감성 분석 모델 학습

### 5.2 Kc-BERT 모델

- **출처**: [GitHub - Beomi/KcBERT](https://github.com/Beomi/KcBERT)
- **아키텍처**: BERT-base
- **학습 데이터**:
    - 한국어 위키백과
    - 뉴스 기사
    - 온라인 커뮤니티 댓글 및 SNS 데이터
- **특징**:
    - 구어체 및 비정형 표현 이해에 강점
    - 감성 분석, 문서 분류 등 한국어 NLP 태스크에 적합

### 5.3 Kc-ELECTRA 모델

- **출처**: [GitHub - Beomi/KcBERT](https://github.com/Beomi/KcBERT)
- **아키텍처**: ELECTRA-base (Generator-Discriminator)
- **학습 데이터**:
    - 한국어 위키백과
    - 뉴스 기사
    - 온라인 커뮤니티 댓글 및 SNS 데이터
- **특징**:
    - BERT 대비 경량화
    - 빠른 학습과 추론이 필요한 환경에서 적합

---

## 6. 실행 가이드

### 6.1 환경 설정
- **실험 및 테스트 환경**
    - Linux, Windows 10, Mac OS
- **Python 버전**
    - Python 3.10
- **Conda 환경 생성**:

```bash
conda create -n ml4 python=3.10
conda activate ml4
pip install -r requirements.txt
```

### 6.2 설정
`config/config.yaml` 파일을 열어 필요한 설정을 확인하고 실험에 맞게 수정

- **데이터셋 설정**: `dataset` 섹션에서 데이터셋 종류와 샘플링 비율 등을 설정
- **모델 설정**: `model` 섹션에서 사용할 모델 이름과 학습 파라미터 등을 설정
- **학습 설정**: `train` 섹션에서 에포크 수, 배치 크기 등을 설정
  
### 6.3 MLflow 서버 실행

```bash
mlflow ui --host 127.0.0.1 --port 5050
```

- 브라우저에서 `http://127.0.0.1:5050`에 접속

### 6.3 모델 학습: train 모듈

```bash
python train.py
```
- 구조
```bash
        Args:
            interactive: 대화형 추론 및 모델 관리 기능 활성화 여부 (옵션: default = False)
        Returns:
            dict: 학습 결과 정보
            {
                'run_id': str,
                'metrics': dict,
                'run_name': str,
                'model': PreTrainedModel,
                'tokenizer': PreTrainedTokenizer,
                'data_module': NSMCDataModule
            }
```
- 사용 예시
```bash
from src.train import SentimentTrainer
# 기본 학습. 설정은 config.yaml 에서 project 항목
trainer = SentimentTrainer()
result = trainer.train()
```
### 6.4 추론 모듈: inference 모듈

**추론 코드**:
- 구조
```bash
        Args:
            text: 입력 텍스트 또는 텍스트 리스트
            return_probs: 확률값 반환 여부
            
        Returns:
            Dict 또는 Dict 리스트: 예측 결과
            {
                'text': str,  # 원본 텍스트
                'label': str,  # '긍정' 또는 '부정'
                'confidence': float,  # 예측 확신도
                'probs': {  # 각 레이블별 확률 (return_probs=True인 경우)
                    '긍정': float,
                    '부정': float
                }
            }
```
- 사용 예시
```bash
from src.inference import SentimentPredictor
predictor = SentimentPredictor() # default: Production (최신 모델)
texts = ["다시 보고 싶은 영화", "별로에요"]
results = predictor.predict(texts)

```

**결과**:

```
{'text': '다시 보고 싶은 영화', 'label': 'positive', 'confidence': 0.93}
{'text': '별로에요', 'label': 'negative', 'confidence': 0.85}
```


### 6.5 모델 관리

학습 완료 후 터미널에 나타나는 모델 관리 관련 메시지에 따라 CLI에서 숫자 또는 `y/n`을 입력하여 모델을 관리.

- **모델 등록**: 모델을 레지스트리에 등록할지 여부 선택
- **모델 단계 설정**: 모델의 단계(stage)를 설정 (예: None, Staging, Production)

### 6.6 결과 확인

- **MLflow UI**: 브라우저에서 실험 결과, 메트릭, 파라미터 및 아티팩트를 확인.
- **폴더 구조**:
    - `mlruns/` 폴더에 실행(run) 관련 로그와 메트릭이 저장.
    - `exp id / exp id / artifacts /` 폴더에 모델 파일 등 아티팩트가 저장.
    - `config/model_info.json` 파일에서 등록된 모델의 단계(stage)를 확인.

### 6.7 Streamlit 인터페이스 실행

```bash
streamlit run app.py
```

- 웹 브라우저에서 인터페이스를 통해 모델 테스트


### 6.8 Docker & Airflow Setup
- init-scripts/init.sh 파일 실행 시 자동으로 실행되는 초기 설정 파일
- Airflow, Slack webhook 연결 계정 자동 생성

#### 📁 init-scripts
- **init.sh**: 초기 설정 파일
#### 📁 
- **Dockerfile**: Docker 컨테이너 설정 파일
- **docker-compose.yml**: Docker 컨테이너 관리 파일

#### 사용법 및 명령어

Airflow를 Docker로 설정하려면 아래 명령어를 실행:

```bash
docker-compose up --build -d

```

#### Slack Webhook 설정

Airflow에서 Slack Webhook을 사용하려면 다음 정보를 `.env` 파일에 저장:

 `.env` 파일 예시

```
env
코드 복사
# Slack Webhook Token 설정
SLACK_WEBHOOK_TOKEN=PUT YOUR SLACK TOKEN

# Airflow 설정
AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////usr/local/ml4/airflow.db
AIRFLOW__PROVIDERS__SLACK__WEBHOOK_CONN_ID=slack_webhook

```

만약 자동 설정이 안되는 경우, Slack Webhook 연결을 위해 Airflow의 Connection ID를 아래와 같이 설정

- **Connection ID**: `slack_webhook`
- **Token**: `.env` 파일에 설정된 `SLACK_WEBHOOK_TOKEN` 값 사용

---

#### Airflow 계정 자동 생성

Airflow 초기 설정 시 다음 기본 계정이 자동으로 생성:

- **ID**: `admin`
- **Password**: `admin`
  
#### 추가 참고 사항

- `docker-compose.yml` 파일이 제대로 구성되어 있는지 확인.
- Airflow를 실행한 후 웹 UI에서 Slack Webhook Connection 설정이 올바르게 등록되었는지 확인.

---
### 6.9 CrawlerService

Python 기반 네이버 블로그 크롤러 서비스입니다. 검색어(Query)를 입력받아 크롤링을 수행하고, 결과를 JSON 또는 CSV 형식으로 저장합니다.

---

#### 주요 기능

- **Scrapy 기반 크롤링**: Scrapy와 Selenium을 사용하여 데이터를 효율적으로 크롤링합니다.
- **결과 저장**: 크롤링 결과를 JSON 및 CSV 형식으로 저장.
- **사용 편의성**: Python 함수 호출 또는 명령줄 인터페이스로 실행 가능.

---

#### 설치 방법

1. 저장소 클론
   git clone <repository_url>
   cd CrawlerService

2. 가상환경 설정
   python3.12 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate

3. 의존성 설치
   pip install -r requirements.txt

---

#### 실행 방법

1. Scrapy 명령어로 실행
   검색어(Query)를 입력하여 크롤링을 실행하고 결과를 저장합니다.
   [예시]
   ```bash
   scrapy crawl naver_blog_spider
   scrapy crawl naver_blog_spider -a verbose=True
   scrapy crawl naver_blog_spider -a verbose=True -o output/naver_blog_crawling.csv
   scrapy crawl naver_blog_spider -a verbose=True -a query=스위스,독일
   scrapy crawl naver_blog_spider -a verbose=True -a query=스위스,독일 -a max_results=4
   scrapy crawl naver_blog_spider -a verbose=True -a query=이탈리아 -a max_results=2
   scrapy crawl naver_blog_spider -a verbose=True -a query=일본,중국,베트남 -a max_results=5
   scrapy crawl naver_blog_spider -a verbose=True -a query=일본,중국,베트남 -a max_results=5 -o output/test001.csv
   ```
   [ERROR]
   아래와 같은 경우 error가 발생합니다.
   ```bash
   # ######################################################################
   crawl: error: running 'scrapy crawl' with more than one spider is not supported
   # ######################################################################
   scrapy crawl naver_blog_spider -a verbose=True -t csv  [crawl: error]
   scrapy crawl naver_blog_spider -a verbose=True -a query= 스위스,독일  [crawl: error]
   # ######################################################################
   ```
2. Python 코드로 실행
   crawler_service.py의 run_crawler 함수를 호출하여 실행합니다:

   [test_crawler.py]
   ```bash
   from CrawlerService.crawler_service import run_crawler
   ```
   #### 크롤링 테스트
   ```bash
   output_filename = "naver_blog_crawling"
   result = run_crawler(query="일본 도쿄,한국 서울", verbose=True, max_results=4, output_filename=output_filename)
   print(f"크롤링 결과 저장 (CSV file): output/{output_filename}.csv")
   print(f"크롤링 결과 저장 (JSON file): output/{output_filename}.json")
   ```
---

#### 개발 및 디버깅

1. 의존성 분석
   pipdeptree를 사용하여 프로젝트 의존성을 분석할 수 있습니다:
   pip install pipdeptree
   pipdeptree

2. 로깅
   크롤링 과정에서 발생하는 모든 로그는 logs/ 폴더에 저장됩니다.

---

##### 프로젝트 디렉토리 구조

프로젝트 디렉토리 구조는 다음과 같습니다:

```bash
CrawlerService
├── CrawlerService
│   ├── __init__.py
│   ├── crawler_service.py
│   ├── items.py
│   ├── pipelines.py
│   ├── settings.py
│   ├── setup.py
│   └── spiders
│       ├── __init__.py
│       └── naver_blog_spider.py
├── README.md
├── app
│   └── main.py
├── logs
├── output
├── requirements.txt
├── scrapy.cfg
├── test_crawler.py
├── .gitignore
└── venv
```

---

## 7. Result 주요 성과 및 개선 방향
### 실시간 감성 분석 서비스 데모 시연
- 감성 챗봇과 대화 분석 기능 등 추가
![실시간 감성 분석 서비스 데모](https://github.com/UpstageAILab5/upstageailab-ml-pjt-ml_p4/blob/main/assets/images/demo.png)

### 7.1 최종 성과

- **실시간 감성 분석 서비스 구축**:
    - 한국어 텍스트의 긍정/부정 분류 모델 개발
    - 정제되지 않은 한국어 텍스트에 대한 감성 분류에서 높은 정확도와 안정성 달성
- **MLOps 파이프라인 구축**:
    - 모델, 데이터 관리 및 MLOps 인프라 전반 이해 및 구현

### 7.2 도전 과제 및 해결 방안

- **데이터셋**:
    - **문제**: 실시간 데이터 수집 및 모델 학습의 어려움
    - **해결**: 목표 데이터 타입과 유사한 특성을 가진 대규모 레이블 데이터셋을 이용, 기본 모델을 학습
- **MLOps 통합**:
    - **문제**: 모델 버전 관리 및 배포의 원활한 연계
    - **해결**: MLflow를 활용한 실험 추적 및 모델 레지스트리 구현
- **모델 배포**:
    - **문제**: 실시간 추론을 위한 낮은 지연 시간 확보
    - **해결**: FastAPI와 비동기 처리로 모델 서빙 최적화

### 7.3 향후 개선 방향
1. **Airflow 자동화 통합**: 데이터 수집 및 전처리, 모델 학습, 배포 자동화
2. **다중 감정 분류 확장**: 중립 및 복합 감정을 추가로 예측
3. **모델 일반화 향상**: 다양한 데이터셋 사용
4. **프론트엔드 개선**: 사용자 경험 최적화
5. **다중 유저 정보 관리**: 다양한 유저 정보 관리 및 모델 테스트


### 7.3 프레젠테이션 및 회의록
- **Report**
  - [Upstage AI Lab Fast-Up Report](https://www.notion.so/Fast-Up-Team-Report-1-c8476dbc79234d85b275fc532dfbbbdd?pvs=4)
- **Presentation**
  - [pdf](https://github.com/UpstageAILab5/upstageailab-ml-pjt-ml_p4/blob/main/assets/docs/%5B%ED%8C%A8%EC%8A%A4%ED%8A%B8%EC%BA%A0%ED%8D%BC%EC%8A%A4%5D-Upstage-AI-Lab-5%EA%B8%B0_%ED%94%84%EB%A1%9C%EC%A0%9D%ED%8A%B8-ML4.pdf)
- **Meeting Log**
  - [회고 및 미팅 로그](https://www.notion.so/a6667a35aec348a2bfbba76bfcdd6343?v=2c6a809bc92b4dd4a77247c99caba688&pvs=4)
  - [피어 세션 및 멘토링](https://www.notion.so/37086a60fda642efb175576e0887fd7c?v=e22cf0fd881a481c8b0c6dab27dfac28&pvs=4)
---

## 8. 참고 자료
### 8.1 데이터 수집
- **Selenium 문서**: [Selenium](https://selenium-python.readthedocs.io/)
- **Scrapy 문서**: [Scrapy](https://docs.scrapy.org/en/latest/)
### 8.2 데이터셋
- **NSMC 데이터셋**: [GitHub - e9t/nsmc](https://github.com/e9t/nsmc)
### 8.3 모델
- **Hugging Face**: [Hugging Face](https://huggingface.co/)
- **Kc-BERT 모델**: [GitHub - Beomi/KcBERT](https://github.com/Beomi/KcBERT)
- **KC-ELECTRA 모델**: [Hugging Face - beomi/KcELECTRA-base](https://huggingface.co/beomi/KcELECTRA-base)
### 8.4 MLOps
- **MLflow 문서**: [MLflow](https://mlflow.org/docs/latest/index.html)
### 8.4 프론트엔드
- **Streamlit 문서**: [Streamlit](https://docs.streamlit.io/)
### 8.5 백엔드  
- **FastAPI 문서**: [FastAPI](https://fastapi.tiangolo.com/)
### 8.6 배포
- **Airflow 문서**: [Airflow](https://airflow.apache.org/)
