# MLOps:

# I. Model Management

## 1. 프로젝트 구조 및 설정

```plaintext
project_root/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py             # Configuration management
│   ├── models/
│   │   ├── __init__.py
│   │   └── kcbert_model.py       # KcBERT Model architecture definitions
│   ├── utils/
│   │   ├── __init__.py
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
├── mlruns/                  # mlflow metadata files folder
├── mlartifacts/                  # mlflow artifacts files folder
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

## 1.1 주요 컴포넌트 설명
### 📁 src
주요 소스 코드, 모듈 위치
- **config.py**: 프로젝트 설정 관리
- **data/**: 데이터 관련 코드
- **models/**: 모델 아키텍처 정의
- **utils/**: 유틸리티 함수 모음
- **train.py**: Model Train 모듈. 직접 실행 가능.
- **inference.py** 추론 모듈. 직접 실행 가능.
- **app.py** 학습 완료된 모델 이용, streamlit web GUI 로 모델 관리 및 추론 테스트
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
### 📁 Mlflow Folder
아래 폴더, 프로젝트 루트에 저장 및 보관. Model Train, inference 시 사용
- **config/model_registry.json**: 학습된 Model 메타데이터 저장
- **mlruns/**: MLflow 메타데이터 저장
- **mlartifacts/**: 모델과 아티팩트 저장
 - **mlartifacts/{run_id}/artifacts/model/data/model.pth**: 실제 모델 저장되는 위치

## 1.2 개발 및 테스트 환경 설정
- Python 3.10
- Windows 10, Linux, Mac OS (Windows 에서 학습시 artifact 경로명 이슈)
- MLflow를 통한 실험 관리

### Conda 환경 생성

Python 3.10 버전의 Conda 가상 환경을 생성하고 활성화.

```bash
conda create -n ml4 python=3.10
conda activate ml4
```

### 필요 모듈 설치
프로젝트에 필요한 의존성 모듈을 설치

```bash
pip install -r requirements.txt
```
## 2. 실행 순서
주의 사항
- Project Root 에서 실행 가정
- mlflow, streamlit, airflow 등 필요한 서버 충돌없이 제대로 가동 중 확인

### 2.1 설정 파일 확인 및 수정

`config/config.yaml` 파일을 열어 필요한 설정을 확인하고 실험에 맞게 수정
- **프로젝트 설정**: `project` 섹션에서 사용할 모델 이름, 데이터셋 이름 설정
- **데이터셋 설정**: `dataset` 섹션에서 데이터셋 종류와 샘플링 비율 등을 설정
- **모델 설정**: `models` 섹션에서 사용할 모델 이름 별 학습 파라미터 등을 설정
- **학습 설정**: `models` 하위 각 모델 `training` 섹션에서 에포크 수, 배치 크기 등을 설정
- 
### 2.2 MLflow 서버 실행

프로젝트 루트 디렉토리에서 다음 명령어를 실행하여 MLflow UI를 시작

```bash
mlflow ui --host 127.0.0.1 --port 5050
```

브라우저에서 [http://127.0.0.1:5050](http://127.0.0.1:5050/) 에 접속하여 MLflow UI에 접근

### 2.3. Train 모듈 
- 구조
```bash
   Args:
            config_path (str, optional): 설정 파일 경로. Defaults to "config/config.yaml".
            model_name (Optional[str], optional): 사용할 모델 이름. 미지정시 config에서 가져옴.
            dataset_name (Optional[str], optional): 사용할 데이터셋 이름. 미지정시 config에서 가져옴.
            sampling_rate (Optional[float], optional): 데이터 샘플링 비율. 미지정시 config에서 가져옴.
            interactive (bool, optional): 대화형 모드 활성화 여부. Defaults to False.
            reset_mlflow (bool, optional): MLflow 환경 초기화 여부. Defaults to False.

        Returns:
            Dict[str, Any]: 학습 결과를 포함하는 딕셔너리
                - run_id (str): MLflow 실행 ID
                - metrics (Dict): 검증 메트릭 결과
                - run_name (str): 실행 이름
                - model: 학습된 모델 객체
                - tokenizer: 토크나이저 객체
                - data_module: 데이터 모듈 객체
```
- 사용 예시
```bash
from src.train import ModelTrainer
# 기본 학습. 설정은 config.yaml 에서 project 항목
trainer = ModelTrainer()
result = trainer.train_model()
```
### 2.4. Inference 모듈
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
### 2.5 모델 학습 시작
터미널에서 다음 명령어를 실행하여 모델 학습을 시작하거나, IDE에서 `train.py`를 실행:

```bash
python train.py
```

### 2.6 모델 관리
학습 완료 후 터미널에 나타나는 모델 관리 관련 메시지에 따라 CLI에서 숫자 또는 `y/n`을 입력하여 모델을 관리.

- **모델 등록**: 모델을 레지스트리에 등록할지 여부 선택
- **모델 단계 설정**: 모델의 단계(stage)를 설정 

### 2.7 결과 확인

- **MLflow UI**: 브라우저에서 실험 결과, 메트릭, 파라미터 및 아티팩트를 확인.
### MLflow 모델 버전 관리: Alias 시스템 (구 Stage)
다음과 같은 alias 시스템을 사용:
- **Candidate**: 새로 학습된 모델 중 성능 기준을 충족하여 검토 대상이 된 모델 (Staging)
- **Champion**: 현재 프로덕션에서 사용 중인 최고 성능의 모델 (Production)
- **Archive**: 이전에 사용되었거나 더 이상 사용하지 않는 모델들의 보관소 (Archive)

### 2.8 Streamlit App 실행
- linux server 에서 실행 후 로컬에서 접속 시
 - vs code 등에 가능한 포트를 {port} 에 넣어 후 아래 명령어로 실행
   ```bash
   streamlit run app.py --server.port {port} --server.address 0.0.0.0
   ```
 - 접속할 클라이언트 쪽에서는 CLI 에서 다음과 같이 실행
   ```bash
   ssh -L {port}:localhost:{port} {user}@{address}
   ```
   ```bash
 - 아래 주소로 접속 가능
   http://localhost:{port}
   ```
---
이 가이드를 따라 프로젝트를 실행하고 모델을 학습 및 관리. 필요에 따라 `config.yaml` 파일의 설정을 조정하여 실험을 진행

## 3. 프로젝트 세부 사항

### 주요 설정 항목 설명

- **데이터셋 종류** (`dataset.name`): 사용할 데이터셋의 이름을 지정. 기본값은 `nsmc`
- **모델 이름** (`model.name`): 사용할 사전 학습된 모델의 이름을 지정. 기본값은 `KcBER`
- **데이터셋 샘플링 비율** (`dataset.sampling_rate`): 데이터셋의 일부만 사용하여 빠른 실험을 진행
- **최대 입력 길이** (`dataset.max_length`): 모델 입력 시퀀스의 최대 길이를 지정
- **모델 등록 최소 기준** (`model.register_threshold`): 모델을 레지스트리에 등록하기 위한 최소 성능 기준을 설정
- **언프리즈할 레이어 수** (`model.unfrozen_layers`): 모델 학습 시 업데이트할 레이어의 수를 지정

# II. Docker for Airflow Setup

## 사용법 및 명령어

Airflow를 Docker로 설정하려면 아래 명령어를 실행:

```bash
docker-compose up --build -d

```

---

## Slack Webhook 설정

Airflow에서 Slack Webhook을 사용하려면 다음 정보를 `.env` 파일에 저장:

### `.env` 파일 예시

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

## Airflow 계정 자동 생성

Airflow 초기 설정 시 다음 기본 계정이 자동으로 생성:

- **ID**: `admin`
- **Password**: `admin`
  
## 추가 참고 사항

- `docker-compose.yml` 파일이 제대로 구성되어 있는지 확인.
- Airflow를 실행한 후 웹 UI에서 Slack Webhook Connection 설정이 올바르게 등록되었는지 확인.

# III. Dataset & Model

## NSMC (Naver Sentiment Movie Corpus) 데이터셋

- **데이터 출처**: 네이버 영화 리뷰 데이터
- **데이터 구성**:
    - **총 데이터 수**: 200,000개
        - 훈련용 데이터: 150,000개
        - 테스트용 데이터: 50,000개
    - **레이블**: 긍정 (1), 부정 (0) 이진 분류
    - **내용**: 사용자 작성 영화 리뷰와 해당 감성 레이블
- **특징**:
    - 한국어 감성 분석을 위한 대표적인 공개 데이터셋
    - 리뷰는 한글과 공백으로만 구성되어 전처리 필요성이 적음
- **라이선스**: 공개 라이선스 (출처 표기 필요)

## KC-BERT 모델

- **모델명**: KC-BERT (Korean Comments BERT)
- **아키텍처**: BERT-base
- **언어**: 한국어
- **모델 크기**:
    - **파라미터 수**: 약 110M (1억 1천만 개)
- **학습 데이터**:
    - 한국어 위키백과
    - 뉴스 기사
    - 온라인 커뮤니티 댓글 및 SNS 데이터
- **특징**:
    - 구어체, 비속어 등 일상 언어에 대한 이해도 향상
    - 한국어 텍스트의 문맥적 의미 파악에서 우수한 성능 발휘
- **라이선스**: MIT 라이선스
