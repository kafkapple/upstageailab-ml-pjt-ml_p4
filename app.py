import streamlit as st
from src.config import Config
from src.utils.mlflow_utils import MLflowModelManager
from src.inference import SentimentPredictor
from src.models.kcbert_model import KcBERT
from src.models.kcelectra_model import KcELECTRA
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import time
import logging# 추가
import torch #추가
import requests #추가
import plotly.express as px #추가
import numpy as np #추가
from streamlit_chat import message  #추가
import streamlit as st
#추가
import csv# 추가
from io import StringIO#추가
import random
logging.basicConfig(level=logging.INFO)

#추가- 모델 성늠 향상  디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #추가 
# API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
# headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}"}

try:
    API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
    headers = {"Authorization": f"Bearer {st.secrets['HUGGINGFACE_TOKEN']}"}
except Exception as e:
    st.error("Hugging Face API 토큰이 설정되지 않았습니다. Streamlit Cloud의 Settings에서 'HUGGINGFACE_TOKEN'을 설정해주세요.")
    headers = {"Authorization": "Bearer "}


#Part 2/4 - 유틸리티 함수들:
# query 함수 수정

def query(payload):
    """Hugging Face API 호출"""
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        return response.json()
    except Exception as e:
        st.error(f"API 호출 오류: {str(e)}")
        return {"error": str(e)}

#유튜브 추천 메시지 
def get_sentiment_message(score):
    """감성 점수에 따른 메시지 반환"""
    if score >= 90:
        return "당신의 행복이 주변을 환하게 비추네요! 이 기쁨을 음악과 함께 나눠보세요 ✨"
    elif score >= 80:
        return "당신의 긍정적인 에너지가 느껴져요! 이 순간을 음악으로 더 특별하게 만들어보세요 🌟"
    elif score >= 70:
        return "좋은 기운이 가득하네요! 기분 좋은 음악과 함께 더 행복해지세요 🎵"
    elif score >= 60:
        return "밝은 에너지가 느껴져요. 음악과 함께 이 기분을 이어가보세요 🎶"
    elif score >= 50:
        return "평온한 마음이 느껴지네요. 감성적인 음악으로 더 깊어져보세요 💫"
    elif score >= 40:
        return "잠시 쉬어가도 괜찮아요. 위로가 되는 음악을 들려드릴게요 🌙"
    elif score >= 30:
        return "힘들 때도 있지만, 그대로의 당신이 충분히 아름다워요 🌷"
    elif score >= 20:
        return "당신의 마음에 작은 위로가 되고 싶어요. 이 음악을 들어보세요 💝"
    elif score >= 10:
        return "힘내세요. 당신은 하나밖에 없는 특별한 별같은 사람이에요 ⭐"
    else:
        return "가장 어두운 밤이 지나면 반드시 새벽이 옵니다. 당신 곁에 있을게요 🌅"

#유튜브 추천 함수   
def recommend_youtube_video(probs):
    # 긍정 확률을 0-100 스케일로 변환
    score = probs['긍정'] * 100
    
    # 감성에 따른 비디오 추천
    if score >= 90:
        videos = [
            "https://www.youtube.com/watch?v=ZbZSe6N_BXs",  # Happy - Pharrell Williams
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",  # Uptown Funk - Mark Ronson ft. Bruno Mars
            "https://www.youtube.com/watch?v=OPf0YbXqDm0"   # Can't Stop the Feeling
        ]
    elif score >= 80:
        videos = [
            "https://www.youtube.com/watch?v=mPVDGOVjRQ0",  # NewJeans - Super Shy
            "https://youtu.be/hzmUVRRKkiw?si=vM1lxyFLGIp8H7K2",  # 𝗧𝗮𝗶 𝗩𝗲𝗿𝗱𝗲𝘀 - 𝗔-𝗢-𝗞 
            "https://www.youtube.com/watch?v=JGwWNGJdvx8"   # Sugar - Maroon 5
        ]
    elif score >= 70:
        videos = [
            "https://www.youtube.com/watch?v=0lapF4DQPKQ",  # BTS - Boy With Luv
            "https://www.youtube.com/watch?v=7PCkvCPvDXk",  # Roar - Katy Perry
            "https://www.youtube.com/watch?v=lp-EO5I60KA"   # All About That Bass - Meghan Trainor
        ]
    elif score >= 60:
        videos = [
            "https://www.youtube.com/watch?v=YqeW9_5kURI",  # Lean On - Major Lazer & DJ Snake
            "https://www.youtube.com/watch?v=gdZLi9oWNZg",  # BTS - Dynamite
            "https://www.youtube.com/watch?v=J9NQFACZYEU"   # Counting Stars - OneRepublic
        ]
    elif score >= 50:
        videos = [
            "https://www.youtube.com/watch?v=ktvTqknDobU",  # Radioactive - Imagine Dragons
            "https://youtu.be/6k8cpUkKK4c?si=4UBBeuvgcb1vf8-j"#Bruno Mars - Count on Me 
            "https://www.youtube.com/watch?v=gdZLi9oWNZg",  # BTS - Dynamite
        ]
    elif score >= 40:
        videos = [
            "https://www.youtube.com/watch?v=2vjPBrBU-TM",  # Stay - Rihanna ft. Mikky Ekko
            "https://www.youtube.com/watch?v=450p7goxZqg",  # Let Me Love You - DJ Snake ft. Justin Bieber
            "https://www.youtube.com/watch?v=J_ub7Etch2U"   # Say You Won't Let Go - James Arthur
        ]
    elif score >= 30:
        videos = [
            "https://www.youtube.com/watch?v=3AtDnEC4zak",  # Shallow - Lady Gaga & Bradley Cooper
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",  # Perfect - Ed Sheeran
            "https://youtu.be/Mu_R2XlRLxQ?si=iL-42Zz-RBroVptr" #Andy Grammer - These Tears
        ]
    elif score >= 20:
        videos = [
            "https://www.youtube.com/watch?v=RgKAFK5djSk",  # See You Again - Wiz Khalifa ft. Charlie Puth
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",  # Uptown Funk - Mark Ronson ft. Bruno Mars
            "https://www.youtube.com/watch?v=OPf0YbXqDm0"   # Can't Stop the Feeling - Justin Timberlake
        ]
    elif score >= 10:
        videos = [
            "https://www.youtube.com/watch?v=ZbZSe6N_BXs",  # Happy - Pharrell Williams
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",  # Uptown Funk - Mark Ronson ft. Bruno Mars
            "https://www.youtube.com/watch?v=OPf0YbXqDm0"   # Can't Stop the Feeling - Justin Timberlake
        ]
    else:
        videos = [
            "https://youtu.be/VXp2dCXYrvQ?si=iL9Yh1xCascP5nMo" #데이식스 한페이지가 될 수 있게
            "https://www.youtube.com/watch?v=RgKAFK5djSk",  # See You Again - Wiz Khalifa ft. Charlie Puth
            "https://youtu.be/gGpPkfFN6pA?si=soYRxQKnv--4bExz" #𝗛𝗲𝗻𝗿𝘆 𝗠𝗼𝗼𝗱𝗶𝗲 - 𝗽𝗶𝗰𝗸 𝘂𝗽 𝘁𝗵𝗲 𝗽𝗵𝗼𝗻𝗲 
        ]
    
    return random.choice(videos)
    
#ui 함수
def create_gauge_chart(value, title):
    """게이지 차트 생성"""
    # 감성에 따른 색상 설정
    if value > 0.8:
        bar_color = "#2ecc71"  # 매우 긍정 - 진한 초록
        steps = [
            {'range': [0, 33], 'color': "#ff9999"},  # 연한 빨강
            {'range': [33, 66], 'color': "#ffeb99"},  # 연한 노랑
            {'range': [66, 100], 'color': "#99ff99"}  # 연한 초록
        ]
        threshold_value = 80
    elif value > 0.6:
        bar_color = "#3498db"  # 긍정 - 파랑
        steps = [
            {'range': [0, 33], 'color': "#ffb399"},  # 연한 주황
            {'range': [33, 66], 'color': "#fff099"},  # 연한 노랑
            {'range': [66, 100], 'color': "#99ffcc"}  # 연한 민트
        ]
        threshold_value = 60
    else:
        bar_color = "#e74c3c"  # 부정 - 빨강
        steps = [
            {'range': [0, 33], 'color': "#ff9999"},  # 연한 빨강
            {'range': [33, 66], 'color': "#ffcc99"},  # 연한 주황
            {'range': [66, 100], 'color': "#ffff99"}  # 연한 노랑
        ]
        threshold_value = 40

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 24}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': bar_color},
            'steps': steps,
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_value
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "black"}
    )
    
    return fig

def create_sentiment_chart(probs):
    """감성 확률 차트 생성"""
    df = pd.DataFrame({
        '감성': ['부정', '긍정'],
        '확률': probs * 100
    })
    
    fig = px.bar(df, x='감성', y='확률',
                 color='감성',
                 color_discrete_map={'긍정': 'green', '부정': 'red'},
                 text=df['확률'].apply(lambda x: f'{x:.1f}%'))
    
    fig.update_layout(
        title='감성 분석 결과',
        yaxis_title='확률 (%)',
        showlegend=False,
        height=400
    )
    return fig

    ##감정 정도에 따른 이모지 세분화 

def get_sentiment_emoji(sentiment, confidence):
    """감성에 따른 이모지 반환"""
    if sentiment == '긍정':
        if confidence > 0.8:
            return "😄"  # 매우 긍정
        elif confidence > 0.6:
            return "🙂"  # 긍정
        else:
            return "😊"  # 약한 긍정
    else:
        if confidence > 0.8:
            return "😢"  # 매우 부정
        elif confidence > 0.6:
            return "😕"  # 부정
        else:
            return "😐"  # 약한 부정






def initialize_session_state():
    """Initialize session state variables"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'total_predictions' not in st.session_state:
        st.session_state.total_predictions = 0
    if 'positive_count' not in st.session_state:
        st.session_state.positive_count = 0
    if 'negative_count' not in st.session_state:
        st.session_state.negative_count = 0
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'model_state_changed' not in st.session_state:
        st.session_state.model_state_changed = False

@st.cache_resource
def load_predictor(model_info):
    """Load model predictor"""
    try:
        # 모델 타입 확인
        model_name = model_info['params']['model_name']
        print(f"Debug: Loading model: {model_name}")
        
        predictor = SentimentPredictor(
            model_name=model_name,
            alias="champion",
            config_path="config/config.yaml"
        )
        
        print("Debug: Predictor loaded successfully")
        return predictor
        
    except Exception as e:
        import traceback
        print(f"Error loading model: {str(e)}")
        traceback.print_exc()
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_sentiment(text: str, predictor: SentimentPredictor):
    """Predict sentiment using predictor 텍스트 데이터 셋"""
    try:
        result = predictor.predict(text, return_probs=True)
        
        return {
            'label': result['label'],
            'confidence': result['confidence'],
            'probabilities': [
                result['probs']['부정'],
                result['probs']['긍정']
            ]
        }
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def update_statistics(sentiment: str):
    """Update prediction statistics"""
    st.session_state.total_predictions += 1
    if sentiment == "긍정":
        st.session_state.positive_count += 1
    else:
        st.session_state.negative_count += 1

def add_to_history(text: str, result: dict, model_info: dict):
    """Add prediction to history"""
    try:
        # 결과 구조 확인 및 변환
        if 'probabilities' in result:  # 이전 형식
            probs = {
                '긍정': result['probabilities'][1],
                '부정': result['probabilities'][0]
            }
        elif 'probs' in result:  # 새로운 형식
            probs = result['probs']
        else:
            raise ValueError("Invalid result format: missing probabilities")

        st.session_state.history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "sentiment": result['label'],
            "confidence": result['confidence'],
            "negative_prob": probs['부정'],
            "positive_prob": probs['긍정'],
            "model_name": model_info['run_name'],
            "model_stage": model_info['stage'],
            "model_version": model_info['version']
        })
        
    except Exception as e:
        print(f"Error adding to history: {str(e)}")
        print(f"Result structure: {result}")
        import traceback
        traceback.print_exc()

def display_model_info(model_info):
    """Display model information in sidebar"""
    st.sidebar.subheader("Selected Model Info")
    st.sidebar.write(f"Model: {model_info['run_name']}")
    st.sidebar.write(f"Stage: {model_info['stage']}")
    
    st.sidebar.subheader("Model Metrics")
    for metric, value in model_info['metrics'].items():
        st.sidebar.metric(metric, f"{value:.4f}")
    
    st.sidebar.write(f"Registered: {model_info['timestamp']}")

def display_statistics():
    """Display prediction statistics"""
    st.sidebar.subheader("Prediction Statistics")
    total = st.session_state.total_predictions
    if total > 0:
        pos_ratio = (st.session_state.positive_count / total) * 100
        neg_ratio = (st.session_state.negative_count / total) * 100
        
        col1, col2, col3 = st.sidebar.columns(3)
        col1.metric("Total", total)
        col2.metric("긍정", f"{pos_ratio:.1f}%")
        col3.metric("부정", f"{neg_ratio:.1f}%")

def display_model_management(model_manager, model_name: str):
    """Display model management interface"""
    st.subheader("모델 관리")
    
    # Get all model versions
    models = model_manager.load_model_info()
    if not models:
        st.warning("등록된 모델이 없습니다.")
        return
    
    # Create DataFrame for better display
    df = pd.DataFrame(models)
    df['model_id'] = df.index + 1
    
    # Reorder columns
    columns = [
        'model_id', 'run_name', 'stage', 'metrics', 
        'timestamp', 'version', 'run_id'
    ]
    df = df[columns]
    
    # Format metrics column
    df['metrics'] = df['metrics'].apply(
        lambda x: f"F1: {x.get('val_f1', 0):.4f}"
    )
    
    # Stage name mapping
    stage_map = {
        'champion': '운영 중',
        'candidate': '검증 중',
        'archived': '보관됨',
        'latest': '최신'
    }
    df['stage'] = df['stage'].map(stage_map)
    
    # Add styling
    def color_stage(val):
        colors = {
            '운영 중': '#99ff99',
            '검증 중': '#ffeb99',
            '보관됨': '#ff9999',
            '최신': '#ffffff'
        }
        return f'background-color: {colors.get(val, "#ffffff")}; color: black'
    
    styled_df = df.style.applymap(
        color_stage,
        subset=['stage']
    )
    
    # Display models table
    st.dataframe(
        styled_df,
        column_config={
            "model_id": "모델 ID",
            "run_name": "모델 이름",
            "stage": "상태",
            "metrics": "성능 지표",
            "timestamp": "등록 시간",
            "version": "버전",
            "run_id": "실행 ID"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Model management controls
    st.markdown("---")
    st.subheader("상태 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_model_id = st.selectbox(
            "관리할 모델 선택",
            options=df['model_id'].tolist(),
            format_func=lambda x: f"Model {x}: {df[df['model_id']==x]['run_name'].iloc[0]}"
        )
        
        selected_model = df[df['model_id'] == selected_model_id].iloc[0]
        
        st.write("현재 정보:")
        st.write(f"- 모델: {selected_model['run_name']}")
        st.write(f"- 상태: {selected_model['stage']}")
        st.write(f"- 버전: {selected_model['version']}")
    
    with col2:
        new_stage = st.selectbox(
            "변경할 상태",
            options=['champion', 'candidate', 'archived'],
            format_func=lambda x: stage_map.get(x, x)
        )
        
        if st.button("상태 변경", type="primary"):
            try:
                selected_model = df[df['model_id'] == selected_model_id].iloc[0]
                version = str(selected_model['version'])
                
                print(f"\nDebug: Changing model state")
                print(f"Debug: Selected model version: {version}")
                print(f"Debug: New state: {new_stage}")
                
                if new_stage == 'champion':
                    model_manager.promote_to_production(model_name, version)
                elif new_stage == 'candidate':
                    model_manager.promote_to_staging(model_name, selected_model['run_id'])
                elif new_stage == 'archived':
                    model_manager.archive_model(model_name, version)
                
                # 상태 변경 후 강제 새로고침
                st.success(f"모델 상태가 {stage_map[new_stage]}(으)로 변경되었습니다.")
                time.sleep(1)  # UI 업데이트를 위한 짧은 대기
                st.rerun()
                
            except Exception as e:
                st.error(f"상태 변경 중 오류가 발생했습니다: {str(e)}")
                print(f"Error details: {str(e)}")
                import traceback
                traceback.print_exc()

def main():
    # 페이지 설정
    st.set_page_config(
        page_title="너의 기분은 어때?", #수정
        page_icon="🤗", #수정 로봇에서 
        layout="wide"
    )
    
      
    st.markdown("""
         <style>
            .sidebar .sidebar-content {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                font-size: 14px;
            }
            .stMarkdown {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                font-size: 14px;
            }
            .sidebar .sidebar-content .stMetric {
                font-size: 13px;
            }
            h1 {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                font-size: 28px;
                font-weight: 1000;
            }
            h2 {
                font-family: 'Helvetica Neue', Arial, sans-serif;
                font-size: 20px;
                font-weight: 500;
            }
            /* 탭 스타일링 */
            .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
                font-size: 16px;
                padding: 0px 10px;
                transition: color 0.3s ease;
            }
            .stTabs [data-baseweb="tab-list"] button:hover [data-testid="stMarkdownContainer"] p {
                color:   #F08080;
            }
            /* 선택된 탭 스타일 */
            .stTabs [data-baseweb="tab-list"] [aria-selected="true"] [data-testid="stMarkdownContainer"] p {
                color: #2F2F2F;  /* 텍스트 색상 */
                font-weight: bold;  /* 텍스트 굵게 */
            }
            /* 탭 선택 표시줄 색상 변경 */
            .stTabs [data-baseweb="tab-list"] [aria-selected="true"]::before {
                background-color: #D3B8E6 !important;
            }
            /* 탭 하단 구분선 색상 */
            .stTabs [data-baseweb="tab-list"] {
                border-bottom-color: #D3B8E6 !important;
            }
            /* 상태바 스타일링 */
            div.stProgress > div > div > div {
                width: 50% !important;
                margin: 0 auto;
            }
            div.stProgress > div > div > div > div {
                height: 8px;
                background-color: #2ecc71;
                border-radius: 4px;
            }
            div.stProgress > div > div > div {
                background-color: #f0f2f6;
            }
            </style>
        """, unsafe_allow_html=True)
    
    st.title("AI 감성 분석 서비스 ")
    
    # Config 및 모델 관리자 초기화
    config = Config()
    model_manager = MLflowModelManager(config)
    
    # 모델 정보 새로 로딩
    model_infos = model_manager.load_model_info()
    
    # 캐시 무시하고 현재 상태 가져오기
    selected_model_info = model_manager.load_production_model_info()
    if not selected_model_info:
        st.warning("운영 중인 모델이 없습니다. 최신 모델을 사용합니다.")
        selected_model_info = model_infos[-1]
    
    # 탭 생성
    tab_predict, tab_history, tab_manage,tab4 = st.tabs(["예측", "히스토리", "모델 관리","AI 감성 챗봇와 영어공부하기"])
    
    with tab_predict:
        # 모델 정보 표시
        with st.sidebar:
            st.markdown("### 현재 모델")
            st.markdown(f"**모델명**: {selected_model_info['run_name']}")
            st.markdown(f"**상태**: {selected_model_info['stage']}")
            st.markdown(f"**등록일**: {selected_model_info['timestamp']}")
            
            if 'metrics' in selected_model_info:
                st.markdown("### 성능 지표")
                metrics = selected_model_info['metrics']
                
                # 메트릭 값 포맷팅 및 세로로 표시
                for metric, value in metrics.items():
                    st.markdown(
                        f"<div style='font-size: 13px;'>{metric}: "
                        f"<span style='font-family: monospace;'>{value:.2f}</span></div>",
                        unsafe_allow_html=True
                    )
        
        # 예측 UI # 수정 
    #     2개 컬럼에서 3개 컬럼으로 변경
    # 첫 번째 컬럼에 이모지와 게이지 차트 추가
    # 두 번째 컬럼은 기존 막대 차트 유지
    # 세 번째 컬럼에 새로운 감성 차트 추가
    # 감성 강도에 대한 설명 텍스트 추가
        try:
            predictor = SentimentPredictor(
                model_name=config.project['model_name'],
                alias=selected_model_info['stage']
            )
            
            # 예측 입력 영역
            text = st.text_area(
                "분석할 텍스트를 입력하세요",
                height=100,
                help="여러 줄의 텍스트를 입력할 수 있습니다."
            )
            
            if text and st.button("분석", type="primary"):
                result = predictor.predict(text, return_probs=True)
                
                # 상단 결과 표시 (2개 컬럼)
                col1, col2  = st.columns(2)
                
                with col1:
                    

                    # # YouTube 비디오 URL
                    # youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

                    # # Streamlit 앱에 YouTube 비디오 임베드
                    # st.video(youtube_url)
                    # 감성과 이모지 표시
                    emoji = get_sentiment_emoji(result['label'], result['confidence'])
                    st.markdown(f"### {result['label']} {emoji}")
                    # st.metric("확신도", f"{result['confidence']:.1%}")
                    
                    # 게이지 차트
                    # 게이지 차트
                    gauge_fig = create_gauge_chart(
                        result['confidence'],
                        ""
                    )
                    gauge_fig.update_layout(
                        title=dict(
                            text="감성 지수",
                            font=dict(size=16),  # 제목 글자 크기를 16으로 줄임
                            y=0.9
                        )
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True)
                
                with col2:
                    # 감정 단어 버블 차트
                    pos_prob = result['probs']['긍정'] * 100
                    
                    import random
                    import math
                    import numpy as np

                    # 확률 구간에 따른 단어 설정
                    if pos_prob >= 90:
                        core_word = '환희'
                        related_words = ['행복', 'happy', '기쁨', '감동', '축복', 'joy', '희열', '감격', '황홀', '행운']
                        color = '#2ecc71'
                    elif pos_prob >= 70:
                        core_word = '행복'
                        related_words = ['즐거움', 'smile', '설렘', '만족', '기대', '희망', '좋음', '신남', '상쾌', '기분좋음']
                        color = '#27ae60'
                    elif pos_prob >= 50:
                        core_word = '긍정'
                        related_words = ['편안', '따뜻', '평화', '안정', '여유', '밝음', '맑음', '산뜻', '포근', '온화']
                        color = '#16a085'
                    elif pos_prob >= 30:
                        core_word = '중립'
                        related_words = ['보통', '일상', '평범', '무난', '담담', '차분', '잔잔', '고요', '평온', '침착']
                        color = '#f39c12'
                    elif pos_prob >= 10:
                        core_word = '부정'
                        related_words = ['걱정', '불안', 'sad', '답답', '지침', '피곤', '고민', '혼란', '불편', '우울']
                        color = '#e67e22'
                    else:
                        core_word = '절망'
                        related_words = ['분노', 'angry', '슬픔', '고통', '비통', '실망', '좌절', '상처', '괴로움', '공포']
                        color = '#e74c3c'

                    def get_random_position(radius, min_distance=0.15):
                        """원 안에서 랜덤한 위치 생성"""
                        for _ in range(100):  # 최대 100번 시도
                            angle = random.uniform(0, 2 * math.pi)
                            r = random.uniform(0.2, radius)  # 최소 거리 설정
                            x = 0.5 + r * math.cos(angle)
                            y = 0.5 + r * math.sin(angle)
                            
                            # 기존 위치들과의 거리 확인
                            valid_position = True
                            for pos in existing_positions:
                                if math.sqrt((x - pos[0])**2 + (y - pos[1])**2) < min_distance:
                                    valid_position = False
                                    break
                            
                            if valid_position:
                                existing_positions.append((x, y))
                                return x, y
                        
                        # 적절한 위치를 찾지 못한 경우 기본값 반환
                        return 0.5 + radius * 0.7 * math.cos(angle), 0.5 + radius * 0.7 * math.sin(angle)

                    # 감정 단어 정의
                    emotions = []
                    existing_positions = []
                    
                    # 핵심 단어는 중앙에 배치
                    emotions.append({
                        'word': core_word,
                        'size': 50,
                        'x': 0.5,
                        'y': 0.5
                    })
                    existing_positions.append((0.5, 0.5))

                    # 관련 단어들은 랜덤 배치
                    radius = 0.50  # 원의 반지름을 키움
                    for i, word in enumerate(related_words):
                        size = 30 - (i * 1.5)  # 글자 크기 차이를 줄임
                        x, y = get_random_position(radius)
                        emotions.append({
                            'word': word,
                            'size': size,
                            'x': x,
                            'y': y
                        })

                    fig = go.Figure()

                    # # 원형 테두리 그리기
                    # circle_points = [
                    #     (0.5 + radius * math.cos(theta), 0.5 + radius * math.sin(theta))
                    #     for theta in np.linspace(0, 2*math.pi, 100)
                    # ]
                    
                    # fig.add_trace(go.Scatter(
                    #     x=[p[0] for p in circle_points],
                    #     y=[p[1] for p in circle_points],
                    #     mode='lines',
                    #     line=dict(color=color, width=1),
                    #     showlegend=False
                    # ))

                    # 감정 단어들을 텍스트로 표시
                    for emotion in emotions:
                        fig.add_trace(go.Scatter(
                            x=[emotion['x']],
                            y=[emotion['y']],
                            mode='text',
                            text=[emotion['word']],
                            textfont=dict(
                                size=emotion['size'],
                                color=color
                            ),
                            showlegend=False
                        ))

                    # 레이아웃 설정
                    fig.update_layout(
                        title=dict(
                            text=f"연관 감정 단어",
                            font=dict(size=20)
                        ),
                        xaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False,
                            range=[0, 1]
                        ),
                        yaxis=dict(
                            showgrid=False,
                            zeroline=False,
                            showticklabels=False,
                            range=[0, 1]
                        ),
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )

                    # # 확률 표시 추가
                    # fig.add_annotation(
                    #     text=f"{pos_prob:.1f}% {'긍정' if pos_prob > 50 else '부정'}",
                    #     xref="paper", yref="paper",
                    #     x=0.5, y=1.05,
                    #     showarrow=False,
                    #     font=dict(size=16, color=color),
                    #     borderpad=4
                    # )

                    st.plotly_chart(fig, use_container_width=True)
                
                # 구분선 추가
                st.markdown("---")
                
                # 하단에 전체 너비로 표시 (col3 내용)
                st.subheader("For you..⭐")
                # 감성 메시지 표시
                sentiment_score = result['probs']['긍정'] * 100
                message_2 = get_sentiment_message(sentiment_score)

                with st.container():
                    # 먼저 메시지 표시
                    st.markdown(f"""
                        <div style="
                            text-align: center;
                            padding: 15px;
                            background-color: #f0f2f6;
                            border-radius: 10px;
                            margin-bottom: 20px;
                            font-family: 'Helvetica Neue', Arial, sans-serif;
                            font-size: 1.1em;
                            color: #333;
                        ">
                            {message_2}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # 그 다음 YouTube 비디오 표시
                    col1, col2, col3 = st.columns([1,2,1])
                    with col2:
                        youtube_url = recommend_youtube_video(result['probs'])
                        st.video(youtube_url)
                

                
                # 히스토리에 추가
                add_to_history(
                    text=text,
                    result=result,
                    model_info=selected_model_info
                )
                update_statistics(result['label'])
                
        except Exception as e:
            st.error(f"모델 로딩 중 오류가 발생했습니다: {str(e)}")
            st.info("모델 관리 탭에서 모델 상태를 확인해주세요.")
            
    with tab_history:
        st.subheader("예측 히스토리")
        
        if not st.session_state.history:
            st.info("아직 예측 기록이 없습니다.")
        else:
            # 데이터프레임 생성
            df = pd.DataFrame(st.session_state.history)
            
            # 컬럼 이름 매핑
            column_config = {
                "timestamp": "시간",
                "text": "입력 텍스트",
                "sentiment": "예측 결과",
                "confidence": "확신도",
                "model_name": "모델",
                "model_stage": "모델 상태",
                "model_version": "모델 버전"
            }
            
            # 확신도를 퍼센트로 표시
            df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
            
            # 긍정/부정 확률 컬럼 추가
            df['확률 분포'] = df.apply(
                lambda row: f"긍정: {row['positive_prob']:.2f}, 부정: {row['negative_prob']:.2f}",
                axis=1
            )
            
            # 표시할 컬럼 선택
            display_columns = [
                'timestamp', 'text', 'sentiment', 'confidence',
                '확률 분포', 'model_name', 'model_stage', 'model_version'
            ]
            
            # 스타일링 함수
            def style_sentiment(val):
                if val == '긍정':
                    return 'background-color: #99ff99'
                return 'background-color: #ff9999'
            
            def style_confidence(val):
                conf = float(val.strip('%')) / 100
                if conf >= 0.9:
                    return 'color: #006400'  # 진한 녹색
                elif conf >= 0.7:
                    return 'color: #008000'  # 녹색
                else:
                    return 'color: #696969'  # 회색
            
            # 데이터프레임 스타일링 적용
            styled_df = df[display_columns].style\
                .applymap(style_sentiment, subset=['sentiment'])\
                .applymap(style_confidence, subset=['confidence'])
            
            # 데이터프레임 표시
            st.dataframe(
                styled_df,
                column_config=column_config,
                hide_index=True,
                use_container_width=True
            )
            
            # 통계 표시
            col1, col2, col3 = st.columns(3)
            
            total = len(df)
            positive = len(df[df['sentiment'] == '긍정'])
            negative = len(df[df['sentiment'] == '부정'])
            
            col1.metric("전체 예측", total)
            col2.metric("긍정", f"{(positive/total)*100:.1f}%")
            col3.metric("부정", f"{(negative/total)*100:.1f}%")
            
            # 시각화
            st.subheader("시계열 분석")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 시간별 감성 분포
            fig = go.Figure()
            
            for sentiment in ['긍정', '부정']:
                mask = df['sentiment'] == sentiment
                fig.add_trace(go.Scatter(
                    x=df[mask]['timestamp'],
                    y=df[mask]['confidence'].apply(lambda x: float(x.strip('%'))),
                    name=sentiment,
                    mode='markers+lines',
                    marker=dict(
                        size=8,
                        color='#99ff99' if sentiment == '긍정' else '#ff9999'
                    )
                ))
            
            fig.update_layout(
                title="시간별 예측 확신도 추이",
                xaxis_title="시간",
                yaxis_title="확신도 (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)

        # BlenderBot Chat 부분
        with tab4:
            st.header("🤖 BlenderBot Chat")
            # 라이선스 정보 추가
            st.markdown("""
                <div style="font-size:0.8em; color:gray; margin-bottom:20px;">
                * This application uses Meta's BlenderBot model
                * Model: facebook/blenderbot-400M-distill
                * License: MIT License
                * Source: Hugging Face Hub
                </div>
            """, unsafe_allow_html=True)
            
            # 주의사항 추가
            st.info("""
                ⚠️ 주의사항:
                - 이 챗봇은 Meta의 BlenderBot을 기반으로 합니다
                - 부적절하거나 부정확한 응답이 생성될 수 있습니다
                - 생성된 응답은 참고용으로만 사용해주세요
            """)
            
            
            try:
                # 감성 분석 모델 초기화
                predictor = SentimentPredictor(
                    model_name=config.project['model_name'],
                    alias=selected_model_info['stage']
                )
                
                # 세션 상태 초기화
                if 'generated' not in st.session_state:
                    st.session_state['generated'] = []
                if 'past' not in st.session_state:
                    st.session_state['past'] = []
                if 'chat_history' not in st.session_state:
                    st.session_state.chat_history = []
                # 세션 상태에 투표 기록 추가
                if 'vote_history' not in st.session_state:
                    st.session_state.vote_history = {}

                # 채팅창 상단에 설명 추가
                st.markdown("""
                    <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin-bottom: 20px;">
                        💡 <b>안내</b>: 대화에 따른 감성분석이 정확했다면 👍 UP, 아니라면 👎 DOWN 버튼을 눌러주세요.
                    </div>
                """, unsafe_allow_html=True)

                with st.form('chat_form', clear_on_submit=True):
                    user_input = st.text_input('You: ', '', key='chat_input')
                    submitted = st.form_submit_button('Send')


                # 챗봇 응답 생성 부분 수정
                if submitted and user_input:
                    try:
                        # 감성 분석
                        result = predictor.predict(user_input, return_probs=True)
                        sentiment = result['label']
                        confidence = result['confidence']
                        
                        # 챗봇 응답 생성
                        max_retries = 3
                        bot_response = "죄송합니다. 지금은 응답을 생성할 수 없습니다."
                        
                        for attempt in range(max_retries):
                            output = query({
                                "inputs": user_input,
                                "wait_for_model": True
                            })
                            
                            if isinstance(output, list) and len(output) > 0:
                                # 리스트 형식의 응답 처리
                                bot_response = output[0].get('generated_text', '')
                                if bot_response:
                                    break
                            elif isinstance(output, dict):
                                # 딕셔너리 형식의 응답 처리
                                if "error" not in output:
                                    bot_response = output.get('generated_text', '')
                                    if bot_response:
                                        break
                            else:
                                # 기타 형식의 응답을 문자열로 변환
                                bot_response = str(output).strip('[]').strip('{}')
                                if 'generated_text' in bot_response:
                                    bot_response = eval(bot_response).get('generated_text', '')
                                break
                                
                            if attempt < max_retries - 1:
                                st.info(f"응답 생성 중... (시도 {attempt + 1}/{max_retries})")
                                time.sleep(2)

                        # # 이모지 추가
                        # emoji = get_sentiment_emoji(sentiment, confidence)
                        # bot_response = f"{bot_response} {emoji}"
                        
                        # 채팅 기록 저장
                        st.session_state.past.append(user_input)
                        st.session_state.generated.append(bot_response)
                        st.session_state.chat_history.append({
                            "user": user_input,
                            "bot": bot_response,
                            "sentiment": sentiment,
                            "confidence": confidence,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

                        # 통계 업데이트
                        update_statistics(sentiment)

                    except Exception as e:
                        st.error(f"채팅 처리 중 오류가 발생했습니다: {str(e)}")

                
                # 채팅 기록 표시 부분 수
                if st.session_state.chat_history:
                    for i, chat in enumerate(st.session_state.chat_history):
                        with st.container():
                            # 사용자 메시지
                            message(chat["user"], is_user=True, key=f"chat_user_{i}")
                            
                            # 봇 응답과 감성 분석 결과
                            col1, col2, col3 = st.columns([6,3,1])
                            with col1:
                                message(chat["bot"], key=f"chat_bot_{i}")
                            with col2:
                                # 감성에 따른 색상과 이모지 설정
                                if chat['sentiment'] == '긍정':
                                    color = "#77DD77"
                                    emoji = "😊"
                                elif chat['sentiment'] == '부정':
                                    color = "#FFB6C1"
                                    emoji = "😔"
                                else:  # 중립
                                    color = "#AEC6CF"
                                    emoji = "😐"
                                
                                # HTML을 사용하여 색상이 적용된 텍스트 표시
                                st.markdown(
                                    f"""
                                    <div style="color: {color}; font-weight: bold;">
                                        감성: {chat['sentiment']} {emoji} ({chat['confidence']:.1%})
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                )
                            
                            # 투표 버튼 추가
                            with col3:
                                vote_key = f"vote_{i}"
                                if vote_key not in st.session_state.vote_history:
                                    st.session_state.vote_history[vote_key] = None
                                
                                # 현재 투표 상태 확인
                                current_vote = st.session_state.vote_history[vote_key]
                                
                                # 투표 버튼 컨테이너
                                with st.container():
                                    # UP 버튼
                                    if st.button("👍", key=f"up_{i}"):
                                        st.session_state.vote_history[vote_key] = "up"
                                        st.rerun()
                                    
                                    # DOWN 버튼
                                    if st.button("👎", key=f"down_{i}"):
                                        st.session_state.vote_history[vote_key] = "down"
                                        st.rerun()
                                    
                                    # # 현재 투표 상태 표시
                                    # if current_vote is not None:
                                    #     vote = st.session_state.vote_history[vote_key]
                                    #     st.markdown(
                                    #         f"""
                                    #         <div style='font-size: small; background-color: #FFFFE0; padding: 5px; border-radius: 5px;'>
                                    #             {'👍' if vote == 'up' else '👎'}
                                    #         </div>
                                    #         """,
                                    #         unsafe_allow_html=True
                                    #     )

                # 사이드바 컨트롤
                with st.sidebar:
                    st.subheader("채팅 컨트롤")
                    
                    # 초기화 버튼
                    if st.button("대화 초기화", key="clear_chat"):
                        st.session_state.generated = []
                        st.session_state.past = []
                        st.session_state.chat_history = []
                        st.experimental_rerun()
                    
                    # 대화 내용 다운로드
                    # 대화 내용 다운로드
                    if st.session_state.chat_history:
                        # CSV 파일을 위한 StringIO 객체 생성
                        output = StringIO()
                        writer = csv.writer(output)
                        
                        # CSV 헤더 작성
                        writer.writerow(['Timestamp', 'User Input', 'Bot Response', 'Confidence', 'Vote'])
                        
                        # 대화 내용 작성
                        for i, chat in enumerate(st.session_state.chat_history):
                            # 투표 결과를 1(UP) 또는 0(DOWN)으로 변환
                            vote = st.session_state.vote_history.get(f'vote_{i}')
                            vote_value = '1' if vote == 'up' else '0' if vote == 'down' else ''
                            
                            writer.writerow([
                                chat['timestamp'],
                                chat['user'],
                                chat['bot'],
                                # chat['sentiment'],
                                f"{chat['confidence']:.1%}",
                                vote_value
                            ])
                        
                        # 다운로드 버튼
                        st.download_button(
                            label="대화 내용 다운로드 (CSV)",
                            data=output.getvalue(),
                            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"모델 로딩 중 오류가 발생했습니다: {str(e)}")
                st.info("모델 관리 탭에서 모델 상태를 확인해주세요.")

            with tab_manage:
                # ... (모델 관리 탭 코드)
                display_model_management(model_manager, config.project['model_name'])

if __name__ == "__main__":
    initialize_session_state()
    main() 


#     

# 주요 변경사항:
# 2개 컬럼에서 3개 컬럼으로 변경
# 첫 번째 컬럼에 이모지와 게이지 차트 추가
# 두 번째 컬럼은 기존 막대 차트 유지
# 세 번째 컬럼에 새로운 감성 차트 추가
# 감성 강도에 대한 설명 텍스트 추가
