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
    """Predict sentiment using predictor"""
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
        page_title="감성 분석 서비스",
        page_icon="🤖",
        layout="wide"
    )
    
    # CSS 스타일 적용
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
            font-weight: 500;
        }
        h2 {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            font-size: 20px;
            font-weight: 500;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("감성 분석 서비스")
    
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
    tab_predict, tab_history, tab_manage = st.tabs(["예측", "히스토리", "모델 관리"])
    
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
        
        # 예측 UI
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
                
                # 결과 표시
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("감성", result['label'])
                    st.metric("확신도", f"{result['confidence']:.1%}")
                
                with col2:
                    # 확률 분포 그래프
                    fig = go.Figure(go.Bar(
                        x=['부정', '긍정'],
                        y=[result['probs']['부정'], result['probs']['긍정']],
                        marker_color=['#ff9999', '#99ff99']
                    ))
                    fig.update_layout(
                        title="감성 분석 확률 분포",
                        yaxis_title="확률",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 히스토리에 추가
                add_to_history(
                    text=text,
                    result=result,  # 원본 결과 그대로 전달
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
    
    with tab_manage:
        # ... (모델 관리 탭 코드)
        display_model_management(model_manager, config.project['model_name'])

if __name__ == "__main__":
    initialize_session_state()
    main() 