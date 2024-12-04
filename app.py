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

def add_to_history(text: str, result: dict, model_id: int):
    """Add prediction to history"""
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "sentiment": result['label'],
        "confidence": result['confidence'],
        "negative_prob": result['probabilities'][0],
        "positive_prob": result['probabilities'][1],
        "model_id": model_id
    })

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
                if new_stage == 'champion':
                    model_manager.promote_to_production(
                        model_name,
                        selected_model['version']
                    )
                elif new_stage == 'archived':
                    model_manager.archive_model(
                        model_name,
                        selected_model['version']
                    )
                elif new_stage == 'candidate':
                    model_manager.promote_to_staging(
                        model_name,
                        selected_model['run_id']
                    )
                
                st.success(f"모델 상태가 {stage_map[new_stage]}(으)로 변경되었습니다.")
                time.sleep(2)
                st.rerun()
                
            except Exception as e:
                st.error(f"상태 변경 중 오류가 발생했습니다: {str(e)}")

def main():
    st.set_page_config(
        page_title="Sentiment Analysis Demo",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Initialize config and model manager
    config = Config()
    model_manager = MLflowModelManager(config)
    
    # Create tabs
    tab_predict, tab_manage = st.tabs(["감성 분석", "모델 관리"])
    
    with tab_predict:
        st.title("한국어 감성 분석 데모")
        st.write("텍스트를 입력하면 긍정/부정을 판단합니다.")
        
        # Get production models
        production_models = model_manager.get_production_models()
        
        if not production_models:
            st.error("No production models found. Please train and promote a model first.")
            st.stop()
        
        # Model selection
        model_options = {
            f"{model['run_name']} ({model['timestamp']})": model 
            for model in production_models
        }
        
        selected_model_name = st.sidebar.selectbox(
            "Select Production Model",
            options=list(model_options.keys())
        )
        
        selected_model_info = model_options[selected_model_name]
        display_model_info(selected_model_info)
        
        # Get model_id from selected model
        model_id = production_models.index(selected_model_info) + 1
        
        # Load predictor
        predictor = load_predictor(selected_model_info)
        if predictor is None:
            st.error("Failed to load the model predictor.")
            st.stop()
        
        # Display statistics
        display_statistics()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input
            text = st.text_area(
                "분석할 텍스트를 입력하세요:",
                height=100,
                help="여러 줄의 텍스트를 입력할 수 있습니다."
            )
            
            if st.button("분석하기", type="primary"):
                if not text:
                    st.warning("텍스트를 입력해주세요.")
                    return
                
                with st.spinner("분석 중..."):
                    result = predict_sentiment(text, predictor)
                    if result:
                        # Update statistics and history
                        update_statistics(result['label'])
                        add_to_history(text, result, model_id)
                        
                        # Display results
                        st.subheader("분석 결과")
                        col_result1, col_result2 = st.columns(2)
                        
                        with col_result1:
                            st.metric("감성", result['label'])
                            st.metric("확신도", f"{result['confidence']:.1%}")
                        
                        with col_result2:
                            fig = go.Figure(go.Bar(
                                x=['부정', '긍정'],
                                y=result['probabilities'],
                                marker_color=['#ff9999', '#99ff99']
                            ))
                            fig.update_layout(
                                title="감성 분석 확률 분포",
                                yaxis_title="확률",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("분석 상세 정보")
            with st.expander("자세히 보기", expanded=True):
                st.write("입력 텍스트 길이:", len(text) if text else 0)
                st.write("토큰 수:", len(predictor.tokenizer.encode(text)) if text else 0)
                if text:
                    st.json({
                        "prediction": {
                            "label": result['label'] if 'result' in locals() else None,
                            "confidence": f"{result['confidence']:.4f}" if 'result' in locals() else None,
                            "probabilities": {
                                "negative": f"{result['probabilities'][0]:.4f}" if 'result' in locals() else None,
                                "positive": f"{result['probabilities'][1]:.4f}" if 'result' in locals() else None
                            }
                        }
                    })
        
        # History section
        st.markdown("---")
        st.subheader("분석 히스토리")
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            df = df.sort_values('timestamp', ascending=False)
            
            # Add styling
            def color_sentiment(val):
                color = '#99ff99' if val == '긍정' else '#ff9999'
                return f'background-color: {color}; color: black'
            
            styled_df = df.style.applymap(
                color_sentiment, 
                subset=['sentiment']
            ).format({
                'confidence': '{:.1%}',
                'negative_prob': '{:.4f}',
                'positive_prob': '{:.4f}'
            })
            
            st.dataframe(
                styled_df,
                column_config={
                    "timestamp": "시간",
                    "text": "텍스트",
                    "sentiment": "감성",
                    "confidence": "확신도",
                    "negative_prob": "부정 확률",
                    "positive_prob": "긍정 확률",
                    "model_id": "모델 ID"
                },
                hide_index=True,
                use_container_width=True
            )
            
            if st.button("히스토리 초기화"):
                st.session_state.history = []
                st.session_state.total_predictions = 0
                st.session_state.positive_count = 0
                st.session_state.negative_count = 0
                st.rerun()
        else:
            st.info("아직 분석 기록이 없습니다.")
    
    with tab_manage:
        display_model_management(model_manager, config.project['model_name'])

if __name__ == "__main__":
    main() 