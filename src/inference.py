from pathlib import Path
import json
from typing import Dict, Union, Optional, List, Any
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.utils.mlflow_utils import MLflowModelManager
from src.config import Config
from transformers import (
    BertForSequenceClassification,
    ElectraForSequenceClassification,
    BertTokenizer,
    ElectraTokenizer,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)
import mlflow

def print_results(results):
    for result in results:
        print(f"\nInput Text: {result['text']}")
        print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2f})")
        print(f"Probabilities: 긍정={result['probs']['긍정']:.2f}, 부정={result['probs']['부정']:.2f}")


class SentimentPredictor:
    """감정 분석 예측기
    
    # Attributes
        model: 로드된 감정 분석 모델
        tokenizer: 텍스트 토크나이저
        model_config: 모델 설정 정보
        device: 실행 디바이스 (CPU/GPU)
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        alias: str = "champion",
        config_path: str = "config/config.yaml",
        device: Optional[str] = None
    ):
        """감정 분석 예측기 초기화"""
        try:
            self.config = Config(config_path)
            self.model_manager = MLflowModelManager(self.config)
            self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # 1. model_registry.json에서 모델 정보 가져오기
            model_infos = self.model_manager.load_model_info()
            if not model_infos:
                raise ValueError("No models found in registry")
            
            # 기본값으로 model_info 초기화
            model_info = None
            
            # alias에 따른 모델 선택
            if alias == "champion":
                model_info = next((info for info in model_infos if info['stage'] == 'champion'), None)
                if not model_info:
                    print("No champion model found, trying candidate...")
                    model_info = next((info for info in model_infos if info['stage'] == 'candidate'), None)
            elif alias == "candidate":
                model_info = next((info for info in model_infos if info['stage'] == 'candidate'), None)
            
            if not model_info:
                print("No champion or candidate model found, using latest model...")
                model_info = model_infos[-1]  # 최신 모델 사용
                
            print(f"Selected model info: {model_info['run_name']} (version: {model_info['version']})")
            
            # 2. 모델 파일 경로 구성
            run_id = model_info['run_id']
            model_path = (
                Path(self.config.mlflow.artifact_location) 
                / run_id 
                / "artifacts" 
                / "model" 
                / "data" 
                / "model.pth"
            )
            
            print(f"\nDebug: Model path details:")
            print(f"Artifact location: {self.config.mlflow.artifact_location}")
            print(f"Run ID: {run_id}")
            print(f"Full path: {model_path}")
            print(f"Path exists: {model_path.exists()}")
            
            # 대체 경로 확인
            alt_path = Path("mlartifacts") / run_id / "artifacts" / "model" / "data" / "model.pth"
            print(f"\nDebug: Alternative path details:")
            print(f"Alt path: {alt_path}")
            print(f"Alt path exists: {alt_path.exists()}")
            
            if not model_path.exists():
                raise ValueError(f"Model file not found: {model_path}")
            
            print(f"Loading model from: {model_path}")
            
            # 3. 모델 설정 가져오기
            self.model_config = model_info['params']
            self.max_length = int(self.model_config.get('max_length', 512))
            
            # 4. 모델 클래스 결정 및 초기화
            model_type = self.model_config['model_name']
            if 'KcBERT' in model_type:
                from src.models.kcbert_model import KcBERT as ModelClass
            elif 'KcELECTRA' in model_type:
                from src.models.kcelectra_model import KcELECTRA as ModelClass
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            print(f"Initializing model class: {ModelClass.__name__}")
            self.model = ModelClass(**self.model_config)
            
            # 5. state_dict 로드
            print(f"Loading state dict from: {model_path}")
            try:
                # 먼저 state_dict로 로드 시도
                state_dict = torch.load(model_path, map_location=self.device)
                if isinstance(state_dict, dict):
                    self.model.load_state_dict(state_dict)
                else:
                    # state_dict가 아닌 경우 전체 모델로 로드 시도
                    loaded_model = state_dict
                    self.model.load_state_dict(loaded_model.state_dict())
            except Exception as e:
                print(f"Error loading state dict: {str(e)}")
                raise
            
            self.model.to(self.device)
            self.model.eval()
            
            # 6. 토크나이저 로드
            print(f"Loading tokenizer: {self.model_config['pretrained_model']}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['pretrained_model'])
            
            print("Model initialization completed successfully")
            
        except Exception as e:
            print(f"Error during model initialization: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    def _load_model_info(
        self,
        model_name: Optional[str],
        version: str,
        registry_path: str
    ) -> Dict[str, Any]:
        """모델 정보 로드
        
        # Returns
            Dict: 모델 메타데이터 및 설정 정보
        """
        with open(registry_path, 'r', encoding='utf-8') as f:
            registry = json.load(f)
            
        # 요청된 버전의 모델 찾기
        available_models = [m for m in registry if m['stage'] == version]
        if not available_models:
            raise ValueError(f"No models found with version: {version}")
            
        if model_name:
            model_info = next((m for m in available_models if m['run_name'] == model_name), None)
            if not model_info:
                raise ValueError(f"Model not found: {model_name}")
        else:
            # 가장 최신 모델 선택
            model_info = sorted(
                available_models,
                key=lambda x: x['timestamp'],
                reverse=True
            )[0]
            
        return model_info
        
    def _get_model_class(self, model_name: str):
        """모델 이름에 따른 모델 클래스 반환"""
        if 'kcbert' in model_name.lower():
            return BertForSequenceClassification, BertTokenizer
        elif 'kcelectra' in model_name.lower():
            return ElectraForSequenceClassification, ElectraTokenizer
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        
    def _load_model_and_tokenizer(self) -> tuple[PreTrainedModel, Any]:
        """모델과 토크나이저 로드
        
        # Returns
            Tuple[PreTrainedModel, Any]: (로드된 모델, 토크나이저)
        """
        # 모델 아티팩트 경로 구성
        model_path = Path(f"mlruns/{self.model_info['experiment_id']}/{self.model_info['run_id']}/artifacts/model")
        
        # 설정 파일 로드 및 모델 레지스트리 정보와 병합
        config_path = model_path / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
            
        # 레지스트리의 파라미터 정보 추가
        model_config.update(self.model_info['params'])
        
        # 모델 타입에 따른 클래스 선택
        model_class, tokenizer_class = self._get_model_class(model_config['pretrained_model'])
        
        # 토크나이저 로드
        tokenizer = tokenizer_class.from_pretrained(model_config['pretrained_model'])
        
        # 모델 로드
        model = model_class.from_pretrained(
            model_config['pretrained_model'],
            num_labels=model_config['num_labels']
        )
        
        # 학습된 가중치 로드
        model.load_state_dict(
            torch.load(model_path / "model.pt", map_location=self.device),
            strict=False
        )
        
        # Freeze layers if specified (original training configuration 유지)
        if 'num_unfreeze_layers' in model_config:
            self._freeze_layers(model, model_config['num_unfreeze_layers'])
            
        model.to(self.device)
        model.eval()
        
        return model, tokenizer
        
    def _freeze_layers(self, model: PreTrainedModel, num_unfreeze_layers: int):
        """레이어 동결 설정"""
        if num_unfreeze_layers <= 0:
            return
            
        # 모든 파라미터 동결
        for param in model.parameters():
            param.requires_grad = False
        
        # 분류기 레이어는 항상 학습
        for param in model.classifier.parameters():
            param.requires_grad = True
            
        if num_unfreeze_layers > 0:
            # 모델 타입에 따른 인코더 레이어 접근
            if isinstance(model, BertForSequenceClassification):
                encoder_layers = model.bert.encoder.layer
            elif isinstance(model, ElectraForSequenceClassification):
                encoder_layers = model.electra.encoder.layer
            else:
                return
                
            # 지정된 수의 레이어만 학습
            total_layers = len(encoder_layers)
            for i in range(total_layers - num_unfreeze_layers, total_layers):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = True
        
    def predict(
        self,
        text: Union[str, List[str]],
        return_probs: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """텍스트 감정 예측
        
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
        """
        # 단일 텍스트를 리스트로 변환
        single_input = isinstance(text, str)
        texts = [text] if single_input else text
        
        # 토크나이즈
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # 필요한 입력만 ���택 (token_type_ids 제외)
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(outputs.logits, dim=-1)
            confidences = torch.max(probs, dim=-1).values
        
        # 결과 변환
        label_map = {0: '부정', 1: '긍정'}
        results = []
        
        for text_input, pred, conf, prob in zip(texts, predictions, confidences, probs):
            result = {
                'text': text_input,
                'label': label_map[pred.item()],
                'confidence': conf.item(),
            }
            
            if return_probs:
                result['probs'] = {
                    '부정': prob[0].item(),
                    '긍정': prob[1].item()
                }
            
            results.append(result)
        
        return results[0] if single_input else results
    

def main():
    try:
        # Config 초기화
        config = Config()
        
        # 기본적으로 champion 모델 시도, 없으면 candidate나 최신 모델 사용
        predictor = SentimentPredictor(
            model_name=config.project['model_name'],
            alias="champion",
            config_path="config/config.yaml"
        )
        
        # 단일 텍스트 예측
        text = "정말 재미있는 영화였어요!"
        print(f"\n==== Prediction ====\nText: {text}\n")
        result = predictor.predict(text)
        print_results([result])
        
        # 배치 예측 & 확률값 포함
        texts = ["다시 보고 싶은 영화", "별로에요"]
        print(f"\n==== Batch Prediction ====\nTexts: {texts}\n")
        results = predictor.predict(
            texts,
            return_probs=True
        )
        print_results(results)
        
        return results
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Config 초기화
    main()
  #  model manage
    config = Config()
    
    # MLflow 모델 관리 초기화
    model_manager = MLflowModelManager(config)
    model_manager.manage_model(config.project['model_name'])
    