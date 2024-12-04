from pathlib import Path
import json
from typing import Dict, Union, Optional, List, Any
import torch
from transformers import (
    BertForSequenceClassification,
    ElectraForSequenceClassification,
    BertTokenizer,
    ElectraTokenizer,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)

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
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        device: Optional[str] = None
    ):
        """감정 분석 예측기 초기화
        
        Args:
            model: 학습된 모델
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이 (None일 경우 모델의 기본값 사용)
            device: 실행 디바이스 (None일 경우 모델의 현재 디바이스 사용)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length or getattr(
            tokenizer,
            'model_max_length',
            512  # fallback value
        )
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        
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
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
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
    
def print_results(results):
    for result in results:
        print(f"\nInput Text: {result['text']}")
        print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2f})")
        print(f"Probabilities: 긍정={result['probs']['긍정']:.2f}, 부정={result['probs']['부정']:.2f}")

if __name__ == "__main__":
    
    predictor = SentimentPredictor()


    text = "정말 재미있는 영화였어요!"
    print(f"\n==== Prediction ====\nText: {text}\n")
    result = predictor.predict([text])
    print_results([result])

    # 배치 예측 & 확률값 포함
    [texts] = ["다시 보고 싶은 영화", "별로에요"]
    print(f"\n==== Batch Prediction ====\nTexts: {texts}\n")
    
    results = predictor.predict(
        texts,
        return_probs=True
    )
    print_results(results)