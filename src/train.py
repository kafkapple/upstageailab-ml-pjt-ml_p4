import os
import warnings
from pathlib import Path
import json
from datetime import datetime
import mlflow
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
import transformers
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from typing import Optional, Dict, Any
import sys
import time
import argparse 
import pandas as pd
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.config import Config
from src.data.nsmc_dataset import NSMCDataModule, log_data_info
from src.utils.mlflow_utils import MLflowModelManager, cleanup_artifacts, initialize_mlflow, setup_mlflow_server
from src.utils.evaluator import ModelEvaluator
from src.inference import SentimentPredictor
# torchvision 관련 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
# Tensor Core 최적화를 위한 precision 설정
torch.set_float32_matmul_precision('medium')  # 또는 'high'

class ModelTrainer:
    """모델 학습 관리자"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = Config(config_path)
        self.setup_mlflow()
        
    def setup_mlflow(self):
        setup_mlflow_server(self.config)
        self.experiment_id = initialize_mlflow(self.config)
        
    def train_and_save(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        interactive: bool = False
    ) -> Dict[str, Any]:
        """모델 학습 및 저장"""
        # 시드 설정으로 재현성 보장
        seed_everything(self.config.project['random_state'], workers=True)
        
        # 설정 업데이트
        model_name = model_name or self.config.project['model_name']
        dataset_name = dataset_name or self.config.project['dataset_name']
        
        # 모델과 데이터셋 설정 업데이트
        self.config.project['model_name'] = model_name
        self.config.project['dataset_name'] = dataset_name
        
        # 데이터셋별 sampling_rate 설정
        dataset_config = self.config._config['dataset'][dataset_name]
        if sampling_rate is not None:
            dataset_config['sampling_rate'] = sampling_rate
        
        # 학습 설정 로깅
        print("\n=== Training Configuration ===")
        print(f"Random Seed: {self.config.project['random_state']}")
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Sampling Rate: {dataset_config['sampling_rate']}")
        print(f"Train Path: {dataset_config['train_data_path']}")
        print(f"Val Path: {dataset_config['val_data_path']}")
        print(f"Column Mapping: {dataset_config['column_mapping']}")
        print(f"Batch Size: {self.config.models[model_name]['training']['batch_size']}")
        print(f"Learning Rate: {self.config.models[model_name]['training']['lr']}")
        print(f"Max Length: {self.config.models[model_name]['training']['max_length']}")
        print("=" * 50)
        
        # 모델 학습
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{self.config.project['model_name']}_{self.config.project['dataset_name']}_{timestamp}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # 데이터 모듈 준비
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.models[self.config.project['model_name']]['pretrained_model']
            )
            data_module = NSMCDataModule(config=self.config, tokenizer=tokenizer)
            data_module.prepare_data()
            data_module.setup(stage='fit')
            log_data_info(data_module)
            
            # 모델 초기화 및 학습
            model = self._initialize_model()
            trainer = self._create_trainer()
            trainer.fit(model, data_module)
            
            # 평가
            metrics = self._evaluate_model(model, tokenizer, data_module)
            
            # 모델 저장 및 로깅
            self._save_model(run, model, metrics, data_module, tokenizer)
            
            # 모델 레지스트리에 등록 (조건부)
            if metrics['val_f1'] > self.config.mlflow.model_registry_metric_threshold:
                model_manager = MLflowModelManager(self.config)
                
                try:
                    # 모델 설정 가져오기
                    model_params = {
                        'model_name': self.config.project['model_name'],
                        'dataset_name': self.config.project['dataset_name'],
                        'sampling_rate': dataset_config['sampling_rate']
                    }
                    
                    # 모델 레지스트리에 등록 (staging으로)
                    model_version = model_manager.promote_to_staging(
                        model_name=self.config.project['model_name'],
                        run_id=run.info.run_id
                    )
                    
                    print(f"Debug: Model registered with version: {model_version.version}")
                    
                    # 모델 정보 저장
                    model_manager.save_model_info(
                        run_id=run.info.run_id,
                        metrics=metrics,
                        params=model_params,
                        version=model_version.version
                    )
                    
                except Exception as e:
                    print(f"Debug: Error during model registration: {str(e)}")
                    raise
            
            result = {
                'run_id': run.info.run_id,
                'metrics': metrics,
                'run_name': run_name,
                'model': model,
                'tokenizer': tokenizer,
                'data_module': data_module
            }
            
            # 나중에 cleanup에서 사용하기 위해 저장
            self.last_metrics = metrics
            self.last_run_id = run.info.run_id
            
            # 대화형 인터페이스 실행
            if interactive:
                self._run_interactive_features(result)
            
            return result
    
    def _initialize_model(self):
        """모델 초기화"""
        model_config = self.config.models[self.config.project['model_name']]
        if 'KcBERT' in model_config['name']:
            from src.models.kcbert_model import KcBERT
            return KcBERT(**model_config['training'])
        elif 'KcELECTRA' in model_config['name']:
            from src.models.kcelectra_model import KcELECTRA
            return KcELECTRA(**model_config['training'])
        else:
            raise ValueError(f"Unknown model: {model_config['name']}")
    
    def _create_trainer(self):
        """Trainer 생성"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                mode='min',
                verbose=True
            ),
            ModelCheckpoint(
                dirpath=self.config.checkpoint['dirpath'],
                filename=self.config.checkpoint['filename'],
                monitor=self.config.checkpoint['monitor'],
                mode=self.config.checkpoint['mode'],
                save_top_k=self.config.checkpoint['save_top_k'],
                save_last=self.config.checkpoint['save_last']
            ),
            LearningRateMonitor(logging_interval='step')
        ]
        
        # 현재 선택된 모델의 training 설정 사용
        model_training_config = self.config.models[self.config.project['model_name']]['training']
        
        trainer_kwargs = {
            'max_epochs': model_training_config['epochs'],
            'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
            'devices': 1,
            'precision': model_training_config.get('precision', 16),
            'deterministic': True,
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': model_training_config.get('accumulate_grad_batches', 1),
            'callbacks': callbacks,
            'logger': TensorBoardLogger(**self.config.common['trainer']['logger'])
        }
        
        return Trainer(**trainer_kwargs)
    
    def _run_interactive_features(self, result):
        """대화형 기능 실행"""
        print("\n=== Interactive Mode ===")
        
        # 샘플 예측 출력
        self._show_sample_predictions(result)
        
        # 대화형 추론 - 현재 학습된 모델 사용
        try:
            predictor = SentimentPredictor(
                model_name=self.config.project['model_name'],
                alias="champion"
            )
        except Exception as e:
            print("No champion model found. Using current model for inference.")
            # 현재 학습된 모델을 사용하는 임시 predictor 생성
            predictor = type('Predictor', (), {
                'model': result['model'],
                'tokenizer': result['tokenizer'],
                'device': result['model'].device,
                'max_length': self.config.models[self.config.project['model_name']]['training']['max_length'],
                'predict': lambda self, text: predictor_predict(
                    self.model, 
                    self.tokenizer, 
                    text, 
                    self.device, 
                    self.max_length
                )
            })()
        
        print("\nEnter text to predict sentiment (or 'q' to quit):")
        while True:
            text = input("\nText: ").strip()
            if text.lower() == 'q':
                break
            if not text:
                continue
                
            prediction = predictor.predict(text)
            print(f"Prediction: {prediction['label']}")
            print(f"Confidence: {prediction['confidence']:.4f}")
            
        # 모델 관리
        if input("\nWould you like to manage models? (y/n): ").lower() == 'y':
            model_manager = MLflowModelManager(self.config)
            model_manager.manage_model(self.config.project['model_name'])
    
    def _predict_with_current_model(self, text):
        """현재 모델로 예측"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        model_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        with torch.no_grad():
            outputs = self.model(**model_inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(outputs.logits, dim=-1).item()
            confidence = probs[0][pred_label].item()
        
        return {
            'text': text,
            'label': '긍정' if pred_label == 1 else '부정',
            'confidence': confidence
        }
    
    def _save_model(self, run, model, metrics, data_module, tokenizer):
        """모델과 아티팩트 저장 (MLflow 로깅)"""
        try:
            artifact_path = "model"
            dataset_path = "dataset"
            metadata_path = "metadata"
            
            # MLflow에 모델 저장
            model_save_path = Path(self.config.mlflow.mlrun_path) / str(run.info.experiment_id) / run.info.run_id / "artifacts" / artifact_path
            
            # 모델 저장 디렉토리 생성
            model_data_path = model_save_path / "data"
            model_data_path.mkdir(parents=True, exist_ok=True)
            
            # 모이터셋 설정 가져오기
            dataset_config = self.config._config['dataset'][self.config.project['dataset_name']]
            
            # 모델 설정 저장
            model_config = {
                **self.config.models[self.config.project['model_name']],  # 전체 모델 설정
                'model_name': self.config.project['model_name'],
                'dataset_name': self.config.project['dataset_name'],
                'sampling_rate': dataset_config['sampling_rate'],
                'pretrained_model': self.config.models[self.config.project['model_name']]['pretrained_model']
            }
            
            # state_dict만 저장
            state_dict = model.state_dict()
            torch.save(state_dict, model_data_path / "model.pth")
            print(f"Model state dict saved to: {model_data_path / 'model.pth'}")
            
            # MLflow에 모델 메타데이터 로깅
            print("\nDebug: Registering model with MLflow")
            print(f"Model name: {self.config.project['model_name']}")
            print(f"Run ID: {run.info.run_id}")
            
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path=artifact_path,
                registered_model_name=self.config.project['model_name'],
                conda_env=None,
                code_paths=["src"],
                pip_requirements=["torch", "transformers"],
                metadata=model_config
            )
            
            print("Debug: Model registration completed")
            
            # 데이터셋 메타데이터 준비
            train_df = pd.read_csv(dataset_config['train_data_path'], sep='\t')
            val_df = pd.read_csv(dataset_config['val_data_path'], sep='\t')
            
            dataset_info = {
                "name": self.config.project['dataset_name'],
                "sampling_rate": dataset_config['sampling_rate'],
                "train_data": {
                    "path": str(dataset_config['train_data_path']),
                    "num_rows": len(train_df),
                    "num_cols": len(train_df.columns),
                    "columns": train_df.columns.tolist(),
                    "column_mapping": dataset_config['column_mapping']
                },
                "val_data": {
                    "path": str(dataset_config['val_data_path']),
                    "num_rows": len(val_df),
                    "num_cols": len(val_df.columns),
                    "columns": val_df.columns.tolist()
                },
                "class_distribution": {
                    "train": train_df[dataset_config['column_mapping']['label']].value_counts().to_dict(),
                    "val": val_df[dataset_config['column_mapping']['label']].value_counts().to_dict()
                }
            }
            
            # MLflow에 데이터셋 정보 로깅
            mlflow.log_dict(dataset_info, "dataset_used.json")
            
            # 데이터셋 통계 로깅
            mlflow.log_param("train_samples", len(train_df))
            mlflow.log_param("val_samples", len(val_df))
            mlflow.log_param("dataset_name", self.config.project['dataset_name'])
            mlflow.log_param("sampling_rate", dataset_config['sampling_rate'])
            
            # 샘플 데이터 로깅 (처음 5개 행)
            sample_data = {
                "train_samples": train_df.head().to_dict(orient='records'),
                "val_samples": val_df.head().to_dict(orient='records')
            }
            mlflow.log_dict(sample_data, "sample_data.json")
            
            # Confusion Matrix 생성 및 저장
            evaluator = ModelEvaluator(model, tokenizer)
            cm_fig = evaluator.plot_confusion_matrix(data_module.val_dataset)
            
            # 메타데이터 준비
            metadata = {
                "model_info": {
                    "name": self.config.project['model_name'],
                    "version": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "metrics": metrics
                },
                "dataset_info": dataset_info,
                "training_config": model_config
            }
            
            # MLflow 아티팩트 로깅
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                
                # 메타데이터 저장
                with open(temp_dir / "metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                # Confusion Matrix 저장
                cm_fig.savefig(temp_dir / "confusion_matrix.png")
                plt.close(cm_fig)
                
                # 아티팩트 로깅
                mlflow.log_artifacts(str(temp_dir), metadata_path)
            
        except Exception as e:
            print(f"Error during model saving: {str(e)}")
            raise
    
    def _evaluate_model(self, model, tokenizer, data_module):
        """모델 평가"""
        evaluator = ModelEvaluator(model, tokenizer)
        base_metrics = evaluator.evaluate_dataset(data_module)
        
        # val_ 접두사 추가
        metrics = {
            'val_accuracy': base_metrics['accuracy'],
            'val_f1': base_metrics['f1'],
            'val_precision': base_metrics['precision'],
            'val_recall': base_metrics['recall']
        }
        
        print("\n=== Validation Results ===")
        print(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"Validation F1 Score: {metrics['val_f1']:.4f}")
        print(f"Validation Precision: {metrics['val_precision']:.4f}")
        print(f"Validation Recall: {metrics['val_recall']:.4f}")
        
        # MLflow에 메트릭 로깅
        mlflow.log_metrics(metrics)
        
        return metrics
    
    def cleanup(self):
        """학습 완료 후 임시 파일 정리"""
        try:
            if hasattr(self, 'last_metrics') and hasattr(self, 'last_run_id'):
                cleanup_artifacts(self.config, self.last_metrics, self.last_run_id)
        except Exception as e:
            print(f"Warning: Cleanup failed - {str(e)}")
    
    def _show_sample_predictions(self, result):
        """검증 데이터셋에서 샘플 예측 출력"""
        print("\n=== Sample Predictions ===")
        
        model = result['model']
        tokenizer = result['tokenizer']
        val_dataset = result['data_module'].val_dataset
        
        indices = torch.randperm(len(val_dataset))[:5].tolist()
        model.eval()
        
        with torch.no_grad():
            for idx in indices:
                text = val_dataset.texts[idx]
                true_label = val_dataset.labels[idx]
                
                # 토크나이즈
                inputs = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)
                
                # 필요한 입력만 선택
                model_inputs = {
                    'input_ids': inputs['input_ids'],
                    'attention_mask': inputs['attention_mask']
                }
                
                # 예측
                outputs = model(**model_inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_label = torch.argmax(outputs.logits, dim=-1).item()
                confidence = probs[0][pred_label].item()
                
                print("\nText:", text)
                print(f"True Label: {'긍정' if true_label == 1 else '부정'}")
                print(f"Prediction: {'긍정' if pred_label == 1 else '부정'}")
                print(f"Confidence: {confidence:.4f}")
                print("-" * 80)

    @classmethod
    def train_model(
        cls,
        config_path: str = "config/config.yaml",
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        interactive: bool = False,
        reset_mlflow: bool = False
    ) -> Dict[str, Any]:
        """모델 학습 함수"""
        trainer = cls(config_path)
        
        # MLflow 초기화 (선택적)
        if reset_mlflow:
            setup_mlflow_server(trainer.config, reset_experiments=True)
            print("MLflow environment has been reset")
        
        try:
            return trainer.train_and_save(
                model_name=model_name,
                dataset_name=dataset_name,
                sampling_rate=sampling_rate,
                interactive=interactive
            )
        finally:
            trainer.cleanup()
# Add new helper function
def predictor_predict(model, tokenizer, text, device, max_length):
    """현재 모델로 예측하는 헬퍼 함수"""
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    
    model_inputs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask']
    }
    
    with torch.no_grad():
        outputs = model(**model_inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(outputs.logits, dim=-1).item()
        confidence = probs[0][pred_label].item()
    
    return {
        'text': text,
        'label': '긍정' if pred_label == 1 else '부정',
        'confidence': confidence
    }

# CLI 실행을 위한 코드
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--sampling-rate", type=float, help="Data sampling rate")
    parser.add_argument("--interactive", default=True, action="store_true", help="Interactive mode")
    parser.add_argument("--reset-mlflow", default=False, action="store_true", help="Reset MLflow environment")

    args = parser.parse_args()
    
    result = ModelTrainer.train_model(
        config_path=args.config,
        model_name=args.model,
        dataset_name=args.dataset,
        sampling_rate=args.sampling_rate,
        interactive=args.interactive,
        reset_mlflow=args.reset_mlflow
    )
    
    print("\n=== Training completed ===")
    print(f"Run ID: {result['run_id']}")
    for metric_name, value in result['metrics'].items():
        if isinstance(value, float):
            print(f"{metric_name}: {value:.4f}")

