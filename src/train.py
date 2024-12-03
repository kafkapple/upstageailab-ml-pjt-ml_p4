import os
import json
from pathlib import Path
from datetime import datetime
import torch
import mlflow
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModel, AutoModelForMaskedLM

from config import Config
from data.nsmc_dataset import NSMCDataModule, log_data_info
from inference import SentimentPredictor
from evaluator import ModelEvaluator
from utils.mlflow_utils import MLflowModelManager, initialize_mlflow
from utils.config_sync import ConfigSynchronizer
from server_mlflow import update_artifact_location

torch.set_float32_matmul_precision('high') 

def display_random_predictions(model, tokenizer, val_dataset, max_length):
    """랜덤 샘플에 대한 예측 결과 출력"""
    print("\n=== Random Validation Samples ===")
    
    # 디바이스 설정
    device = next(model.parameters()).device
    
    # 랜덤 샘플링
    indices = torch.randperm(len(val_dataset))[:5]
    
    for idx in indices:
        sample = val_dataset[idx]
        text = val_dataset.texts[idx]
        true_label = sample['label'].item()
        
        # 예측을 위한 입력 생성
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        # token_type_ids 제거
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        
        # 입력을 모델의 디바이스로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()
        
        # 결과 출력
        print("\n" + "=" * 100)
        print(f"Text: {text}")
        print(f"True Label: {'긍정' if true_label == 1 else '부정'}")
        print(f"Predicted: {'긍정' if pred_label == 1 else '부정'}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Correct: {'✓' if pred_label == true_label else '✗'}")
    
    print("=" * 100 + "\n")

def train_and_evaluate(config: Config):
    """모델 학습 및 평가"""
    print("=" * 50)
    print("\n=== Training Configuration ===")
    print(f"Model: {config.project['model_name']}")
    print(f"Pretrained Model: {config.model_config['pretrained_model']}")
    print(f"Batch Size: {config.training_config['batch_size']}")
    print(f"Learning Rate: {config.training_config['lr']}")
    print(f"Epochs: {config.training_config['epochs']}")
    print(f"Max Length: {config.training_config['max_length']}")
    print(f"Optimizer: {config.training_config['optimizer']}")
    print(f"LR Scheduler: {config.training_config['lr_scheduler']}")
    print(f"Precision: {config.training_config['precision']}")
    print(f"Accumulate Grad Batches: {config.training_config['accumulate_grad_batches']}")
    print("=" * 50 + "\n")
    
    # MLflow 설정
    experiment_id = initialize_mlflow(config)
    
    seed_everything(config.project['random_state'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.project['model_name']}_{config.project['dataset_name']}_{timestamp}"
    
    # 임시 작업 디렉토리 생성 (MLflow 실험 구조에 맞춤)
    temp_dir = Path("temp_artifacts")
    temp_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        with mlflow.start_run(run_name=run_name) as run:
            print("\n=== MLflow Configuration ===")
            print(f"Run Name: {run_name}")
            print(f"Experiment ID: {experiment_id}")
            print(f"Run ID: {run.info.run_id}")
            print(f"Artifact URI: {run.info.artifact_uri}")
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")
            
            # MLflow 클라이언트 설정
            client = mlflow.tracking.MlflowClient()
            
            # 실험 정보 출력
            experiment = client.get_experiment(experiment_id)
            print("\n=== Experiment Details ===")
            print(f"Experiment Name: {experiment.name}")
            print(f"Experiment ID: {experiment.experiment_id}")
            print(f"Artifact Location: {experiment.artifact_location}")
            print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
            
            # MLflow 아티팩트 경로 설정 (실제 저장될 경로)
            artifact_base = str(Path(config.project_root) / config.mlflow.artifact_location).replace("\\", "/")
            artifact_uri = f"file:///{artifact_base}"
            
            print("\n=== Artifact Paths ===")
            print(f"Base Path: {artifact_base}")
            print(f"Artifact URI: {artifact_uri}")
            print(f"Temp Directory: {temp_dir}")
            
            # 실험 아티팩트 URI 설정
            try:
                client.update_experiment(
                    experiment_id=experiment_id,
                    artifact_location=artifact_uri
                )
                print("Successfully updated experiment artifact location")
            except Exception as e:
                print(f"Error updating experiment artifact location: {str(e)}")
            
            # 경로 존재 확인
            print("\n=== Path Verification ===")
            print(f"Project Root exists: {Path(config.project_root).exists()}")
            print(f"Artifact Base exists: {Path(artifact_base).exists()}")
            print(f"Temp Directory exists: {temp_dir.exists()}")
            
            # 파일 시스템 권한 확인
            try:
                test_file = temp_dir / "test.txt"
                with open(test_file, 'w') as f:
                    f.write("test")
                print("Successfully wrote test file")
                os.remove(test_file)
                print("Successfully removed test file")
            except Exception as e:
                print(f"File system permission error: {str(e)}")
            
            print("\n=== Starting Model Training ===")
            
            # 모델 저장 경로 설정
            model_save_dir = Path(config.models[config.project['model_name']]['model_dir']) / config.project['model_name']
            model_save_dir = Path(str(model_save_dir).replace("\\", "/"))
            model_save_dir.mkdir(parents=True, exist_ok=True)
            
            # Model initialization based on model type
            if config.project['model_name'].startswith('KcBERT'):
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_config['pretrained_model'],
                    cache_dir=str(model_save_dir)
                )
                from models.kcbert_model import KcBERT
                model = KcBERT(pretrained_model=config.model_config['pretrained_model'])
                
            elif config.project['model_name'].startswith('KcELECTRA'):
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_config['pretrained_model'],
                    cache_dir=str(model_save_dir)
                )
                from models.kcelectra_model import KcELECTRA
                model = KcELECTRA(pretrained_model=config.model_config['pretrained_model'])
            else:
                raise ValueError(f"Unknown model: {config.project['model_name']}")
            
            # Trainer 설정
            trainer_config = config.common['trainer'].copy()  # 복사본 생성
            trainer_config['max_epochs'] = int(config.training_config['epochs'])
            
            # logger가 trainer_config에 있다면 제거
            if 'logger' in trainer_config:
                del trainer_config['logger']
            
            # Checkpoint 콜백 설정
            checkpoint_config = config.common['checkpoint']
            checkpoint_callback = ModelCheckpoint(
                **checkpoint_config
            )
            
            # 다른 콜백들 설정
            callbacks = [
                checkpoint_callback,
                EarlyStopping(monitor='val_loss', patience=3, mode='min'),
                LearningRateMonitor(logging_interval='step')
            ]
            
            # Logger 설정
            logger = pl.loggers.MLFlowLogger(
                experiment_name=config.mlflow.experiment_name,
                tracking_uri=config.mlflow.tracking_uri,
                run_id=run.info.run_id
            )
            
            # epochs 값 확인을 위한 로그 추가
            print(f"\nTraining epochs: {trainer_config['max_epochs']}")
            
            # Trainer 초기화
            trainer = pl.Trainer(
                **trainer_config,
                callbacks=callbacks,
                logger=logger
            )
            
            # Data module 초기화
            data_module = NSMCDataModule(
                config=config,
                tokenizer=tokenizer
            )
            
            data_module.prepare_data()
            data_module.setup(stage='fit')
            
            # Log data info
            log_data_info(data_module)
            
            # 데이터셋 정보 로깅
            train_labels = [item['label'] for item in data_module.train_dataset]
            val_labels = [item['label'] for item in data_module.val_dataset]
            
            mlflow.log_params({
                "train_size": len(data_module.train_dataset),
                "val_size": len(data_module.val_dataset),
                "train_positive": train_labels.count(1),
                "train_negative": train_labels.count(0),
                "val_positive": val_labels.count(1),
                "val_negative": val_labels.count(0)
            })
            
            # Train the model
            trainer.fit(model, data_module)
            
            # Evaluation
            evaluator = ModelEvaluator(model, tokenizer)
            eval_metrics = evaluator.evaluate_dataset(data_module)
            
            val_accuracy = eval_metrics['accuracy']
            val_f1 = eval_metrics['f1']
            
            print("\n=== Validation Results ===")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation F1 Score: {val_f1:.4f}")
            
            mlflow.log_metrics({
                "val_accuracy": val_accuracy,
                "val_f1": val_f1
            })
            
            mlflow.log_params({
                "model_name": config.project['model_name'],
                "pretrained_model": config.model_config['pretrained_model'],
                "batch_size": config.training_config['batch_size'],
                "learning_rate": config.training_config['lr'],
                "epochs": config.training_config['epochs'],
                "max_length": config.training_config['max_length'],
                "optimizer": config.training_config['optimizer'],
                "lr_scheduler": config.training_config['lr_scheduler'],
                "precision": config.training_config['precision'],
                "accumulate_grad_batches": config.training_config['accumulate_grad_batches']
            })
            
            print("\nSaving model artifacts...")
            
            # 모델 상태 저장 (model.pt)
            model_path = temp_dir / "model.pt"
            print(f"Saving model to: {model_path}")
            torch.save(model.state_dict(), model_path)
            
            try:
                print(f"Logging model artifact from: {model_path}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Model path exists: {model_path.exists()}")
                print(f"Model path absolute: {model_path.absolute()}")
                mlflow.log_artifact(str(model_path), "model")
                print("Successfully logged model artifact")
            except Exception as e:
                print(f"Error logging model artifact: {str(e)}")
                print(f"MLflow run ID: {run.info.run_id}")
                print(f"MLflow artifact URI: {run.info.artifact_uri}")
                raise
            
            # 모델 설정 저장 (config.json)
            model_config = {
                "model_type": config.project['model_name'],
                "pretrained_model": config.model_config['pretrained_model'],
                "num_labels": config.training_config['num_labels'],
                "learning_rate": config.training_config['lr'],
                "num_train_epochs": config.training_config['epochs'],
                "per_device_train_batch_size": config.training_config['batch_size'],
                "per_device_eval_batch_size": config.training_config['batch_size'],
                "max_length": config.training_config['max_length'],
                "report_cycle": config.training_config['report_cycle'],
                "optimizer": config.training_config['optimizer'],
                "lr_scheduler": config.training_config['lr_scheduler'],
                "precision": config.training_config['precision'],
                "num_unfreeze_layers": config.training_config.get('num_unfreeze_layers', -1),
                "accumulate_grad_batches": config.training_config['accumulate_grad_batches']
            }
            
            config_path = temp_dir / "config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(model_config, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(str(config_path), "model")
            
            print(f"Model artifacts saved successfully!")
            
            # Confusion Matrix 저장
            confusion_matrix_img = ModelEvaluator.plot_confusion_matrix(data_module.val_dataset, model, tokenizer)
            confusion_matrix_path = temp_dir / "confusion_matrix.png"
            confusion_matrix_img.save(str(confusion_matrix_path))
            mlflow.log_artifact(str(confusion_matrix_path), "confusion_matrix")
            
            # 데이터셋 샘플 저장
            dataset_samples = []
            for i in range(min(5, len(data_module.train_dataset))):
                sample = data_module.train_dataset[i]
                dataset_samples.append({
                    'text': data_module.train_dataset.texts[i],
                    'label': sample['label'].item() if isinstance(sample['label'], torch.Tensor) else sample['label'],
                    'input_ids': sample['input_ids'].tolist() if isinstance(sample['input_ids'], torch.Tensor) else sample['input_ids'],
                    'attention_mask': sample['attention_mask'].tolist() if isinstance(sample['attention_mask'], torch.Tensor) else sample['attention_mask']
                })
            
            dataset_samples_path = temp_dir / "dataset_samples.json"
            with open(dataset_samples_path, 'w', encoding='utf-8') as f:
                json.dump(dataset_samples, f, indent=2, ensure_ascii=False)
            mlflow.log_artifact(str(dataset_samples_path), "dataset_info")
            
            # Model registration and management
            model_manager = MLflowModelManager(config)
            if val_f1 > config.mlflow.model_registry_metric_threshold:
                model_version = model_manager.register_model(
                    model_name=config.project['model_name'],
                    run_id=run.info.run_id,
                    stage='Staging',
                    model_uri='model',
                    metrics={"val_accuracy": val_accuracy, "val_f1": val_f1}
                )
            else:
                print(f"\nModel is not staged because F1 score is below the threshold: {config.mlflow.model_registry_metric_threshold}")
            
            print(f"\n=== Model Registration ===")
            print(f"Model Name: {config.project['model_name']}")
            print(f"Model Version: {model_version.version}")
            print(f"Model Status: {model_version.status}")
            print("=" * 50 + "\n")
            
            # 랜덤 샘플 예측 결과 
            display_random_predictions(model, tokenizer, data_module.val_dataset, config.training_config['max_length'])
            
            # Interactive inference
            print("\n=== Interactive Inference ===")
            print("Enter your text (or 'q' to quit):")
            
            inferencer = SentimentPredictor(
                model=model,
                tokenizer=tokenizer,
                max_length=config.training_config['max_length']
            )
            
            while True:
                user_input = input("\nText: ").strip()
                if user_input.lower() == 'q':
                    break
                    
                if not user_input:
                    continue
                    
                result = inferencer.predict(user_input)[0]
                print(f"Prediction: {'긍정' if result['label'] == '긍정' else '부정'}")
                print(f"Confidence: {result['confidence']:.4f}")
            
            print("\n=== Run Information ===")
            print(f"Experiment ID: {run.info.experiment_id}")
            print(f"Run ID: {run.info.run_id}")
            print(f"Run Name: {run_name}")
            print("=" * 50 + "\n")
            
            return run.info.run_id, {"val_accuracy": val_accuracy, "val_f1": val_f1}, run_name
            
    finally:
        # 임시 파일 정리
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

def main():
    # MLflow 파일 URI 허용 설정
    os.environ['MLFLOW_ALLOW_FILE_URI_AS_MODEL_VERSION_SOURCE'] = 'true'
    
    # Load configuration
    config = Config()
    
    # MLflow 초기화 (utils/mlflow_utils.py에서 한 번만 수행)
    experiment_id = initialize_mlflow(config)
    
    # 활성 run이 있다면 종료
    active_run = mlflow.active_run()
    if active_run:
        print(f"Found active run {active_run.info.run_id}. Ending it...")
        mlflow.end_run()
    
    try:
        # Train and evaluate
        run_id, metrics, run_name = train_and_evaluate(config)
    finally:
        # 실행 종료 보장
        if mlflow.active_run():
            mlflow.end_run()

if __name__ == '__main__':
    main() 