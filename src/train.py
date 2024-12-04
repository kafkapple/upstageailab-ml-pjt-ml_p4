import os
from pathlib import Path
import json
from datetime import datetime
import torch
import mlflow
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from transformers import AutoTokenizer
import time

from config import Config
from data.nsmc_dataset import NSMCDataModule, log_data_info
from inference import SentimentPredictor
from evaluator import ModelEvaluator
from utils.mlflow_utils import MLflowModelManager, initialize_mlflow
from utils.config_sync import ConfigSynchronizer
torch.set_float32_matmul_precision('medium')

from utils.path_utils import convert_to_mlflow_uri, get_mlflow_paths, join_mlflow_path
def load_run_info(run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    
    # 저장된 하이퍼파라미터 가져오기
    params = run.data.params
    
    return params
def setup_mlflow_hybrid(config):
    mlflow_paths = get_mlflow_paths(config)
    mlrun_path =mlflow_paths['backend_store_uri']
    print(f"=== Prev. mlrun_path: {mlrun_path}")
    print(f"=== Prev. artifact_root: {mlflow_paths['artifact_root']}")
    
    backend_store_uri = convert_to_mlflow_uri(mlflow_paths['backend_store_uri'])#f"file:///{mlruns_path}"
    artifact_root = convert_to_mlflow_uri(mlflow_paths['artifact_root'])
    # 1. 로컬 파일 시스템 경로 설정
    MLRUNS_DIR = mlrun_path#Path(config.mlflow.mlrun_path).absolute()
    
    # 2. MLflow 서버 URI 설정 (HTTP)
    MLFLOW_SERVER_URI = backend_store_uri
    print(f"=== New mlrun_path: {mlflow_paths['backend_store_uri']}")
    print(f"=== New artifact_root: {mlflow_paths['artifact_root']}")
    
    # 3. 실험 설정
    experiment_name = config.mlflow.experiment_name
    
    # 4. 서버 연결 및 실험 생성/가져오기
    mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
    print(f"== MLflow tracking URI: {mlflow.get_tracking_uri()}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        # 새 실험 생성 시 로컬 artifact 저장 경로 지정
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=str(MLRUNS_DIR)  # 로컬 파일 시스템 경로
        )
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    print(f"Experiment ID: {experiment_id}")
    print(f"MLRUNS_DIR /  Artifact location: {MLRUNS_DIR}")
    print(f"MLFLOW_SERVER_URI: {MLFLOW_SERVER_URI}")
    print(f"Experiment Name: {experiment_name}")

    
    return experiment, MLRUNS_DIR

def load_saved_model(run_id, config):
    config, _ = load_mlflow_config(config)
    
    # Tracking URI 설정 유지
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    # 모델 로드
    model_uri = f"runs:/{run_id}/model"
    loaded_model = mlflow.pytorch.load_model(model_uri)
    
    return loaded_model, model_uri
def load_mlflow_config(config):
 
    
    # 운영체제 독립적인 경로 생성
    base_path = Path(config.mlflow.mlrun_path).absolute()
    
    return config, base_path

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
    #experiment = setup_mlflow(config)
    ### server
    experiment, mlruns_dir = setup_mlflow_hybrid(config)
    ### Client
    mlflow_paths = get_mlflow_paths(config)
    # # MLflow 설정
    # mlruns_dir = Path(config.mlflow.mlrun_path)
    # mlruns_dir.mkdir(parents=True, exist_ok=True)
   
    artifact_root = mlflow_paths['artifact_root']
    
    print(f"\n=== Path Debugging ===")
    print(f"1. Server: MLflow directory: {mlruns_dir}")
    print(f"2. Client: Artifact root: {artifact_root}")
    
    # Set/get experiment
    # try:
    #     experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
    #     if experiment is not None:
    #         print(f"4. Existing experiment artifact location: {experiment.artifact_location}")
        
    #     if experiment is None:
    #         print(f"Creating new experiment: {config.mlflow.experiment_name}")
    #         experiment_id = mlflow.create_experiment(
    #             name=config.mlflow.experiment_name,
    #             artifact_location=artifact_root
    #         )
    #         experiment = mlflow.get_experiment(experiment_id)
    #         print(f"5. New experiment artifact location: {experiment.artifact_location}")
    #     else:
    #         print(f"Using existing experiment: {config.mlflow.experiment_name}")
    #         experiment_id = experiment.experiment_id
    
    #     print(f"\n=== MLflow Experiment ===")
    #     print(f"Experiment ID: {experiment_id}")
    #     print(f"Experiment Name: {experiment.name}")
    #     print(f"Artifact Location: {experiment.artifact_location}")
    #     print(f"Lifecycle Stage: {experiment.lifecycle_stage}")
        
    # except Exception as e:
    #     print(f"Error handling experiment: {str(e)}")
    #     raise
    
    seed_everything(config.project['random_state'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.project['model_name']}_{config.project['dataset_name']}_{timestamp}"
    
    try:
        with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
            run_id = run.info.run_id
            print("\n=== MLflow Run Configuration ===")
            print(f"Run Name: {run_name}")
            print(f"Run ID: {run_id}")
            print(f"Artifact URI: {run.info.artifact_uri}")
            
            # Model initialization based on model type
            if config.project['model_name'].startswith('KcBERT'):
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_config['pretrained_model'],
                    model_max_length=config.training_config['max_length']
                )
                from models.kcbert_model import KcBERT
                model = KcBERT(
                    pretrained_model=config.model_config['pretrained_model'],
                    num_labels=config.training_config['num_labels'],
                    num_unfreeze_layers=config.training_config['num_unfreeze_layers'],
                    learning_rate=config.training_config['lr'],
                    optimizer=config.training_config['optimizer'],
                    scheduler=config.training_config['lr_scheduler']
                )
            elif config.project['model_name'].startswith('KcELECTRA'):
                tokenizer = AutoTokenizer.from_pretrained(
                    config.model_config['pretrained_model'],
                    model_max_length=config.training_config['max_length']
                )
                from models.kcelectra_model import KcELECTRA
                model = KcELECTRA(
                    pretrained_model=config.model_config['pretrained_model'],
                    num_labels=config.training_config['num_labels'],
                    num_unfreeze_layers=config.training_config['num_unfreeze_layers'],
                    learning_rate=config.training_config['lr'],
                    optimizer=config.training_config['optimizer'],
                    scheduler=config.training_config['lr_scheduler']
                )
            else:
                raise ValueError(f"Unknown model: {config.project['model_name']}")
            
            # Trainer configuration
            trainer_config = config.common['trainer'].copy()
            trainer_config.update({
                'max_epochs': int(config.training_config['epochs']),
                'precision': config.training_config['precision'],
                'accumulate_grad_batches': config.training_config['accumulate_grad_batches'],
            })
            
            if 'logger' in trainer_config:
                del trainer_config['logger']
            
            # Callbacks
            checkpoint_callback = ModelCheckpoint(**config.common['checkpoint'])
            callbacks = [
                checkpoint_callback,
                EarlyStopping(monitor='val_loss', patience=3, mode='min'),
                LearningRateMonitor(logging_interval='step')
            ]
            
            # Logger
            logger = pl.loggers.MLFlowLogger(
                experiment_name=config.mlflow.experiment_name,
                tracking_uri=mlflow.get_tracking_uri(),
                run_id=run.info.run_id
            )
            
            trainer = pl.Trainer(
                **trainer_config,
                callbacks=callbacks,
                logger=logger
            )
            
            # Data module
            data_module = NSMCDataModule(
                config=config,
                tokenizer=tokenizer,
                batch_size=config.training_config['batch_size'],
                max_length=config.training_config['max_length']
            )
            
            data_module.prepare_data()
            data_module.setup(stage='fit')
            
            # Log data info
            log_data_info(data_module)
            
            # Train
            trainer.fit(model, data_module)
            
            # Evaluate
            evaluator = ModelEvaluator(model, tokenizer)
            eval_metrics = evaluator.evaluate_dataset(data_module)
            
            val_accuracy = eval_metrics['accuracy']
            val_f1 = eval_metrics['f1']
            
            print("\n=== Validation Results ===")
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation F1 Score: {val_f1:.4f}")
            
            # Log metrics
            mlflow.log_metrics({
                "val_accuracy": val_accuracy,
                "val_f1": val_f1
            })
            
            # Save and log model
            print("\nSaving model artifacts...")
            try:
                # Save model locally first
                # 실제 저장 경로 출력
                
      
                # 모델 저장
                # artifact_model_path = "model"
                artifact_path = join_mlflow_path(artifact_root, str(experiment.experiment_id),str(run_id),  "artifacts/model")
                print(f"Artifact path: {artifact_path}")
                mlflow.pytorch.log_model(model, 'model')
                #log_model -> Model.log -> log_artifacts -> MlflowClient().log_artifacts -> _get_artifact_repo
                params  = load_run_info(run.info.run_id)
                print(f"Run Param Info: {params}")
                print(f"Model saved at: {artifact_path}")

                
                model_artifact_path = join_mlflow_path(artifact_path, "data", "model.pth")
                print(f"Model artifact path: {model_artifact_path}")
                # mlflow.log_artifact(model_artifact_path, "model")
                
                # local_model_path = Path(config.paths['models_base']) / config.project['model_name'] / run_name
                # local_model_path.mkdir(parents=True, exist_ok=True)
                # print(f"Local model path: {local_model_path}")
                # # Save state dict
                # print("Saving model state dict...")
                # torch.save(model.state_dict(), local_model_path / "model.pt")
                # Save config
                print("Saving model config...")
                config_content = {
                    "model_type": config.project['model_name'],
                    "pretrained_model": config.model_config['pretrained_model'],
                    "num_labels": config.training_config['num_labels'],
                    "max_length": config.training_config['max_length'],
                    "num_unfreeze_layers": config.training_config['num_unfreeze_layers']
                }
                with open(join_mlflow_path(artifact_path, "config.json"), 'w', encoding='utf-8') as f:
                    json.dump(config.training_config, f, indent=2, ensure_ascii=False)
                with open(join_mlflow_path(artifact_path, "config_content.json"), 'w', encoding='utf-8') as f:
                    json.dump(config_content, f, indent=2, ensure_ascii=False)
                with open(join_mlflow_path(artifact_path, "config_params.json"), 'w', encoding='utf-8') as f:
                    json.dump(params, f, indent=2, ensure_ascii=False)
                # Log artifacts
                print("Logging model artifacts...")
                mlflow.log_artifact(join_mlflow_path(artifact_path, "model.pt"), "model")
                mlflow.log_artifact(join_mlflow_path(artifact_path, "config.json"), "model")
                mlflow.log_artifact(join_mlflow_path(artifact_path, "config_content.json"), "model")
                mlflow.log_artifact(join_mlflow_path(artifact_path, "config_params.json"), "model")
                print("Successfully logged model artifacts")
                
                # Model registration using MLflowModelManager
                model_manager = MLflowModelManager(config)
                if val_f1 > config.mlflow.model_registry_metric_threshold:
                    try:
                        print("\nAttempting to register model...")
                        
                        _, model_uri = load_saved_model(run_id, config)
                        print(f"Model URI: {model_uri}")
                        model_version = model_manager.register_model(
                            model_name=config.project['model_name'],
                            run_id=run_id,
                            stage='Staging',
                            model_uri=model_uri,
                            metrics={"val_accuracy": val_accuracy, "val_f1": val_f1}
                        )
                        print(f"\n=== Model Registration ===")
                        print(f"Model Name: {config.project['model_name']}")
                        if model_version:
                            print(f"Model Version: {model_version.version}")
                            print(f"Model Status: {model_version.status}")
                    except Exception as e:
                        print(f"\nError registering model: {str(e)}")
                        print("Model artifacts were saved but registration failed.")
                        print("You can register the model manually using the MLflow UI.")
                else:
                    print(f"\nModel is not staged because F1 score ({val_f1:.4f}) is below the threshold ({config.mlflow.model_registry_metric_threshold})")
            except Exception as e:
                print(f"Error during model artifact handling: {str(e)}")
                raise
            
            return run_id, {"val_accuracy": val_accuracy, "val_f1": val_f1}, run_name
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

def main():
    # Load configuration
    config = Config()
    
    try:
        # Train and evaluate
        run_id, metrics, run_name = train_and_evaluate(config)
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 