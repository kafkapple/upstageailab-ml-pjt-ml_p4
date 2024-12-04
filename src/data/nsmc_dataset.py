import os
from pathlib import Path
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Any, Tuple
from transformers import PreTrainedTokenizerBase
import torch
import requests
from src.config import Config

import re
import emoji

def clean_text(text: str) -> str:
    """Clean text by removing special characters and emojis"""
    # 이모지 제거
    text = emoji.replace_emoji(text, '')
    
    # HTML 태그 제거
    text = re.sub(r'<[^>]+>', '', text)
    
    # 특수 문자 및 숫자 제거 (한글, 영문, 공백만 남김)
    text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text

# def preprocess_text(text: str) -> str:
#     """Preprocess text for NSMC dataset"""
#     # 기본 클리닝
#     text = clean_text(text)
    
#     # 빈 문자열 처리
#     if not text:
#         text = "빈 텍스트"
        
    return text
def download_nsmc(config):
    """Download NSMC dataset"""
    base_url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_{}.txt"
    raw_dir = Path(config.raw_data_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'test']:
        output_file = raw_dir / f"ratings_{split}.txt"
        if not output_file.exists():
            print(f"Downloading {split} dataset...")
            response = requests.get(base_url.format(split))
            output_file.write_bytes(response.content)
            print(f"Downloaded {split} dataset to {output_file}")

def sample_data(path: str, n_samples: int = 10000, random_state: int = 42):
    """Sample n rows from dataset"""
    df = pd.read_csv(path, sep='\t')
    return df.sample(n=n_samples, random_state=random_state)

class NSMCDataset(Dataset):
    def __init__(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int
    ):
        """
        NSMC 데이터셋
        
        Args:
            data: (texts, labels) 튜플 - document 컬럼을 text로 매핑
            tokenizer: 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.texts, self.labels = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # 텍스트 전처리
        text = clean_text(text)
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 배치 차원 제거
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label)
        }

class NSMCDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        config: Config, 
        tokenizer: PreTrainedTokenizerBase,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        # 데이터셋 설정
        dataset_config = self.config._config['dataset'][self.config.project['dataset_name']]
        
        # 컬럼 매핑
        self.text_column = dataset_config['column_mapping']['text']  # 기본값 사용
        self.label_column = dataset_config['column_mapping']['label']
        
        # custom 컬럼명이 제공된 경우 덮어쓰기
        if text_column:
            self.text_column = text_column
        if label_column:
            self.label_column = label_column
        
        # 데이터 경로
        self.train_path = dataset_config['train_data_path']
        self.val_path = dataset_config['val_data_path']
        
        # 모델 설정
        model_config = self.config.models[self.config.project['model_name']]['training']
        self.max_length = model_config['max_length']
        self.batch_size = model_config['batch_size']
        
        self.train_dataset = None
        self.val_dataset = None
    
    def prepare_data(self):
        """데이터 준비"""
        # 데이터 디렉토리 생성
        dataset_dir = self.config.paths['raw_data'] / self.config.project['dataset_name']
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 데이터 파일이 없으면 다운로드 (NSMC 데이터셋의 경우)
        if self.config.project['dataset_name'] == 'nsmc':
            if not self.train_path.exists():
                download_dataset('ratings_train.txt', dataset_dir)
            if not self.val_path.exists():
                download_dataset('ratings_test.txt', dataset_dir)
        
        # 데이터프레임 로드
        train_df = pd.read_csv(self.train_path, sep='\t')
        val_df = pd.read_csv(self.val_path, sep='\t')
        
        # 데이터 샘플링 적용
        dataset_config = self.config._config['dataset'][self.config.project['dataset_name']]
        sampling_rate = dataset_config['sampling_rate']
        
        if sampling_rate < 1.0:
            print(f"\nApplying sampling rate: {sampling_rate}")
            train_size = int(len(train_df) * sampling_rate)
            val_size = int(len(val_df) * sampling_rate)
            
            train_df = train_df.sample(n=train_size, random_state=42)
            val_df = val_df.sample(n=val_size, random_state=42)
            
            print(f"Sampled train size: {len(train_df)}")
            print(f"Sampled val size: {len(val_df)}")
        
        # 데이터셋 생성
        self.train_dataset = NSMCDataset(
            data=(train_df[self.text_column].values, train_df[self.label_column].values),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
        
        self.val_dataset = NSMCDataset(
            data=(val_df[self.text_column].values, val_df[self.label_column].values),
            tokenizer=self.tokenizer,
            max_length=self.max_length
        )
    
    def setup(self, stage: Optional[str] = None):
        """데이터셋 설정"""
        pass  # prepare_data에서 이미 처리됨
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

def log_data_info(data_module: NSMCDataModule):
    """데이터셋 정보 출력"""
    print("\n=== Dataset Information ===")
    
    # 학습 데이터 레이블 분포
    train_labels = [sample['labels'].item() for sample in data_module.train_dataset]
    print("\nTrain Label Distribution:")
    print(pd.Series(train_labels).value_counts())
    
    # 검증 데이터 레이블 분포
    val_labels = [sample['labels'].item() for sample in data_module.val_dataset]
    print("\nValidation Label Distribution:")
    print(pd.Series(val_labels).value_counts())

def download_dataset(filename: str, save_path: Path):
    """NSMC 데이터셋 다운로드"""
    base_url = "https://raw.githubusercontent.com/e9t/nsmc/master"
    
    # 저장 경로 생성
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / filename
    
    # 파일 다운로드
    url = f"{base_url}/{filename}"
    response = requests.get(url)
    response.raise_for_status()
    
    # 파일 저장
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename} to {file_path}")

def load_dataset(file_path: str, column_mapping: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """데이터셋 로드 및 전처리"""
    df = pd.read_csv(file_path, sep='\t')
    
    # 컬럼 이름 매핑
    text_col = column_mapping['text']
    label_col = column_mapping['label']
    
    return df[text_col].values, df[label_col].values

def sample_dataset(data: Tuple[np.ndarray, np.ndarray], sampling_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    """데이터셋 샘플링"""
    if sampling_rate >= 1.0:
        return data
    
    texts, labels = data
    n_samples = int(len(texts) * sampling_rate)
    indices = np.random.choice(len(texts), n_samples, replace=False)
    
    return texts[indices], labels[indices]