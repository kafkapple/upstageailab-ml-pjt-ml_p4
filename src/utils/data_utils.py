import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Optional
import random
import sys


def split_dataset(
    file_path: Union[str, Path], 
    n_splits: int, 
    seed: Optional[int] = 42,
    shuffle: bool = True
) -> list:
    """데이터셋을 n개로 랜덤 분할하여 저장
    
    Args:
        file_path (str or Path): 데이터셋 파일 경로 (.txt 또는 .tsv)
        n_splits (int): 분할할 개수
        seed (int, optional): 랜덤 시드. 기본값 42
        shuffle (bool): 데이터 셔플 여부. 기본값 True
    
    Returns:
        list: 생성된 파일들의 경로 리스트
    """
    try:
        # 경로 객체로 변환
        file_path = Path(file_path)
        
        # 파일 확장자 확인
        if file_path.suffix not in ['.txt', '.tsv']:
            raise ValueError("지원되는 파일 형식은 .txt와 .tsv입니다.")
        
        # 데이터 로드
        print(f"데이터셋 로드 중: {file_path}")
        if file_path.suffix == '.txt':
            df = pd.read_csv(file_path, sep='\t')
        else:
            df = pd.read_csv(file_path, sep='\t')
        
        # 데이터 셔플
        if shuffle:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            df = df.sample(frac=1).reset_index(drop=True)
        
        # 데이터 크기 계산
        total_size = len(df)
        split_size = total_size // n_splits
        
        # 결과 파일 경로 저장용
        result_files = []
        
        # 데이터 분할 및 저장
        for i in range(n_splits):
            # 마지막 분할은 나머지 모든 데이터 포함
            if i == n_splits - 1:
                split_df = df[i * split_size:]
            else:
                split_df = df[i * split_size:(i + 1) * split_size]
            
            # 새 파일명 생성
            new_file_path = file_path.parent / f"{file_path.stem}_{i+1}{file_path.suffix}"
            
            # 데이터 저장
            split_df.to_csv(new_file_path, sep='\t', index=False)
            result_files.append(new_file_path)
            
            print(f"분할 {i+1}/{n_splits} 저장 완료: {new_file_path} (크기: {len(split_df)})")
        
        print(f"\n데이터셋 분할 완료:")
        print(f"- 원본 크기: {total_size}")
        print(f"- 분할 개수: {n_splits}")
        print(f"- 기본 분할 크기: {split_size}")
        
        return result_files
        
    except Exception as e:
        print(f"데이터셋 분할 중 오류 발생: {str(e)}")
        raise

# 사용 예시
if __name__ == "__main__":
    # 예시 실행
    try:
        file_path = 'data/processed/nsmc_train.tsv'
        n_splits = 5
        print(f"file_path: {file_path}")
        
        split_files = split_dataset(
            file_path=file_path,
            n_splits=n_splits,
            seed=42
        )
        
        print("\n생성된 파일들:")
        for file_path in split_files:
            print(f"- {file_path}")
            
    except Exception as e:
        print(f"실행 중 오류 발생: {str(e)}") 