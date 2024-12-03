import torch
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tempfile
import os
import torch
from PIL import Image
import io

class ModelEvaluator:
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate_dataset(self, data_module, n_samples: int = 5) -> Dict:
        """전체 데이터셋에 대한 평가 수행"""
        metrics = {}
        
        # 전체 검증 세트에 대한 메트릭 계산
        predictions, labels, confidences = self._get_predictions(data_module.val_dataloader())
        
        # 기본 메트릭 계산
        metrics['accuracy'] = self._calculate_accuracy(predictions, labels)
        metrics['avg_confidence'] = np.mean(confidences)
        
        # F1 스코어 계산
        metrics['f1'] = f1_score(labels, predictions, average='weighted')
        
        # 신뢰도 구간별 정확도
        confidence_metrics = self._calculate_confidence_bins(predictions, labels, confidences)
        metrics.update(confidence_metrics)
        
        # 샘플 예측 결과
        sample_predictions = self._get_sample_predictions(data_module.val_dataset, n_samples)
        metrics['sample_predictions'] = sample_predictions
        
        return metrics
    
    def _get_predictions(self, dataloader) -> Tuple[List, List, List]:
        """데이터로더로부터 예측 수행"""
        predictions, labels, confidences = [], [], []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**{k: v for k, v in batch.items() if k != 'label'})
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                preds = torch.argmax(logits, dim=-1)
                confs = torch.max(probs, dim=-1)[0]
                
                predictions.extend(preds.cpu().numpy())
                labels.extend(batch['label'].cpu().numpy())
                confidences.extend(confs.cpu().numpy())
                
        return predictions, labels, confidences    
    def _calculate_accuracy(self, predictions: List, labels: List) -> float:
        """정확도 계산"""
        return sum(p == l for p, l in zip(predictions, labels)) / len(labels)
    
    def _calculate_confidence_bins(self, predictions: List, labels: List, confidences: List) -> Dict:
        """신뢰도 구간별 정확도 계산"""
        bins = {}
        confidence_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        
        for low, high in confidence_bins:
            mask = [(c >= low and c < high) for c in confidences]
            if sum(mask) > 0:
                bin_acc = sum(p == l for p, l, m in zip(predictions, labels, mask) if m) / sum(mask)
                bin_count = sum(mask)
                bins[f'confidence_{int(low*100)}_{int(high*100)}'] = {
                    'accuracy': bin_acc,
                    'count': bin_count
                }
        
        return {'confidence_bins': bins}
    
    def _get_sample_predictions(self, dataset, n_samples=5):
        """샘플 데이터에 대한 예측 결과 반환"""
        indices = torch.randperm(len(dataset))[:n_samples].tolist()
        predictions = []
        
        self.model.eval()
        with torch.no_grad():
            for idx in indices:
                text = dataset.texts[idx]
                true_label = dataset.labels[idx]
                
                sample = dataset[idx]
                inputs = {
                    'input_ids': sample['input_ids'].unsqueeze(0).to(self.model.device),
                    'attention_mask': sample['attention_mask'].unsqueeze(0).to(self.model.device)
                }
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_label = torch.argmax(logits, dim=-1).item()
                confidence = probs[0][pred_label].item()
                
                predictions.append({
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidence
                })
        
        return predictions
        
    @staticmethod
    def plot_confusion_matrix(dataset, model, tokenizer, labels=['Negative', 'Positive'], normalize=True):
        """Generate Confusion Matrix from model predictions
        
        Args:
            dataset: Dataset to evaluate
            model: Model to evaluate
            tokenizer: Tokenizer
            labels: Label names
            normalize: Whether to normalize values
        
        Returns:
            PIL Image object
        """
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                inputs = {
                    'input_ids': sample['input_ids'].unsqueeze(0).to(model.device),
                    'attention_mask': sample['attention_mask'].unsqueeze(0).to(model.device)
                }
                
                outputs = model(**inputs)
                logits = outputs.logits
                pred_label = torch.argmax(logits, dim=-1).item()
                
                y_true.append(sample['label'].item())
                y_pred.append(pred_label)
        
        # Save image to memory
        buf = io.BytesIO()
        
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        
        plt.figure(figsize=(10, 8))
        
        # Use 'Blues' colormap for better visibility
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2%' if normalize else 'd',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar=True,
            square=True
        )
        
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
                fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12, labelpad=10)
        plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
        
        # Set font properties for better visibility
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.size'] = 12
        
        plt.tight_layout()
        
        # Save image to memory and convert to PIL Image
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        
        buf.seek(0)
        image = Image.open(buf)
        
        return image
