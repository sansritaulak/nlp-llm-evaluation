"""
Baseline sentiment classifier with W&B logging
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
from loguru import logger
import wandb

class BaselineClassifier:
    """TF-IDF + ML baseline"""
    
    def __init__(
        self,
        model_type: str = "logistic",
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        **model_kwargs
    ):
        self.model_type = model_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=5,
            max_df=0.8
        )
        
        # Initialize classifier
        if model_type == "logistic":
            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42,
                **model_kwargs
            )
        elif model_type == "svm":
            self.classifier = LinearSVC(
                max_iter=1000,
                random_state=42,
                **model_kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Initialized {model_type} with {max_features} features")
    
    def fit(self, X_train: pd.Series, y_train: pd.Series):
        """Train the model"""
        logger.info(f"Training on {len(X_train):,} samples...")
        
        # Fit vectorizer
        X_train_vec = self.vectorizer.fit_transform(X_train)
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        
        # Train classifier
        self.classifier.fit(X_train_vec, y_train)
        logger.info("✅ Training complete")
        
        return self
    
    def predict(self, X: pd.Series) -> np.ndarray:
        """Make predictions"""
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict(X_vec)
    
    def predict_proba(self, X: pd.Series) -> np.ndarray:
        """Get probabilities (logistic only)"""
        if self.model_type != "logistic":
            raise ValueError("predict_proba only for logistic")
        X_vec = self.vectorizer.transform(X)
        return self.classifier.predict_proba(X_vec)
    
    def evaluate(self, X: pd.Series, y: pd.Series) -> Dict[str, Any]:
        """Calculate metrics"""
        y_pred = self.predict(X)
        
        accuracy = accuracy_score(y, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average='macro'
        )
        
        # Per-class metrics
        precision_pc, recall_pc, f1_pc, support = \
            precision_recall_fscore_support(y, y_pred, average=None)
        
        cm = confusion_matrix(y, y_pred)
        
        return {
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "per_class": {
                "negative": {
                    "precision": float(precision_pc[0]),
                    "recall": float(recall_pc[0]),
                    "f1": float(f1_pc[0]),
                    "support": int(support[0])
                },
                "positive": {
                    "precision": float(precision_pc[1]),
                    "recall": float(recall_pc[1]),
                    "f1": float(f1_pc[1]),
                    "support": int(support[1])
                }
            },
            "confusion_matrix": cm.tolist()
        }
    
    def get_important_features(self, top_n: int = 20) -> Dict[str, Any]:
        """Get most important features"""
        if self.model_type != "logistic":
            return {}
        
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        coef = self.classifier.coef_[0]
        
        # Top positive
        top_pos_idx = np.argsort(coef)[-top_n:][::-1]
        top_pos = [(feature_names[i], float(coef[i])) for i in top_pos_idx]
        
        # Top negative
        top_neg_idx = np.argsort(coef)[:top_n]
        top_neg = [(feature_names[i], float(coef[i])) for i in top_neg_idx]
        
        return {
            "positive_features": top_pos,
            "negative_features": top_neg
        }
    
    def save(self, path: Path):
        """Save model"""
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.vectorizer, path / "vectorizer.pkl")
        joblib.dump(self.classifier, path / "classifier.pkl")
        
        config = {
            "model_type": self.model_type,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range
        }
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"💾 Model saved to {path}")
    
    @classmethod
    def load(cls, path: Path):
        """Load model"""
        with open(path / "config.json", 'r') as f:
            config = json.load(f)
        
        model = cls(**config)
        model.vectorizer = joblib.load(path / "vectorizer.pkl")
        model.classifier = joblib.load(path / "classifier.pkl")
        
        logger.info(f"📂 Model loaded from {path}")
        return model

def train_baseline(model_type: str = "logistic"):
    """Train baseline with W&B logging"""
    
    # Load data
    from src.data.loader import load_processed_data
    
    train_df = load_processed_data('train')
    val_df = load_processed_data('val')
    test_df = load_processed_data('test')
    
    # Initialize W&B
    wandb.init(
        project="week4-nlp-llms",
        name=f"baseline_{model_type}",
        config={
            "model_type": model_type,
            "max_features": 10000,
            "ngram_range": (1, 2),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df)
        }
    )
    
    # Train model
    model = BaselineClassifier(model_type=model_type)
    model.fit(train_df['text'], train_df['label'])
    
    # Evaluate on validation
    val_results = model.evaluate(val_df['text'], val_df['label'])
    logger.info(f"\n📊 Validation Results:")
    logger.info(f"   Accuracy: {val_results['accuracy']:.4f}")
    logger.info(f"   Macro F1: {val_results['macro_f1']:.4f}")
    
    # Log to W&B
    wandb.log({
        "val_accuracy": val_results['accuracy'],
        "val_macro_f1": val_results['macro_f1'],
        "val_precision": val_results['macro_precision'],
        "val_recall": val_results['macro_recall']
    })
    
    # Evaluate on test
    test_results = model.evaluate(test_df['text'], test_df['label'])
    logger.info(f"\n📊 Test Results:")
    logger.info(f"   Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"   Macro F1: {test_results['macro_f1']:.4f}")
    
    # Log to W&B
    wandb.log({
        "test_accuracy": test_results['accuracy'],
        "test_macro_f1": test_results['macro_f1'],
        "test_precision": test_results['macro_precision'],
        "test_recall": test_results['macro_recall']
    })
    
    # Log confusion matrix
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=test_df['label'].values,
            preds=model.predict(test_df['text']),
            class_names=["Negative", "Positive"]
        )
    })
    
    # Save results
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f"baseline_{model_type}_results.json", 'w') as f:
        json.dump({
            "validation": val_results,
            "test": test_results,
            "important_features": model.get_important_features()
        }, f, indent=2)
    
    # Save model
    model_dir = Path(f"outputs/models/baseline_{model_type}")
    model.save(model_dir)
    
    # Save model to W&B
    wandb.save(str(model_dir / "*"))
    
    wandb.finish()
    
    return model

if __name__ == "__main__":
    # Login to W&B first
    wandb.login()
    
    # --- 1. Train Logistic Regression ---
    logger.info("="*60)
    logger.info("Training Logistic Regression...")
    train_baseline(model_type="logistic")
    
    # --- 2. Train Linear SVM Classifier ---
    logger.info("\n" + "="*60)
    logger.info("Training Linear SVM Classifier...")
    train_baseline(model_type="svm")
    
    logger.info("\n" + "="*60)
    logger.info("✅ All training complete! Check W&B dashboard for results.")