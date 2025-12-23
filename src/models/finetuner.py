"""
LoRA fine-tuning for sentiment analysis
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from loguru import logger
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

class SentimentDataset(Dataset):
    """Dataset for sentiment classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class LoRAFineTuner:
    """LoRA fine-tuning for sentiment analysis"""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        device: str = None
    ):
        """
        Initialize LoRA fine-tuner
        
        Args:
            model_name: Base model to fine-tune
            lora_r: LoRA rank
            lora_alpha: LoRA alpha scaling
            lora_dropout: Dropout probability
        """
        self.model_name = model_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading {model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_lin", "v_lin"] if "distilbert" in model_name.lower() else ["query", "value"]
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
        logger.info(f"✅ Model with LoRA loaded!")
    
    def prepare_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        max_train_samples: Optional[int] = None
    ):
        """Prepare datasets"""
        
        # Optionally subsample for faster training
        if max_train_samples and len(train_df) > max_train_samples:
            logger.info(f"Subsampling training data: {max_train_samples} samples")
            train_df = train_df.sample(n=max_train_samples, random_state=42)
        
        # Create datasets
        self.train_dataset = SentimentDataset(
            train_df['text'].values,
            train_df['label'].values,
            self.tokenizer
        )
        
        self.val_dataset = SentimentDataset(
            val_df['text'].values,
            val_df['label'].values,
            self.tokenizer
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(
        self,
        output_dir: str = "outputs/models/lora_finetuned",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        weight_decay: float = 0.01
    ):
        """Train model with LoRA"""
        
        logger.info("Starting LoRA fine-tuning...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="wandb",
            remove_unused_columns=False,
            push_to_hub=False,
            fp16=True
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Track training time
        start_time = time.time()
        
        # Train
        train_result = trainer.train()
        
        training_time = time.time() - start_time
        
        # Log training stats
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Train loss: {train_result.training_loss:.4f}")
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training metadata
        metadata = {
            "model_name": self.model_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "training_time": training_time,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset)
        }
        
        with open(Path(output_dir) / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Model saved to {output_dir}")
        
        return trainer, metadata
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate model on test set"""
        
        logger.info("Evaluating on test set...")
        
        # Create test dataset
        test_dataset = SentimentDataset(
            test_df['text'].values,
            test_df['label'].values,
            self.tokenizer
        )
        
        # Create dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_size=16,
            shuffle=False
        )
        
        # Evaluation
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro'
        )
        
        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
        
        logger.info(f"\n📊 Test Results:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   F1 Score: {f1:.4f}")
        
        return results

def train_lora_model(
    model_name: str = "distilbert-base-uncased",
    max_train_samples = 10000,
    num_epochs = 3,
    batch_size = 64
):
    """Complete LoRA fine-tuning pipeline"""
    
    from src.data.loader import load_processed_data
    
    # Load data
    logger.info("Loading datasets...")
    train_df = load_processed_data('train')
    val_df = load_processed_data('val')
    test_df = load_processed_data('test')
    
    # Initialize W&B
    wandb.init(
        project="week4-nlp-llms",
        name=f"lora_{model_name.split('/')[-1]}",
        config={
            "model_name": model_name,
            "lora_r": 8,
            "lora_alpha": 16,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "max_train_samples": max_train_samples
        }
    )
    
    # Initialize fine-tuner
    finetuner = LoRAFineTuner(
        model_name=model_name,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    # Prepare data
    finetuner.prepare_data(train_df, val_df, max_train_samples=max_train_samples)
    
    # Train
    trainer, metadata = finetuner.train(
        output_dir="outputs/models/lora_finetuned",
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=2e-4
    )
    
    # Evaluate on test set
    test_results = finetuner.evaluate(test_df)
    
    # Log to W&B
    wandb.log({
        "test_accuracy": test_results['accuracy'],
        "test_f1": test_results['f1'],
        "test_precision": test_results['precision'],
        "test_recall": test_results['recall'],
        "training_time": metadata['training_time']
    })
    
    # Save results
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "lora_results.json", 'w') as f:
        json.dump({
            "metadata": metadata,
            "test_results": test_results
        }, f, indent=2)
    
    wandb.finish()
    
    logger.info(f"\n✅ LoRA fine-tuning complete!")
    logger.info(f"   Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"   Test F1: {test_results['f1']:.4f}")
    logger.info(f"   Training Time: {metadata['training_time']:.2f}s")
    
    return finetuner, test_results

if __name__ == "__main__":
    # Login to W&B
    wandb.login()
    
    # Train LoRA model
    # Options:
    # - "distilbert-base-uncased" (fast, 66M params)
    # - "bert-base-uncased" (better, 110M params)
    # - "roberta-base" (best, 125M params)
    
    train_lora_model(
        model_name="distilbert-base-uncased",
        max_train_samples=10000,  # Use subset for faster training
        num_epochs=3,
        batch_size=16
    )