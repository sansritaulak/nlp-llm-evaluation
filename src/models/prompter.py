"""
LLM-based sentiment analysis using HuggingFace FLAN-T5
"""
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from loguru import logger
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import wandb

class SentimentPrompter:
    """Prompt-based sentiment analysis using FLAN-T5"""
    
    # Prompt templates
    PROMPTS = {
        "zero_shot": """Analyze the sentiment of the following movie review.
Respond with ONLY one word: either "positive" or "negative".

Review: {text}

Sentiment:""",
        
        "one_shot": """Analyze the sentiment of movie reviews.

Example:
Review: "This movie was fantastic! The acting was superb and the plot kept me engaged."
Sentiment: positive

Now analyze this review:
Review: {text}

Sentiment:""",
        
        "three_shot": """Analyze the sentiment of movie reviews.

Examples:
Review: "This movie was fantastic! The acting was superb and the plot kept me engaged."
Sentiment: positive

Review: "Terrible waste of time. The plot made no sense and the acting was wooden."
Sentiment: negative

Review: "A masterpiece of cinema. Every scene was beautifully crafted."
Sentiment: positive

Now analyze this review:
Review: {text}

Sentiment:""",
        
        "chain_of_thought": """Analyze the sentiment of the following movie review step by step.

Review: {text}

Think through this carefully:
1. What are the key emotional words?
2. Is the reviewer praising or criticizing?
3. What's the overall tone?

Based on your analysis, is the sentiment positive or negative? Respond with ONLY: "positive" or "negative"

Sentiment:""",
        
        "role_based": """You are an expert movie critic and sentiment analyst with years of experience.

Your task is to determine if the following movie review expresses a positive or negative sentiment.

Review: {text}

Based on your expert analysis, classify this as either "positive" or "negative". Respond with ONLY one word.

Sentiment:"""
    }
    
    def __init__(
        self, 
        model_name: str = "google/flan-t5-base",
        hf_token: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize with HuggingFace FLAN-T5 model
        
        Args:
            model_name: HuggingFace model name (flan-t5-small, base, or large)
            hf_token: HuggingFace token (optional, for gated models)
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading {model_name} on {self.device}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            token=self.hf_token
        ).to(self.device)
        
        logger.info(f"✅ Model loaded successfully!")
    
    def predict_single(
        self,
        text: str,
        prompt_type: str = "zero_shot",
        max_new_tokens: int = 10,
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """
        Predict sentiment for a single review
        
        Returns:
            Dict with prediction, confidence, latency
        """
        if prompt_type not in self.PROMPTS:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        # Format prompt
        prompt = self.PROMPTS[prompt_type].format(text=text[:500])  # Truncate long texts
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Time the inference
        start_time = time.time()
        
        try:
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    num_beams=1
                )
            
            latency = time.time() - start_time
            
            # Decode response
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
            
            # Parse sentiment
            if "positive" in raw_response:
                sentiment = 1
            elif "negative" in raw_response:
                sentiment = 0
            else:
                logger.warning(f"Unexpected response: {raw_response}")
                sentiment = -1  # Invalid
            
            return {
                "prediction": sentiment,
                "raw_response": raw_response,
                "latency": latency,
                "tokens_input": inputs['input_ids'].shape[1],
                "tokens_output": outputs.shape[1],
                "prompt_type": prompt_type
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                "prediction": -1,
                "raw_response": str(e),
                "latency": time.time() - start_time,
                "tokens_input": 0,
                "tokens_output": 0,
                "prompt_type": prompt_type,
                "error": str(e)
            }
    
    def predict_batch(
        self,
        texts: List[str],
        prompt_type: str = "zero_shot",
        max_samples: Optional[int] = None,
        batch_size: int = 8
    ) -> pd.DataFrame:
        """Predict sentiment for multiple reviews"""
        
        if max_samples:
            texts = texts[:max_samples]
        
        results = []
        
        logger.info(f"Processing {len(texts)} samples with prompt type: {prompt_type}")
        
        # Process in batches for efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Evaluating {prompt_type}"):
            batch_texts = texts[i:i+batch_size]
            
            for text in batch_texts:
                result = self.predict_single(text, prompt_type=prompt_type)
                result['text'] = text[:100]  # Store truncated text
                results.append(result)
        
        return pd.DataFrame(results)
    
    def evaluate_prompt(
        self,
        df: pd.DataFrame,
        prompt_type: str,
        max_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate a specific prompt on a dataset
        
        Args:
            df: DataFrame with 'text' and 'label' columns
            prompt_type: Type of prompt to use
            max_samples: Maximum number of samples to evaluate
        
        Returns:
            Dict with metrics and detailed results
        """
        # Sample data
        eval_df = df.sample(n=min(max_samples, len(df)), random_state=42)
        
        logger.info(f"Evaluating {prompt_type} on {len(eval_df)} samples...")
        
        # Get predictions
        results_df = self.predict_batch(
            eval_df['text'].tolist(),
            prompt_type=prompt_type,
            batch_size=8
        )
        
        # Add true labels
        results_df['true_label'] = eval_df['label'].values
        
        # Calculate metrics
        valid_mask = results_df['prediction'] != -1
        valid_results = results_df[valid_mask]
        
        if len(valid_results) == 0:
            logger.error(f"No valid predictions for {prompt_type}")
            return {"error": "No valid predictions"}
        
        accuracy = (valid_results['prediction'] == valid_results['true_label']).mean()
        
        # Per-class accuracy
        pos_mask = valid_results['true_label'] == 1
        neg_mask = valid_results['true_label'] == 0
        
        pos_acc = (valid_results[pos_mask]['prediction'] == 1).mean() if pos_mask.any() else 0
        neg_acc = (valid_results[neg_mask]['prediction'] == 0).mean() if neg_mask.any() else 0
        
        # Latency stats
        avg_latency = valid_results['latency'].mean()
        p95_latency = valid_results['latency'].quantile(0.95)
        
        # Token usage
        total_tokens = results_df['tokens_input'].sum() + results_df['tokens_output'].sum()
        avg_tokens = total_tokens / len(results_df)
        
        metrics = {
            "prompt_type": prompt_type,
            "accuracy": float(accuracy),
            "positive_accuracy": float(pos_acc),
            "negative_accuracy": float(neg_acc),
            "samples_evaluated": len(results_df),
            "valid_predictions": len(valid_results),
            "invalid_predictions": len(results_df) - len(valid_results),
            "avg_latency_sec": float(avg_latency),
            "p95_latency_sec": float(p95_latency),
            "total_tokens": int(total_tokens),
            "avg_tokens_per_request": float(avg_tokens)
        }
        
        logger.info(f"\n📊 Results for {prompt_type}:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Avg Latency: {avg_latency:.2f}s")
        logger.info(f"   Total Tokens: {total_tokens:,}")
        
        return {
            "metrics": metrics,
            "detailed_results": results_df.to_dict('records')
        }
    
    def compare_prompts(
        self,
        df: pd.DataFrame,
        prompt_types: Optional[List[str]] = None,
        max_samples: int = 100
    ) -> pd.DataFrame:
        """Compare multiple prompt types"""
        
        if prompt_types is None:
            prompt_types = list(self.PROMPTS.keys())
        
        # Initialize W&B
        wandb.init(
            project="week4-nlp-llms",
            name="prompt_comparison",
            config={
                "model": self.model_name,
                "max_samples": max_samples,
                "device": self.device
            }
        )
        
        all_results = []
        
        for prompt_type in prompt_types:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {prompt_type}")
            logger.info(f"{'='*60}")
            
            result = self.evaluate_prompt(df, prompt_type, max_samples)
            
            if "error" not in result:
                all_results.append(result['metrics'])
                
                # Log to W&B
                wandb.log({
                    f"{prompt_type}_accuracy": result['metrics']['accuracy'],
                    f"{prompt_type}_latency": result['metrics']['avg_latency_sec'],
                    f"{prompt_type}_pos_acc": result['metrics']['positive_accuracy'],
                    f"{prompt_type}_neg_acc": result['metrics']['negative_accuracy']
                })
                
                # Save detailed results
                output_dir = Path("outputs/results/prompts")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                with open(output_dir / f"{prompt_type}_results.json", 'w') as f:
                    json.dump(result, f, indent=2)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_results)
        
        # Save comparison
        comparison_df.to_csv("outputs/results/prompt_comparison.csv", index=False)
        
        # Log comparison table to W&B
        wandb.log({
            "prompt_comparison": wandb.Table(dataframe=comparison_df)
        })
        
        logger.info("\n" + "="*60)
        logger.info("PROMPT COMPARISON")
        logger.info("="*60)
        print(comparison_df[['prompt_type', 'accuracy', 'avg_latency_sec', 'positive_accuracy', 'negative_accuracy']])
        
        wandb.finish()
        
        return comparison_df

def run_prompt_evaluation(
    model_name: str = "google/flan-t5-base",
    max_samples: int = 100
):
    """Main evaluation pipeline"""
    from src.data.loader import load_processed_data
    
    logger.info(f"Starting prompt evaluation with {model_name}")
    
    # Load validation data
    val_df = load_processed_data('val')
    
    # Initialize prompter
    prompter = SentimentPrompter(model_name=model_name)
    
    # Compare all prompts
    comparison = prompter.compare_prompts(
        val_df,
        max_samples=max_samples
    )
    
    return comparison

if __name__ == "__main__":
    # Login to W&B
    wandb.login()
    
    # Run evaluation
    # Options: "google/flan-t5-small" (fast), "google/flan-t5-base" (balanced), "google/flan-t5-large" (best)
    run_prompt_evaluation(
        model_name="google/flan-t5-base",
        max_samples=100  # Adjust based on your time/resources
    )