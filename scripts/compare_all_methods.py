"""
Compare all three methods: Baseline, Prompting, LoRA Fine-tuning
"""
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
import wandb

def load_baseline_results():
    """Load baseline model results"""
    path = Path("outputs/results/baseline_logistic_results.json")
    if not path.exists():
        logger.warning("Baseline results not found")
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return {
        "method": "Baseline (TF-IDF + Logistic)",
        "accuracy": data['test']['accuracy'],
        "f1": data['test']['macro_f1'],
        "precision": data['test']['macro_precision'],
        "recall": data['test']['macro_recall'],
        "training_time": "~3 min",  # Approximate
        "inference_time": "~0.01s",  # Very fast
        "model_size": "~50MB"
    }

def load_prompting_results():
    """Load best prompting results"""
    path = Path("outputs/results/prompt_comparison.csv")
    if not path.exists():
        logger.warning("Prompting results not found")
        return None
    
    df = pd.read_csv(path)
    best_idx = df['accuracy'].idxmax()
    best = df.iloc[best_idx]
    
    return {
        "method": f"Prompting ({best['prompt_type']})",
        "accuracy": best['accuracy'],
        "f1": None,  # Not calculated for prompting
        "precision": None,
        "recall": None,
        "training_time": "0 (no training)",
        "inference_time": f"{best['avg_latency_sec']:.2f}s",
        "model_size": "N/A (API)"
    }

def load_lora_results():
    """Load LoRA fine-tuning results"""
    path = Path("outputs/results/lora_results.json")
    if not path.exists():
        logger.warning("LoRA results not found")
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    training_time = data['metadata']['training_time']
    
    return {
        "method": "LoRA Fine-tuned",
        "accuracy": data['test_results']['accuracy'],
        "f1": data['test_results']['f1'],
        "precision": data['test_results']['precision'],
        "recall": data['test_results']['recall'],
        "training_time": f"{training_time/60:.1f} min",
        "inference_time": "~0.05s",
        "model_size": "~5MB (adapter only)"
    }

def create_comparison():
    """Create comprehensive comparison"""
    
    logger.info("Loading results from all methods...")
    
    results = []
    
    # Load each method
    baseline = load_baseline_results()
    if baseline:
        results.append(baseline)
    
    prompting = load_prompting_results()
    if prompting:
        results.append(prompting)
    
    lora = load_lora_results()
    if lora:
        results.append(lora)
    
    if not results:
        logger.error("No results found! Run all experiments first.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Display comparison
    print("\n" + "="*80)
    print("METHOD COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save comparison
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "method_comparison.csv", index=False)
    
    with open(output_dir / "method_comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Comparison saved to {output_dir}")
    
    # Create visualizations
    create_visualizations(df)
    
    # Log to W&B
    log_to_wandb(df)
    
    return df

def create_visualizations(df):
    """Create comparison visualizations"""
    
    output_dir = Path("outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # 1. Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(df['method'], df['accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Comparison Across Methods', fontsize=14, fontweight='bold')
    ax.set_ylim([0.5, 1.0])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_comparison.png", dpi=300)
    plt.close()
    
    logger.info(f"📊 Saved accuracy comparison to {output_dir / 'accuracy_comparison.png'}")
    
    # 2. Metrics comparison (if F1 available)
    if df['f1'].notna().any():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['accuracy', 'precision', 'recall']
        titles = ['Accuracy', 'Precision', 'Recall']
        
        for ax, metric, title in zip(axes, metrics, titles):
            valid_df = df[df[metric].notna()]
            if not valid_df.empty:
                ax.bar(valid_df['method'], valid_df[metric])
                ax.set_title(title, fontweight='bold')
                ax.set_ylim([0.5, 1.0])
                ax.tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_comparison.png", dpi=300)
        plt.close()
        
        logger.info(f"📊 Saved metrics comparison to {output_dir / 'metrics_comparison.png'}")
    
    # 3. Best method summary
    best_idx = df['accuracy'].idxmax()
    best_method = df.iloc[best_idx]
    
    print(f"\n🏆 BEST METHOD: {best_method['method']}")
    print(f"   Accuracy: {best_method['accuracy']:.4f}")
    if best_method['f1']:
        print(f"   F1 Score: {best_method['f1']:.4f}")
    print(f"   Training Time: {best_method['training_time']}")
    print(f"   Inference Time: {best_method['inference_time']}")

def log_to_wandb(df):
    """Log comparison to W&B"""
    
    wandb.init(
        project="week4-nlp-llms",
        name="final_comparison",
        config={
            "comparison_type": "all_methods"
        }
    )
    
    # Log comparison table
    wandb.log({
        "method_comparison": wandb.Table(dataframe=df)
    })
    
    # Log metrics
    for _, row in df.iterrows():
        wandb.log({
            f"{row['method']}_accuracy": row['accuracy']
        })
    
    # Log images
    fig_dir = Path("outputs/figures")
    if (fig_dir / "accuracy_comparison.png").exists():
        wandb.log({
            "accuracy_comparison": wandb.Image(str(fig_dir / "accuracy_comparison.png"))
        })
    
    wandb.finish()
    logger.info("✅ Logged comparison to W&B")

if __name__ == "__main__":
    create_comparison()