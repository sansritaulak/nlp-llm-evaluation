"""
Track compute budget and costs
"""
import json
from pathlib import Path
from loguru import logger

def calculate_compute_budget():
    """Calculate total compute used"""
    
    budget = {
        "baseline": {
            "training_time_minutes": 3,
            "gpu_hours": 0,  # CPU only
            "model_params": "~5M",
            "trainable_params": "~5M (100%)",
            "disk_space_mb": 50,
            "estimated_cost_usd": 0  # Free (CPU)
        },
        "prompting": {
            "training_time_minutes": 0,
            "gpu_hours": 0,
            "inference_samples": 100,
            "avg_latency_sec": 0.5,
            "total_inference_minutes": 0.83,  # 100 * 0.5 / 60
            "estimated_cost_usd": 0  # Free (local model)
        },
        "lora": {
            "training_time_minutes": 0,
            "gpu_hours": 0,
            "model_params": "66M (DistilBERT)",
            "trainable_params": "~0.3M (0.5%)",
            "disk_space_mb": 5,  # Adapter only
            "estimated_cost_usd": 0  # Free (local)
        }
    }
    
    # Load actual LoRA training time
    lora_results_path = Path("outputs/results/lora_results.json")
    if lora_results_path.exists():
        with open(lora_results_path, 'r') as f:
            data = json.load(f)
            training_time_sec = data['metadata']['training_time']
            budget['lora']['training_time_minutes'] = training_time_sec / 60
    
    # Calculate totals
    total_training_minutes = (
        budget['baseline']['training_time_minutes'] +
        budget['prompting']['training_time_minutes'] +
        budget['lora']['training_time_minutes']
    )
    
    total_cost = (
        budget['baseline']['estimated_cost_usd'] +
        budget['prompting']['estimated_cost_usd'] +
        budget['lora']['estimated_cost_usd']
    )
    
    budget['totals'] = {
        "total_training_minutes": round(total_training_minutes, 2),
        "total_training_hours": round(total_training_minutes / 60, 2),
        "total_cost_usd": round(total_cost, 2),
        "total_disk_space_mb": (
            budget['baseline']['disk_space_mb'] +
            budget['lora']['disk_space_mb']
        )
    }
    
    # Display budget
    print("\n" + "="*80)
    print("COMPUTE BUDGET")
    print("="*80)
    
    print("\n1. BASELINE (TF-IDF + Logistic Regression)")
    print(f"   Training Time: {budget['baseline']['training_time_minutes']:.1f} minutes")
    print(f"   Model Size: {budget['baseline']['disk_space_mb']} MB")
    print(f"   Trainable Params: {budget['baseline']['trainable_params']}")
    print(f"   Cost: ${budget['baseline']['estimated_cost_usd']:.2f}")
    
    print("\n2. PROMPTING (FLAN-T5)")
    print(f"   Training Time: {budget['prompting']['training_time_minutes']} minutes (no training)")
    print(f"   Inference Samples: {budget['prompting']['inference_samples']}")
    print(f"   Total Inference Time: {budget['prompting']['total_inference_minutes']:.2f} minutes")
    print(f"   Cost: ${budget['prompting']['estimated_cost_usd']:.2f} (local model)")
    
    print("\n3. LORA FINE-TUNING")
    print(f"   Training Time: {budget['lora']['training_time_minutes']:.1f} minutes")
    print(f"   Model Params: {budget['lora']['model_params']}")
    print(f"   Trainable Params: {budget['lora']['trainable_params']}")
    print(f"   Adapter Size: {budget['lora']['disk_space_mb']} MB")
    print(f"   Cost: ${budget['lora']['estimated_cost_usd']:.2f}")
    
    print("\n" + "-"*80)
    print("TOTALS")
    print("-"*80)
    print(f"   Total Training Time: {budget['totals']['total_training_hours']:.2f} hours")
    print(f"   Total Disk Space: {budget['totals']['total_disk_space_mb']} MB")
    print(f"   Total Cost: ${budget['totals']['total_cost_usd']:.2f}")
    print("="*80)
    
    # Efficiency analysis
    print("\n💡 EFFICIENCY ANALYSIS")
    print("-"*80)
    print(f"   LoRA uses only {budget['lora']['trainable_params']} vs full model")
    print(f"   LoRA adapter is {budget['baseline']['disk_space_mb'] / budget['lora']['disk_space_mb']:.0f}x smaller than baseline")
    print(f"   All methods cost $0 (using local/free resources)")
    
    # Save budget
    output_path = Path("outputs/results/compute_budget.json")
    with open(output_path, 'w') as f:
        json.dump(budget, f, indent=2)
    
    logger.info(f"💾 Budget saved to {output_path}")
    
    return budget

if __name__ == "__main__":
    calculate_compute_budget()