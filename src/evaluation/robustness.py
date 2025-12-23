"""
Robustness evaluation for sentiment models
"""
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from loguru import logger
import wandb
from sklearn.metrics import accuracy_score, classification_report

class RobustnessEvaluator:
    """Evaluate model robustness on stress tests"""
    
    def __init__(self):
        self.results = {}
    
    def load_stress_tests(self, test_type: str = "all") -> List[Dict]:
        """Load stress test dataset"""
        
        if test_type == "all":
            path = Path("data/stress_tests/all_stress_tests.json")
        else:
            path = Path(f"data/stress_tests/{test_type}_test.json")
        
        if not path.exists():
            raise FileNotFoundError(f"Stress tests not found at {path}")
        
        with open(path, 'r') as f:
            tests = json.load(f)
        
        logger.info(f"Loaded {len(tests)} stress tests from {path}")
        return tests
    
    def evaluate_baseline(self, tests: List[Dict]) -> Dict[str, Any]:
        """Evaluate baseline model on stress tests"""
        
        from src.models.baseline import BaselineClassifier
        
        logger.info("Evaluating baseline model...")
        
        # Load model
        model_path = Path("outputs/models/baseline_logistic")
        if not model_path.exists():
            logger.warning("Baseline model not found, skipping")
            return None
        
        model = BaselineClassifier.load(model_path)
        
        # Extract texts and labels
        texts = pd.Series([t['text'] for t in tests])
        true_labels = [t['label'] for t in tests]
        types = [t['type'] for t in tests]
        
        # Predict
        predictions = model.predict(texts)
        
        # Calculate overall accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Calculate accuracy by type
        type_accuracies = {}
        for test_type in set(types):
            type_mask = [t == test_type for t in types]
            type_true = [l for l, m in zip(true_labels, type_mask) if m]
            type_pred = [p for p, m in zip(predictions, type_mask) if m]
            type_accuracies[test_type] = accuracy_score(type_true, type_pred)
        
        # Find failures
        failures = []
        for test, pred, true in zip(tests, predictions, true_labels):
            if pred != true:
                failures.append({
                    "text": test['text'],
                    "predicted": int(pred),
                    "true": true,
                    "type": test['type']
                })
        
        results = {
            "model": "baseline",
            "overall_accuracy": float(accuracy),
            "total_tests": len(tests),
            "correct": int((predictions == true_labels).sum()),
            "incorrect": int((predictions != true_labels).sum()),
            "accuracy_by_type": {k: float(v) for k, v in type_accuracies.items()},
            "failures": failures
        }
        
        logger.info(f"Baseline accuracy on stress tests: {accuracy:.3f}")
        
        return results
    
    def evaluate_lora(self, tests: List[Dict]) -> Dict[str, Any]:
        """Evaluate LoRA model on stress tests"""
        
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        import torch
        
        logger.info("Evaluating LoRA model...")
        
        # Load model
        model_path = Path("outputs/models/lora_finetuned")
        if not model_path.exists():
            logger.warning("LoRA model not found, skipping")
            return None
        
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=2
            )
            model = PeftModel.from_pretrained(base_model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load LoRA model: {e}")
            return None
        
        # Predict
        predictions = []
        for test in tests:
            inputs = tokenizer(
                test['text'],
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
        
        true_labels = [t['label'] for t in tests]
        types = [t['type'] for t in tests]
        
        # Calculate overall accuracy
        accuracy = accuracy_score(true_labels, predictions)
        
        # Calculate accuracy by type
        type_accuracies = {}
        for test_type in set(types):
            type_mask = [t == test_type for t in types]
            type_true = [l for l, m in zip(true_labels, type_mask) if m]
            type_pred = [p for p, m in zip(predictions, type_mask) if m]
            type_accuracies[test_type] = accuracy_score(type_true, type_pred)
        
        # Find failures
        failures = []
        for test, pred, true in zip(tests, predictions, true_labels):
            if pred != true:
                failures.append({
                    "text": test['text'],
                    "predicted": int(pred),
                    "true": true,
                    "type": test['type']
                })
        
        results = {
            "model": "lora",
            "overall_accuracy": float(accuracy),
            "total_tests": len(tests),
            "correct": int(sum([p == t for p, t in zip(predictions, true_labels)])),
            "incorrect": int(sum([p != t for p, t in zip(predictions, true_labels)])),
            "accuracy_by_type": {k: float(v) for k, v in type_accuracies.items()},
            "failures": failures
        }
        
        logger.info(f"LoRA accuracy on stress tests: {accuracy:.3f}")
        
        return results
    
    def compare_models(self, test_type: str = "all") -> pd.DataFrame:
        """Compare all models on stress tests"""
        
        # Load tests
        tests = self.load_stress_tests(test_type)
        
        # Evaluate each model
        baseline_results = self.evaluate_baseline(tests)
        lora_results = self.evaluate_lora(tests)
        
        # Compile results
        all_results = []
        
        if baseline_results:
            all_results.append(baseline_results)
            self.results['baseline'] = baseline_results
        
        if lora_results:
            all_results.append(lora_results)
            self.results['lora'] = lora_results
        
        # Create comparison DataFrame
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                "Model": result['model'],
                "Overall Accuracy": result['overall_accuracy'],
                "Correct": result['correct'],
                "Incorrect": result['incorrect']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Print comparison
        print("\n" + "="*80)
        print(f"ROBUSTNESS EVALUATION: {test_type.upper()} TESTS")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        # Print accuracy by type for each model
        for result in all_results:
            print(f"\n{result['model'].upper()} - Accuracy by Type:")
            print("-"*60)
            for test_type, acc in sorted(result['accuracy_by_type'].items(), key=lambda x: x[1]):
                print(f"  {test_type:30s}: {acc:.3f}")
        
        return comparison_df
    
    def generate_report(self) -> str:
        """Generate detailed robustness report"""
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ROBUSTNESS & FAILURE ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        for model_name, results in self.results.items():
            report_lines.append(f"\n## {model_name.upper()} MODEL")
            report_lines.append("-"*80)
            report_lines.append(f"Overall Accuracy: {results['overall_accuracy']:.3f}")
            report_lines.append(f"Total Tests: {results['total_tests']}")
            report_lines.append(f"Correct: {results['correct']}")
            report_lines.append(f"Incorrect: {results['incorrect']}")
            report_lines.append("")
            
            # Worst performing categories
            report_lines.append("### Worst Performing Categories:")
            sorted_types = sorted(
                results['accuracy_by_type'].items(),
                key=lambda x: x[1]
            )
            for test_type, acc in sorted_types[:5]:
                report_lines.append(f"  - {test_type}: {acc:.3f}")
            report_lines.append("")
            
            # Sample failures
            report_lines.append("### Sample Failures (showing first 10):")
            for i, failure in enumerate(results['failures'][:10], 1):
                report_lines.append(f"\n  {i}. Type: {failure['type']}")
                report_lines.append(f"     Text: {failure['text'][:100]}...")
                report_lines.append(f"     Predicted: {failure['predicted']}, True: {failure['true']}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        report = "\n".join(report_lines)
        
        # Save report
        output_dir = Path("outputs/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "robustness_report.txt", 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Report saved to {output_dir / 'robustness_report.txt'}")
        
        return report

def run_robustness_evaluation():
    """Main robustness evaluation pipeline"""
    
    # Initialize evaluator
    evaluator = RobustnessEvaluator()
    
    # Initialize W&B
    wandb.init(
        project="week4-nlp-llms",
        name="robustness_evaluation",
        config={"evaluation_type": "stress_tests"}
    )
    
    # Evaluate on each test category
    test_categories = ["negation", "sarcasm", "ood", "all"]
    
    all_comparisons = {}
    
    for category in test_categories:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating on {category} tests")
        logger.info(f"{'='*60}")
        
        try:
            comparison = evaluator.compare_models(category)
            all_comparisons[category] = comparison
            
            # Log to W&B
            wandb.log({
                f"{category}_comparison": wandb.Table(dataframe=comparison)
            })
            
        except Exception as e:
            logger.error(f"Failed to evaluate {category}: {e}")
    
    # Generate comprehensive report
    report = evaluator.generate_report()
    print(report)
    
    # Save all results
    output_dir = Path("outputs/results")
    with open(output_dir / "robustness_results.json", 'w') as f:
        json.dump(evaluator.results, f, indent=2)
    
    wandb.finish()
    
    logger.info("\n✅ Robustness evaluation complete!")

if __name__ == "__main__":
    wandb.login()
    run_robustness_evaluation()