"""
Simplified FastAPI application for sentiment analysis
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

app = FastAPI(title="Sentiment Analysis API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
models = {"baseline": None, "lora": None, "tokenizer": None}
device = "cuda" if torch.cuda.is_available() else "cpu"

# Request/Response models
class SentimentRequest(BaseModel):
    text: str
    model_type: str = "baseline"

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    # Load Baseline
    try:
        from src.models.baseline import BaselineClassifier
        baseline_path = Path("outputs/models/baseline_logistic")
        if baseline_path.exists():
            models["baseline"] = BaselineClassifier.load(baseline_path)
    except Exception as e:
        print(f"Baseline load failed: {e}")
    
    # Load LoRA
    try:
        lora_path = Path("outputs/models/lora_finetuned")
        if lora_path.exists():
            base = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=2
            )
            models["lora"] = PeftModel.from_pretrained(base, str(lora_path)).to(device)
            models["lora"].eval()
            models["tokenizer"] = AutoTokenizer.from_pretrained(str(lora_path))
    except Exception as e:
        print(f"LoRA load failed: {e}")

@app.get("/")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "baseline": models["baseline"] is not None,
        "lora": models["lora"] is not None,
        "device": device
    }

@app.post("/predict", response_model=SentimentResponse)
async def predict(request: SentimentRequest):
    """Predict sentiment"""
    
    if request.model_type == "baseline":
        if not models["baseline"]:
            raise HTTPException(503, "Baseline model not loaded")
        
        pred = models["baseline"].predict(pd.Series([request.text]))[0]
        proba = models["baseline"].predict_proba(pd.Series([request.text]))[0]
        
        return SentimentResponse(
            text=request.text[:100],
            sentiment="Positive" if pred == 1 else "Negative",
            confidence=float(max(proba))
        )
    
    elif request.model_type == "lora":
        if not models["lora"]:
            raise HTTPException(503, "LoRA model not loaded")
        
        inputs = models["tokenizer"](
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = models["lora"](**inputs)
        
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred = torch.argmax(probs).item()
        
        return SentimentResponse(
            text=request.text[:100],
            sentiment="Positive" if pred == 1 else "Negative",
            confidence=probs[pred].item()
        )
    
    raise HTTPException(400, f"Unknown model: {request.model_type}")

@app.post("/predict_batch")
async def predict_batch(texts: List[str], model_type: str = "baseline"):
    """Batch prediction"""
    if len(texts) > 100:
        raise HTTPException(400, "Max 100 texts")
    
    results = []
    for text in texts:
        try:
            result = await predict(SentimentRequest(text=text, model_type=model_type))
            results.append(result)
        except Exception as e:
            results.append({"text": text[:100], "error": str(e)})
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)