"""
Load and process IMDB dataset
"""
import pandas as pd
from pathlib import Path
import re
from typing import Optional
from loguru import logger

def clean_text(text: str) -> str:
    """Clean HTML and normalize text"""
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = ' '.join(text.split())  # Normalize whitespace
    return text

def load_imdb_raw(data_dir: Path = Path("data/raw/aclImdb")) -> pd.DataFrame:
    """Load raw IMDB dataset"""
    
    logger.info(f"Loading IMDB dataset from {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found. Run scripts/download_data.py first.")
    
    data = []
    
    # Load train and test splits
    for split in ["train", "test"]:
        for sentiment, label in [("pos", 1), ("neg", 0)]:
            folder = data_dir / split / sentiment
            for file in folder.glob("*.txt"):
                with open(file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    data.append({
                        "text": clean_text(text),
                        "label": label,
                        "split": split
                    })
    
    df = pd.DataFrame(data)
    logger.info(f"Loaded {len(df):,} reviews")
    return df

def create_validation_split(df: pd.DataFrame, val_size: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Create validation split from training data"""
    
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Sample validation from train
    val_df = train_df.sample(frac=val_size, random_state=seed)
    train_df = train_df.drop(val_df.index)
    
    # Update labels
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    logger.info(f"Splits - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    return final_df

def save_processed_data(df: pd.DataFrame):
    """Save processed dataset"""
    
    # Save full dataset
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "imdb_processed.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"💾 Saved to {output_path}")
    
    # Save splits
    splits_dir = Path("data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        split_path = splits_dir / f"{split}.parquet"
        split_df.to_parquet(split_path, index=False)
        logger.info(f"💾 Saved {split}: {len(split_df):,} samples")

def load_processed_data(split: Optional[str] = None) -> pd.DataFrame:
    """Load processed dataset"""
    if split:
        path = Path(f"data/splits/{split}.parquet")
    else:
        path = Path("data/processed/imdb_processed.parquet")
    
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}. Run data processing first.")
    
    return pd.read_parquet(path)

if __name__ == "__main__":
    # Process data
    df = load_imdb_raw()
    df = create_validation_split(df, val_size=0.1, seed=42)
    save_processed_data(df)
    
    # Test loading
    train = load_processed_data('train')
    print(f"\n✅ Data processing complete!")
    print(f"Sample review: {train.iloc[0]['text'][:200]}...")
    print(f"Label: {'Positive' if train.iloc[0]['label'] == 1 else 'Negative'}")