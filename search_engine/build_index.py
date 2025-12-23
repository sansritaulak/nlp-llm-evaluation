"""
Build BM25 index from IMDB reviews
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import load_processed_data
from search import BM25Search

def build():
    """Build search index"""
    
    # Load IMDB data
    print("Loading IMDB dataset...")
    df = load_processed_data()
    
    # Use subset for demo (or all data)
    df = df.sample(n=10000, random_state=42)
    
    # Prepare corpus
    corpus = []
    for idx, row in df.iterrows():
        corpus.append({
            'text': row['text'],
            'title': f"Review {idx} ({'Pos' if row['label'] == 1 else 'Neg'})"
        })
    
    # Build index
    engine = BM25Search()
    engine.build_index(corpus)
    engine.save()
    
    print("\n✓ Index built successfully!")
    print(f"  Total documents: {len(corpus)}")
    print(f"  Index location: search_engine/index/")
    print("\nNext steps:")
    print("1. Edit search_engine/qrels.json with relevant doc IDs")
    print("2. Run: python search_engine/search.py 'your query' --k 5")
    print("3. Run: python search_engine/eval.py")

if __name__ == "__main__":
    build()