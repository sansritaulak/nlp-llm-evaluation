"""
BM25 Search Engine - Side Quest Implementation
Usage: python search.py "your query" --k 5
"""
import re
import json
import pickle
import sys
from pathlib import Path
from typing import List, Dict
from rank_bm25 import BM25Okapi

class BM25Search:
    def __init__(self, index_dir: str = "search_engine/index"):
        self.index_dir = Path(index_dir)
        self.bm25 = None
        self.documents = []
        self.titles = []
    
    def _preprocess(self, text: str) -> List[str]:
        """Convert to lowercase, split on whitespace, and remove punctuation/non-alphanumeric chars."""
        # This replaces all non-word characters (including punctuation) with a space, 
        # lowercases, and splits, ensuring tokens are clean.
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text) # Replace punctuation with space
        # Filter out empty strings that result from multiple spaces
        return [token for token in text.split() if token]
    
    def build_index(self, corpus: List[Dict]):
        """Build BM25 index from corpus"""
        print(f"Building index from {len(corpus)} documents...")
        
        self.documents = [doc['text'] for doc in corpus]
        self.titles = [doc.get('title', doc['text'][:50]) for doc in corpus]
        
        # Tokenize
        tokenized = [self._preprocess(doc) for doc in self.documents]
        
        # Build BM25
        self.bm25 = BM25Okapi(tokenized)
        
        print(f"✓ Index built: {len(self.documents)} documents")
    
    def save(self):
        """Save index"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.index_dir / "bm25.pkl", 'wb') as f:
            pickle.dump(self.bm25, f)
        
        with open(self.index_dir / "metadata.json", 'w') as f:
            json.dump({
                'documents': self.documents,
                'titles': self.titles
            }, f)
        
        print(f"✓ Index saved to {self.index_dir}")
    
    def load(self):
        """Load index"""
        with open(self.index_dir / "bm25.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)
        
        with open(self.index_dir / "metadata.json", 'r') as f:
            data = json.load(f)
            self.documents = data['documents']
            self.titles = data['titles']
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search and return top-k results"""
        query_tokens = self._preprocess(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        top_indices = scores.argsort()[-k:][::-1]
        
        results = []
        for rank, idx in enumerate(top_indices, 1):
            snippet = self.documents[idx][:200] + "..." if len(self.documents[idx]) > 200 else self.documents[idx]
            results.append({
                'rank': rank,
                'doc_id': idx,
                'score': float(scores[idx]),
                'title': self.titles[idx],
                'snippet': snippet
            })
        
        return results
    
    def print_results(self, query: str, results: List[Dict]):
        """Print search results"""
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results")
        print("=" * 80)
        
        for r in results:
            print(f"\n{r['rank']}. {r['title']}")
            print(f"   Score: {r['score']:.4f} | Doc ID: {r['doc_id']}")
            print(f"   {r['snippet']}")

if __name__ == "__main__":
    # CLI: python search.py "query" --k 5
    if len(sys.argv) < 2:
        print("Usage: python search.py 'query' --k 5")
        sys.exit(1)
    
    query = sys.argv[1]
    k = 5
    
    if '--k' in sys.argv:
        k = int(sys.argv[sys.argv.index('--k') + 1])
    
    # Load and search
    engine = BM25Search()
    engine.load()
    results = engine.search(query, k)
    engine.print_results(query, results)