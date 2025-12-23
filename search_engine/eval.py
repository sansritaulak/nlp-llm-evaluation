"""
Search Engine Evaluation
Prints MRR@5 and nDCG@5
"""
import json
import numpy as np
from search import BM25Search

def calculate_mrr(relevance: list, k: int = 5) -> float:
    """Calculate Mean Reciprocal Rank"""
    for rank, rel in enumerate(relevance[:k], 1):
        if rel == 1:
            return 1.0 / rank
    return 0.0

def calculate_dcg(relevance: list, k: int = 5) -> float:
    """Calculate Discounted Cumulative Gain"""
    dcg = 0.0
    for i, rel in enumerate(relevance[:k], 1):
        dcg += rel / np.log2(i + 1)
    return dcg

def calculate_ndcg(relevance: list, k: int = 5) -> float:
    """Calculate Normalized DCG"""
    dcg = calculate_dcg(relevance, k)
    ideal = sorted(relevance, reverse=True)
    idcg = calculate_dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

def evaluate():
    """Main evaluation function"""
    
    # Load qrels
    with open("search_engine/qrels.json", 'r') as f:
        qrels = json.load(f)
    
    # Load search engine
    engine = BM25Search()
    engine.load()
    
    mrr_scores = []
    ndcg_scores = []
    hits_count = 0
    
    print("Evaluating search engine...")
    print("=" * 80)
    
    for query_id, query_data in qrels.items():
        query = query_data['query']
        relevant_docs = set(query_data['relevant_docs'])
        
        # Search
        results = engine.search(query, k=5)
        retrieved = [r['doc_id'] for r in results]
        
        # Create relevance list
        relevance = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved]
        
        # Calculate metrics
        mrr = calculate_mrr(relevance, k=5)
        ndcg = calculate_ndcg(relevance, k=5)
        
        mrr_scores.append(mrr)
        ndcg_scores.append(ndcg)
        
        if mrr > 0:
            hits_count += 1
        
        print(f"{query_id}: {query[:40]:40s} | MRR: {mrr:.3f} | nDCG: {ndcg:.3f}")
    
    # Calculate averages
    avg_mrr = np.mean(mrr_scores)
    avg_ndcg = np.mean(ndcg_scores)
    
    print("=" * 80)
    print(f"\nMRR@5:  {avg_mrr:.4f}")
    print(f"nDCG@5: {avg_ndcg:.4f}")
    print(f"Hits in top-5: {hits_count}/10")
    print()
    
    # Pass/Fail
    pass_mrr = avg_mrr >= 0.7
    pass_ndcg = avg_ndcg >= 0.6
    pass_hits = hits_count >= 7
    
    if pass_mrr:
        print("✓ PASS: MRR@5 >= 0.7")
    else:
        print(f"✗ FAIL: MRR@5 = {avg_mrr:.4f} < 0.7")
    
    if pass_ndcg:
        print("✓ PASS: nDCG@5 >= 0.6")
    else:
        print(f"✗ FAIL: nDCG@5 = {avg_ndcg:.4f} < 0.6")
    
    if pass_hits:
        print(f"✓ PASS: {hits_count}/10 queries have hits in top-5")
    else:
        print(f"✗ FAIL: Only {hits_count}/10 queries have hits")
    
    print()
    
    if pass_mrr and pass_ndcg and pass_hits:
        print("=" * 80)
        print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("=" * 80)
    
    return avg_mrr, avg_ndcg, hits_count

if __name__ == "__main__":
    evaluate()