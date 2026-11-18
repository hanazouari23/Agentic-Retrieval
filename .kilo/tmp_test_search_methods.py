from pathlib import Path
import sys

root_path = Path('.').resolve()
src_path = root_path / 'src'
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

from src.utils.retriever import Retriever

queries = [
    'how much to have your house reshingled',
    "what do cpa's earn",
    'what county is sheridan',
]

retriever = Retriever(index='msmarco-v2.1-segmented')
for method in ['bm25', 'qld', 'tfidf']:
    print(f'=== Method: {method} ===')
    for query in queries:
        try:
            results = retriever.retrieve(query, top_k=50, search_method=method)
            print(f'  {query[:40]}: {len(results)} docs')
        except Exception as e:
            print(f'  {query[:40]}: ERROR {e}')
    print()
