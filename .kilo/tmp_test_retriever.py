from pathlib import Path
import sys

root_path = Path('.').resolve()
src_path = root_path / 'src'
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

from src.utils.retriever import Retriever, create_retriever_callable

retriever = Retriever(index='msmarco-v2.1-segmented')

queries = [
    ('329114', 'how much to have your house reshingled'),
    ('622658', "what do cpa's earn"),
    ('612670', 'what county is sheridan'),
]

for qid, query in queries:
    print(f'=== {qid}: {query} ===')
    func = create_retriever_callable(retriever, adaptive_top_k=False)
    doc_ids, scores, corpus = func(query, 50)
    print(f'  non-adaptive size=50: {len(doc_ids)} docs')

    func_adaptive = create_retriever_callable(retriever, adaptive_top_k=True, min_top_k=1)
    doc_ids2, scores2, corpus2 = func_adaptive(query, 50)
    print(f'  adaptive fallback: {len(doc_ids2)} docs')
    print()
