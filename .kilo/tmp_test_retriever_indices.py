from pathlib import Path
import sys

root_path = Path('.').resolve()
src_path = root_path / 'src'
sys.path.insert(0, str(root_path))
sys.path.insert(0, str(src_path))

from src.utils.retriever import Retriever, create_retriever_callable

queries = [
    ('329114', 'how much to have your house reshingled'),
    ('622658', "what do cpa's earn"),
    ('612670', 'what county is sheridan'),
]

for index in ['msmarco-v2.1-segmented', 'msmarco-v2.1']:
    print(f'=== Index: {index} ===')
    retriever = Retriever(index=index)
    func = create_retriever_callable(retriever, adaptive_top_k=False)
    for qid, query in queries:
        doc_ids, scores, corpus = func(query, 50)
        print(f'  {qid}: {len(doc_ids)} docs')
    print()
