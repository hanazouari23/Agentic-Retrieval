import pandas as pd
from pathlib import Path

policy_files = {
    'CQL': '../results/cql_full_tree_traces_50_msmarco_dev.csv',
    'STOP': '../results/stop_traces_50_msmarco_dev.csv',
    'Rerank': '../results/rerank_traces_50_msmarco_dev.csv',
    'Reformulate': '../results/reformulate_traces_50_msmarco_dev.csv',
    'Expert': '../results/expert_traces_50_msmarco_dev.csv',
}
records = []
for policy_name, path in policy_files.items():
    p = Path(path)
    if not p.exists():
        print(f'{policy_name}: missing')
        continue
    df = pd.read_csv(p)
    if df.empty:
        print(f'{policy_name}: empty')
        continue
    grouped = df.groupby('query_id')
    per_query = pd.DataFrame({
        'cumulative_cost': grouped['cumulative_cost'].max(),
        'ndcg_final': grouped['ndcg_after'].last(),
        'recall_final': grouped['recall_after'].last(),
    })
    records.append({
        'policy': policy_name,
        'mean_cost': per_query['cumulative_cost'].mean(),
        'mean_ndcg': per_query['ndcg_final'].mean(),
        'mean_recall': per_query['recall_final'].mean(),
        'n_queries': len(per_query),
    })
print(pd.DataFrame(records))
