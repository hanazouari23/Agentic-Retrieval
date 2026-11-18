import pandas as pd
from pathlib import Path

for alpha in ['alpha05', 'alpha10', 'alpha15']:
    path = Path(f'RQs/relevant_csvs/cql_full_tree_150_msmarco_traces_{alpha}.csv')
    src = Path(f'RQs/relevant_csvs/cql_full_tree_50_msmarco_traces_{alpha}.csv')
    df = pd.read_csv(path)
    src_df = pd.read_csv(src)
    ref = pd.read_csv('RQs/relevant_csvs/msmarco_dev_reference_success_150.csv')
    print(f'=== {alpha} ===')
    print(f'Rows: {len(df)}')
    print(f'Unique query_id: {df["query_id"].nunique()}')
    print(f'Source 50 unique query_id: {src_df["query_id"].nunique()}')
    print(f'First 5 unique IDs: {list(df["query_id"].unique()[:5])}')
    print(f'Last 5 unique IDs: {list(df["query_id"].unique()[-5:])}')
    missing = set(ref['query_id']) - set(df['query_id'])
    extra = set(df['query_id']) - set(ref['query_id'])
    print(f'Missing from ref 150: {len(missing)}')
    if missing:
        print(f'  Missing IDs: {sorted(missing)[:20]}')
    print(f'Extra beyond ref 150: {len(extra)}')
    print()
