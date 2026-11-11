"""Standalone ChatNoir retrieval test (avoids src import dependencies)."""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("CHATNOIR_API_URL", "https://www.chatnoir.eu/api/v1/_search").strip()
API_KEY = os.getenv("CHATNOIR_API_KEY", "").strip()

queries = [
    "what was the reason for the snake war",
    "what does cooper mean in england",
    "what is va home loan coe",
    "commercial roaming definition",
    "what was the result of the battle of tippecanoe?",
]

configs = [
    ("msmarco-v2.1-segmented", 50),
    ("msmarco-v2.1-segmented", 20),
    ("msmarco-v2.1", 50),
    ("msmarco-v2.1", 20),
]

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

for index, top_k in configs:
    print(f"\n=== index={index}, top_k={top_k} ===")
    for q in queries:
        payload = {"query": q, "index": [index], "size": top_k, "search_method": "bm25"}
        try:
            t0 = time.time()
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            elapsed = (time.time() - t0) * 1000
            if resp.status_code == 200:
                n = len(resp.json().get("results", []))
                print(f"  OK   | {n:3d} docs | {elapsed:6.1f}ms | {q[:60]}")
            else:
                print(f"  FAIL | HTTP {resp.status_code} | {resp.text[:80]:80s} | {q[:60]}")
        except Exception as e:
            print(f"  FAIL | {str(e)[:80]:80s} | {q[:60]}")
        time.sleep(0.5)
