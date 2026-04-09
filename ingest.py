"""
ingest.py
---------
Owner: TBD

What this file does:
    1. Loads the Facebook Hateful Memes dataset (train.jsonl)
    2. Cleans the meme text (lowercase, remove noise, handle emojis)
    3. Generates sentence embeddings using all-MiniLM-L6-v2
    4. Stores everything into ChromaDB vector database

Input:  data/train.jsonl
Output: ChromaDB collection with 8,500+ embedded meme entries

How to run:
    python ingest.py

Note: Run this once before starting the app.
      Takes ~10-15 minutes on first run.
"""
