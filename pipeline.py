"""
pipeline.py
-----------
Owner: Bhoomika Panday

What this file does:
    1. Takes a meme text string as input
    2. Embeds the text using all-MiniLM-L6-v2
    3. Searches ChromaDB for top-5 most similar memes
    4. Builds an augmented prompt with retrieved examples
    5. Sends prompt to Llama 3 running on GCP via Ollama
    6. Returns structured result with explanation + hate label + citations

Input:  meme text (string)
Output: dict with keys:
        - explanation   (str)
        - hate_label    (str: "hateful" or "not hateful")
        - reasoning     (str)
        - citations     (list of dicts with text + url + dataset)

How to run:
    from pipeline import analyze_meme
    result = analyze_meme("nobody: literally nobody: me at 3am eating cereal")
"""
