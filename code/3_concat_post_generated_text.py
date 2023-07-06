import pandas as pd
from utils import path

sentences = (path["data"] / "chatgpt_completions").glob("*.txt")
posts = pd.read_parquet(path["data"] / "full.snappy.parquet")

for sentence in sentences:
    index = int(sentence.stem)
    posts.loc[index, "completion"] = sentence.read_text()

posts.to_parquet(path["data"] / "full_with_suggestions.snappy.parquet")
