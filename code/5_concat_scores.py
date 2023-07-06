import json

import pandas as pd
from utils import path

scores = (path["data"] / "scores").glob("*.json")
posts = pd.read_parquet(path["data"] / "full_with_suggestions.snappy.parquet")

for score in scores:
    raw_scores = score.read_text()
    index = int(score.stem)
    if raw_scores == "":
        posts = posts.drop(index)
    actual_score = json.loads(raw_scores)
    for key, value in actual_score.items():
        posts.loc[index, key] = value
posts.to_parquet(path["data"] / "full_with_suggestions_and_scores.snappy.parquet")
