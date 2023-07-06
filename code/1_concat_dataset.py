import pandas as pd
from utils import path

train = pd.read_csv(path["psysym"] / "data" / "symp_data_w_control" / "train.csv")
valid = pd.read_csv(path["psysym"] / "data" / "symp_data_w_control" / "val.csv")
test = pd.read_csv(path["psysym"] / "data" / "symp_data_w_control" / "test.csv")

train["type"] = "train"
valid["type"] = "valid"
test["type"] = "test"

posts = pd.concat([train, valid, test])

posts = posts.sample(frac=1, random_state=42).reset_index(drop=True)

posts.to_parquet(path["data"] / "full.snappy.parquet")

(path["data"] / "chatgpt_completions").mkdir(exist_ok=True)
