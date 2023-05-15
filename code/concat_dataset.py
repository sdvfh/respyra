import pandas as pd
from utils import path

train_1 = pd.read_csv(path["psysym"] / "data" / "symp_data" / "train.csv")
valid_1 = pd.read_csv(path["psysym"] / "data" / "symp_data" / "val.csv")
test_1 = pd.read_csv(path["psysym"] / "data" / "symp_data" / "test.csv")

train_0 = pd.read_csv(path["psysym"] / "data" / "symp_data_w_control" / "train.csv")
valid_0 = pd.read_csv(path["psysym"] / "data" / "symp_data_w_control" / "val.csv")
test_0 = pd.read_csv(path["psysym"] / "data" / "symp_data_w_control" / "test.csv")

train_1["type"] = "train"
valid_1["type"] = "valid"
test_1["type"] = "test"

train_0["type"] = "train"
valid_0["type"] = "valid"
test_0["type"] = "test"

pos_label = pd.concat([train_1, valid_1, test_1])
neg_label = pd.concat([train_0, valid_0, test_0])

pos_label["classification"] = 1
neg_label["classification"] = 0

posts = pd.concat([pos_label, neg_label])
posts = posts.sample(frac=1, random_state=42).reset_index(drop=True)

posts.to_csv(path["data"] / "full.csv", index=False)

(path["data"] / "chatgpt_completions").mkdir(exist_ok=True)
