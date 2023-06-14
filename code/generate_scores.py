import os
import random

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import path


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


seed_everything(666)

posts = pd.read_parquet(
    path["data"] / "full_with_suggestions_with_scores.snappy.parquet"
)
posts = posts[posts["score_completion_anxious mood"].isnull()]

symptoms = [
    "anxious mood",
    "autonomic symptoms",
    "cardiovascular symptoms",
    "catatonic behavior",
    "decreased energy tiredness fatigue",
    "depressed mood",
    "gastrointestinal symptoms",
    "genitourinary symptoms",
    "hyperactivity agitation",
    "impulsivity",
    "inattention",
    "indecisiveness",
    "respiratory symptoms",
    "suicidal ideas",
    "worthlessness and guilty",
    "avoidance of stimuli",
    "compensatory behaviors to prevent weight gain",
    "compulsions",
    "diminished emotional expression",
    "do things easily get painful consequences",
    "drastical shift in mood and energy",
    "fear about social situations",
    "fear of gaining weight",
    "fears of being negatively evaluated",
    "flight of ideas",
    "intrusion symptoms",
    "loss of interest or motivation",
    "more talktive",
    "obsession",
    "panic fear",
    "pessimism",
    "poor memory",
    "sleep disturbance",
    "somatic muscle",
    "somatic symptoms others",
    "somatic symptoms sensory",
    "weight and appetite change",
    "anger irritability",
]

other_analyses = ["sarcasm", "ambiguity"]

all_hypothesis_psychologist = [
    [
        'This is an example of a psychologist describing their patient with "{label}" symptom',
        symptoms,
    ],
    [
        'This is an example of a psychologist discussing whether the text provided by their patient exhibits "{label}"',
        other_analyses,
    ],
]

all_hypothesis_post = [
    ['This is an example of a post of a user with "{label}" symptom', symptoms],
    ['This is an example of a post that exhibits "{label}"', other_analyses],
]

all_hypothesis = [
    [all_hypothesis_psychologist, "completion"],
    [all_hypothesis_post, "sentence"],
]

nli_model = AutoModelForSequenceClassification.from_pretrained(
    "facebook/bart-large-mnli", device="cuda"
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli", device="cuda")

for i, line in posts.iterrows():
    scores_path = path["data"] / "scores" / f"{i}.json"
    if scores_path.exists():
        continue

    scores = pd.Series(name=i, dtype=float)
    for actual_hypothesis, name_column in all_hypothesis:
        text_to_classify = line[name_column]
        for hypothesis_template, candidate_labels in actual_hypothesis:
            for label in candidate_labels:
                hypothesis = hypothesis_template.format(label=label)

                try:
                    x = tokenizer.encode(
                        text_to_classify,
                        hypothesis,
                        return_tensors="pt",
                        truncation="only_first",
                    ).to("cuda")
                except TypeError:
                    scores_path.touch()
                    print(f"Error in {i}")
                    break

                logits = nli_model(x)[0]

                entail_contradiction_logits = logits[:, [0, 2]]
                probs = entail_contradiction_logits.softmax(dim=1)
                prob_label_is_true = probs[:, 1].item()
                scores[f"score_{name_column}_{label}"] = prob_label_is_true
    scores.to_json(scores_path)
    print(f"Finished {i} out of {len(posts)}")
