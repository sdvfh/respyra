import time

import openai
import pandas as pd
from joblib import Parallel, delayed
from openai.error import OpenAIError, RateLimitError
from utils import openai_api_key, path


def make_completion(i, line):
    chatgpt_completion = path["data"] / "chatgpt_completions" / f"{i}.txt"
    if not chatgpt_completion.exists():
        prompt = incomplete_prompt.format(sentence=line["sentence"])
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                )
                break
            except (RateLimitError, OpenAIError):
                print("OpenAI error, waiting 10 seconds...")
                time.sleep(10)

        response = completion.choices[0].message.content
        chatgpt_completion.write_text(response)
        print(f"{i + 1}/{len(posts)} | {(i + 1) / len(posts) * 100:.2f}%")
    return


openai.api_key = openai_api_key

incomplete_prompt = (
    "Please write a 250-word text as a psychologist with 20 years of experience in the field of mental "
    "disorders about possible symptoms and diagnoses of mental disorders regarding the user who wrote the post at "
    "the end of this text, which is enclosed in parentheses. Your text should also address whether there are any signs"
    'of ambiguity and/or sarcasm. "{sentence}"'
)

model = "gpt-3.5-turbo"

posts = pd.read_csv(path["data"] / "full.csv")

Parallel(n_jobs=100)(delayed(make_completion)(i, line) for i, line in posts.iterrows())
