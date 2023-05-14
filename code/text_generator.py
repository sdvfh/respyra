import openai
import pandas as pd
from utils import openai_api_key, path

openai.api_key = openai_api_key

incomplete_prompt = (
    "Please write a 250-word text as a psychologist with 20 years of experience in the field of mental "
    "disorders about possible symptoms and diagnoses of mental disorders regarding the user who wrote the post at "
    "the end of this text, which is enclosed in parentheses. Your text should also address whether there are any signs"
    'of ambiguity and/or sarcasm. "{sentence}"'
)

model = "gpt-3.5-turbo"

posts = pd.read_csv(path["data"] / "full.csv")
for i, line in posts.iterrows():
    if not isinstance(line["chatgpt"], str):
        prompt = incomplete_prompt.format(sentence=line["sentence"])
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        response = completion.choices[0].message.content
        posts.loc[i, "chatgpt"] = response
        posts.to_csv(path["data"] / "full.csv", index=False)
