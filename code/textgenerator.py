import openai
import pandas

api_key = "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"


class SymptomGenerator:
    def __init__(self, model_engine="gpt-3.5-turbo"):
        self.api_key = api_key
        openai.api_key = self.api_key
        self.model_engine = model_engine
        print(f"model = {self.model_engine}")
        self.incomplete_prompt = (
            "Please write a 250-word text as a psychologist with 20 years of experience in the field of mental disorders "
            "about possible symptoms and diagnoses of mental disorders regarding the user who wrote the post at the end of "
            "this text, which is enclosed in parentheses. "
            "Your text should also address whether there are any signs of ambiguity and/or sarcasm. ({})"
        )

    def complement_prompt(self, sentence: str):
        return self.incomplete_prompt.format(sentence)

    def generate_symptoms(self, sentence: str):
        completion = openai.ChatCompletion.create(
            model=self.model_engine,
            messages=[{"role": "user", "content": self.complement_prompt(sentence)}],
            temperature=0.1,
        )

        response = completion.choices[0].message
        print(f"\n\nPrompt = {self.complement_prompt(sentence)}")
        print(f'\nResponse : {response["content"]}\n\n')
        return response["content"], None


if __name__ == "__main__":
    from constants import FULL_CSV_PATH, GENERATE_CSV_PATH

    pandas.set_option("display.max_columns", None)
    data_raw = pandas.read_csv(FULL_CSV_PATH)
    print(data_raw.head())
    data_copy = data_raw.copy()
    symptom_generator = SymptomGenerator()
    data_copy["Generate_Text"] = data_copy["sentence"].apply(
        symptom_generator.generate_symptoms
    )
    data_copy.to_csv(GENERATE_CSV_PATH)
