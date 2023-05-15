import os
from pathlib import Path

openai_api_key = os.getenv("OPENAI_API_KEY")

path = {"root": Path(__file__).parent.parent.absolute()}
path["data"] = path["root"] / "data"
path["psysym"] = path["data"] / "PsySym"
