import os
from pathlib import Path

openai_api_key = os.getenv("OPENAI_API_KEY")

path = {"root": Path(__file__).parent.parent.absolute()}
path["data"] = Path("/mnt/f/respyra")
path["psysym"] = path["root"] / "data" / "PsySym"
