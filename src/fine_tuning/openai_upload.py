# script to fine tune openai models
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

from models.models import OpenAIModels


load_dotenv()

def main():
    cwd = Path.cwd()
    filepath = cwd/"src"/"fine_tuning"/"data"/"20231031-openai-tuning.jsonl"
    print(filepath.name)

    openai.api_key = os.getenv("OPENAI_API_KEY")
    file = openai.File.create(
        file=open(filepath, "rb"),
        purpose='fine-tune',
        user_provided_filename=filepath.name
    )

    print("file uploaded")
    print(file)

    ft = openai.FineTuningJob.create(training_file=file["id"], model="gpt-3.5-turbo")
    print("fine tuning job created")
    print(ft)

if __name__ == "__main__":
    main()

