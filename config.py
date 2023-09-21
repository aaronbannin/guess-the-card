from os import getenv

from dotenv import load_dotenv

load_dotenv()


class Config:
    ARIZE_API_KEY = getenv("ARIZE_API_KEY")
    ARIZE_SPACE_KEY = getenv("ARIZE_SPACE_KEY")
    OPENAI_API_KEY = getenv("OPENAI_API_KEY")
    POSTGRES_DB = getenv("POSTGRES_DB")
    POSTGRES_HOST = getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = getenv("POSTGRES_PORT", 5432)
    POSTGRES_USER = getenv("POSTGRES_USER")
