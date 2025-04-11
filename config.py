from dotenv import load_dotenv
import os

load_dotenv()

HUGGINGFACE_HUB_TOKEN: str = os.getenv("HUGGINGFACE_HUB_TOKEN")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")