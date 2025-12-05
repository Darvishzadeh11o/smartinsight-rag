import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Read OpenAI API Key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")

# Base folder of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data folders
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

# Chunking settings for later
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
