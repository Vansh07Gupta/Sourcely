import os

from dotenv import load_dotenv
from ollama import Client as OllamaClient

load_dotenv()

OLLAMA_ENDPOINT = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
inference_client = OllamaClient(host=OLLAMA_ENDPOINT)

NEO4J_DEFAULT_URI = os.environ.get('NEO4J_URI', 'bolt://127.0.0.1:7687')
NEO4J_DEFAULT_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_DEFAULT_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2')
