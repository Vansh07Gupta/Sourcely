import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

OLLAMA_ENDPOINT = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

NEO4J_DEFAULT_URI = os.environ.get('NEO4J_URI', 'bolt://127.0.0.1:7687')
NEO4J_DEFAULT_USER = os.environ.get('NEO4J_USER', 'neo4j')
NEO4J_DEFAULT_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')

DEFAULT_MODEL = os.environ.get('OLLAMA_MODEL', 'llama3.2')
EMBEDDING_MODEL = os.environ.get('OLLAMA_EMBED_MODEL', 'nomic-embed-text')


def get_chat_model(model_name: str, json_mode: bool = False) -> ChatOllama:
    kwargs = {"format": "json"} if json_mode else {}
    return ChatOllama(base_url=OLLAMA_ENDPOINT, model=model_name, **kwargs)


embedding_model = OllamaEmbeddings(base_url=OLLAMA_ENDPOINT, model=EMBEDDING_MODEL)
