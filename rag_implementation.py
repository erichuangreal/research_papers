from dotenv import load_dotenv, find_dotenv
from rag_class import RAGClass
import os

_env_path = find_dotenv(usecwd=True)
load_dotenv(_env_path, override=True)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or api_key.startswith("YOUR_") or api_key.strip() == "":
    raise RuntimeError(f"OPENAI_API_KEY missing or placeholder. Ensure a valid key is set in your .env (loaded from: {_env_path or 'not found'}).")
os.environ["OPENAI_API_KEY"] = api_key


rag = RAGClass("processed_text/")

# Load and process documents
rag.load_documents()
rag.split_documents()
rag.create_vectorstore()
rag.setup_retriever()
rag.setup_qa_chain()

# Answer a sample query
rag.answer_query("What is AI?")

'''
valuate the system with sample queries and ground truths
sample_queries = [""]
sample_ground_truths = [""]
rag.evaluate(sample_queries, sample_ground_truths)
'''
