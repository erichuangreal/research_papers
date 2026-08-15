import os
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from pathlib import Path

class RAGClass:
    def __init__(self, data_path) :
        self.data_path = data_path
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        
    def load_documents(self):
        self.documents = []
        txt_files = Path(self.data_path).rglob("*.txt")

        for txt_file in txt_files:
            loader = TextLoader(str(txt_file), encoding="utf-8")
            documents = loader.load()

            # Keep track of which paper each document came from
            for document in documents:
                document.metadata["source"] = str(txt_file)
                document.metadata["paper_name"] = txt_file.stem

            self.documents.extend(documents)
            print(f"Loaded {len(self.documents)} documents.")

        return self.documents
    
    def split_documents(self, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split documents into {len(self.text_chunks)} chunks.")
            
    def create_vectorstore(self):
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma.from_documents(self.text_chunks, embedding=embeddings)

    def setup_retriever(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        self.retriever = self.vectorstore.as_retriever()
        print("Retriever set up from vectorstore.")
        display(HTML(f"<b>Retriever details:</b> {self.retriever}"))
        return self.retriever
    
    def setup_qa_chain(self):
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever)
        print("QA chain initialized with LLM and retriever.")
        display(HTML(f"<b>QA chain details:</b> {self.qa_chain}"))
        return self.qa_chain
    
        def answer_query(self, query: str):
            """
            Answers a query using the QA chain.
            Returns the answer string.
            """
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        result = self.qa_chain.run(query)
        display(HTML(f"Query: {query}Answer: {result}"))
        return result

    def evaluate(self, queries: list, ground_truths: list):
        """
        Evaluates the QA system using a list of queries and ground truths.
        Returns the accuracy as a float.
        """
        if len(queries) != len(ground_truths):
            raise ValueError("Queries and ground truths must be of the same length.")
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        correct = 0
        for idx, (query, truth) in enumerate(zip(queries, ground_truths)):
            answer = self.qa_chain.run(query)
            display(HTML(f"Query {idx+1}: {query}Expected: {truth}Model Answer: {answer}"))
            if truth.lower() in answer.lower():
                correct += 1
        accuracy = correct / len(queries)
        display(HTML(f"Evaluation Accuracy: {accuracy * 100:.2f}%"))
        return accuracy
    