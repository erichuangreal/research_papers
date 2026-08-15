import os
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from pathlib import Path

class RAGClass:
    def __init__(self, data_path) :
        self.data_path = data_path
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.embeddings = None
        self.result = None
        
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
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.text_chunks = text_splitter.split_documents(self.documents)
        print(f"Split documents into {len(self.text_chunks)} chunks.")
            
    def create_vectorstore(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = Chroma.from_documents(self.text_chunks, embedding=self.embeddings)
        print("Vectorstore created with embeddings.")

    def setup_retriever(self):
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        self.retriever = self.vectorstore.as_retriever()
        print("Retriever set up from vectorstore.")
        print("Retriever details:", self.retriever)
        return self.retriever
    
    def setup_qa_chain(self):
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")
        llm = ChatOpenAI(
            model_name="gpt-4",
            temperature=0)
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Answer the user's question using ONLY the retrieved context.

                Do not use prior knowledge.

                For every factual claim:
                - It must be supported by the retrieved context.
                - Cite the corresponding source metadata.

                If a claim cannot be supported by the retrieved context,
                do not include it.

                If the retrieved context is insufficient to answer the question,
                say that the retrieved papers do not contain enough information.

                Never invent sources, citations, authors, page numbers, or results.
                
                Retrieved context:
                {context}
                """
            ),
            ("human", "{input}")
        ])

        # QA chain is set up below to connect llm and retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        self.qa_chain = create_retrieval_chain(self.retriever, document_chain)
        print("QA chain set up.")
        return self.qa_chain
    
    def answer_query(self, query: str):
        if self.qa_chain is None:
            raise ValueError("QA chain not initialized.")
        
        
        response = self.qa_chain.invoke({"input": query})
        self.result = response["answer"]
        print("Query:", query, "\nAnswer:", self.result)
        return self.result

    # using cosine similarity to test system accuracy
    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)

        return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )
    
    def evaluate(self, queries: list, ground_truths: list):
        # Determines the system's accuracy with sample queries and ground truths
        if len(queries) != len(ground_truths):
            raise ValueError("Queries and ground truths must be of the same length.")
        
        total_sim = 0
        correct = 0
        for idx, (query, truth) in enumerate(zip(queries, ground_truths)):
            response = self.qa_chain.invoke({"input": query})
            answer = response["answer"]
            
            truth_embedding = self.embeddings.embed_query(truth)
            answer_embedding = self.embeddings.embed_query(answer)
            
            similarity = self.cosine_similarity(
                truth_embedding,
                answer_embedding
            )
            
            print(
            "Query:", idx + 1,
            "\nExpected:", truth,
            "\nModel Answer:", answer,
            "\nSimilarity:", similarity
            )
            total_sim += similarity
            
            threshold = 0.80
            if similarity >= threshold:
                correct += 1
        length = len(queries) 
        accuracy = correct / length
        avg_sim = total_sim / length
        print("Avg similiarity: ", f"{avg_sim * 100:.2f}%")
        print(correct, " / ", length, " passed\n", f"Accuracy: {accuracy * 100:.2f}%")
        return avg_sim
    