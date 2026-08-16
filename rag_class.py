import os
import numpy as np
import json
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableLambda
from pathlib import Path

class RAGClass:
    def __init__(self, data_path) :
        self.data_path = Path(data_path)
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.embeddings = None
        self.result = None
        
        self.paper_metadata = {}
        
    def load_documents(self):
        self.documents = []
        self.paper_metadata = {}
        metadata_files = list(self.data_path.rglob("metadata.json"))
        
        if not metadata_files:
            raise FileNotFoundError(
            f"No metadata.json files found under: {self.data_path.resolve()}"
            )
        
        for metadata_path in metadata_files:
            batch_folder = metadata_path.parent
            # Load metadata for this batch
            with open(metadata_path, "r", encoding="utf-8") as f:
                batch_metadata = json.load(f)
            metadata_lookup = {}

            for metadata in batch_metadata:
                paper_id = Path(metadata["local_pdf_path"]).stem
                metadata_lookup[paper_id] = metadata
            for txt_file in batch_folder.glob("*.txt"):
                paper_id = txt_file.stem
            
                loader = TextLoader(str(txt_file), encoding="utf-8")
                documents = loader.load()

                # Store full metadata ONCE, will access after relevant chunks are found
                if paper_id in metadata_lookup:
                    self.paper_metadata[paper_id] = metadata_lookup[paper_id]

                # Only attach lightweight ID to document
                for document in documents:
                    document.metadata = {
                        "paper_id": paper_id
                    }

                self.documents.extend(documents)
        print(f"Loaded {len(self.documents)} documents.")
        print(f"Loaded metadata for {len(self.paper_metadata)} papers.")

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
        base_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        
        def retrieve_with_metadata(inputs):
            query = inputs["input"]

            docs = base_retriever.invoke(query)

            for doc in docs:
                paper_id = doc.metadata["paper_id"]

                metadata = self.paper_metadata.get(
                    paper_id,
                    {}
                )

                doc.metadata["title"] = metadata.get(
                    "title",
                    "Unknown"
                )

                doc.metadata["arxiv_id"] = metadata.get(
                    "arxiv_id",
                    "Unknown"
                )

                authors = metadata.get("authors", [])

                doc.metadata["authors"] = ", ".join(authors)

            return docs
        self.retriever = RunnableLambda(retrieve_with_metadata)
        print("Retriever set up from vectorstore.")
        print("Retriever details:", base_retriever)
        return self.retriever
    
    def setup_qa_chain(self):
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")
        llm = ChatOpenAI(
            model="gpt-5-nano"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                Answer the user's question using ONLY the retrieved context.

                Do not use prior knowledge.

                For every factual claim:
                - It must be supported by the retrieved context.
                - Cite the corresponding source metadata with page numbers.

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
        
        document_prompt = PromptTemplate.from_template(
            """
            Paper: {title}
            Authors: {authors}
            arXiv ID: {arxiv_id}

            Content:
            {page_content}
            """
        )

        # QA chain is set up below to connect llm and retriever
        document_chain = create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt)
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
    