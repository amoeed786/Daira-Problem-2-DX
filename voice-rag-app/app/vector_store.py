# app/vector_store.py
import pysqlite3                     # swap in newer SQLite
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import os
import chromadb
from chromadb.config import Settings
import uuid

class VectorStore:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
    
    def create_collection(self, collection_name):
        """Create a new collection or get existing one"""
        try:
            collection = self.client.get_collection(collection_name)
            return collection
        except:
            return self.client.create_collection(collection_name)
    
    def add_documents(self, collection_name, chunks, embeddings):
        """Add documents to collection"""
        collection = self.create_collection(collection_name)
        
        # Generate IDs if not provided
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Add documents to collection
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids
        )
        
        return ids
    
    def search(self, collection_name, query_embedding, top_k=5):
        """Search for similar documents"""
        collection = self.client.get_collection(collection_name)
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return {
            "ids": results["ids"][0],
            "documents": results["documents"][0],
            "distances": results["distances"][0]
        }
    
    def get_all_collections(self):
        """Get all collections"""
        return self.client.list_collections()
    
    def delete_collection(self, collection_name):
        """Delete a collection"""
        self.client.delete_collection(collection_name)