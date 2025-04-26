import os
import fitz  # PyMuPDF
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class PDFProcessor:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page in doc:
                text += page.get_text()
            
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def chunk_text(self, text):
        """Split text into chunks"""
        return self.text_splitter.split_text(text)
    
    def create_embeddings(self, chunks):
        """Create embeddings for text chunks"""
        return self.embedding_model.encode(chunks)
    
    def process_pdf(self, pdf_path):
        """Process PDF file: extract text, chunk, and create embeddings"""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            # Split text into chunks
            chunks = self.chunk_text(text)
            
            # Create embeddings
            embeddings = self.create_embeddings(chunks)
            
            return {
                "text": text,
                "chunks": chunks,
                "embeddings": embeddings
            }
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")