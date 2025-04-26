# app/rag_engine.py
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

class RAGEngine:
    def __init__(
        self,
        embedding_model_name="all-MiniLM-L6-v2",
        llm_model_name="meta-llama/Llama-2-7b-chat-hf",  # Replace with appropriate model
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.device = device
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
    
    def embed_query(self, query):
        """Create embedding for query"""
        return self.embedding_model.encode(query)
    
    def generate_answer(self, query, retrieved_contexts, max_new_tokens=512):
        """Generate answer using LLM"""
        # Construct prompt
        prompt = self._construct_prompt(query, retrieved_contexts)
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
        
        # Decode answer
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the answer part (removing the prompt)
        answer = answer[len(prompt):].strip()
        
        return answer
    
    def _construct_prompt(self, query, retrieved_contexts):
        """Construct prompt for LLM"""
        context_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(retrieved_contexts)])
        
        # Template for Llama-2-chat
        prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant that answers questions based on the provided context.
If the context doesn't contain relevant information, admit that you don't know.
Always ground your answers in the context provided and be precise.
<</SYS>>

I have the following contexts:

{context_str}

Based on these contexts, please answer the following question:
{query} [/INST]
"""
        return prompt

    def process_query(self, query, vector_store, collection_name, top_k=5):
        """
        Process query through the RAG pipeline:
        1. Embed query
        2. Retrieve relevant chunks
        3. Generate answer
        """
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Retrieve relevant chunks
        search_results = vector_store.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Generate answer
        answer = self.generate_answer(
            query=query,
            retrieved_contexts=search_results["documents"]
        )
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": search_results["documents"],
            "chunk_ids": search_results["ids"]
        }