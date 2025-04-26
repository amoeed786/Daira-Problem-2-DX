# app/summarizer.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Summarizer:
    def __init__(
        self,
        model_name="meta-llama/Llama-2-7b-chat-hf",  # Replace with appropriate model
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
    
    def _chunk_long_text(self, text, max_chunk_size=3000):
        """Split long text into chunks for processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) >= max_chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _construct_summary_prompt(self, text):
        """Construct prompt for summarization"""
        prompt = f"""<s>[INST] <<SYS>>
You are a helpful AI assistant that creates clear, concise, and informative summaries.
Focus on the key points and main ideas in the text.
Your summary should be well-structured and capture the essence of the original text.
<</SYS>>

Please create a comprehensive summary of the following text:

{text}

Provide a well-structured summary that covers the main points, key findings, and important conclusions. [/INST]
"""
        return prompt
    
    def _summarize_chunk(self, chunk, max_new_tokens=512):
        """Summarize a single chunk"""
        prompt = self._construct_summary_prompt(chunk)
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate summary
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
            )
        
        # Decode summary
        summary = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the summary part (removing the prompt)
        summary = summary[len(prompt):].strip()
        
        return summary
    
    def summarize(self, text, max_new_tokens=512):
        """
        Generate abstractive summary of text
        For long texts, chunks the text and summarizes each chunk
        """
        # Check if text is too long
        if len(text.split()) > 3000:
            # Chunk the text
            chunks = self._chunk_long_text(text)
            
            # Summarize each chunk
            chunk_summaries = [self._summarize_chunk(chunk) for chunk in chunks]
            
            # Combine chunk summaries
            combined_summary = " ".join(chunk_summaries)
            
            # Create final summary of summaries
            final_summary = self._summarize_chunk(combined_summary)
            return final_summary
        else:
            # Summarize directly
            return self._summarize_chunk(text, max_new_tokens)
    
    def summarize_chunks(self, chunks, max_new_tokens=512):
        """Summarize a set of retrieved chunks"""
        # Combine chunks into a single text
        combined_text = "\n\n".join(chunks)
        
        # Generate summary
        return self.summarize(combined_text, max_new_tokens)