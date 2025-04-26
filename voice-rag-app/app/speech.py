# app/speech.py
import numpy as np
import torch
import torchaudio
import pyttsx3
import threading
import tempfile
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

class SpeechProcessor:
    def __init__(self, whisper_model="openai/whisper-base"):
        # Initialize ASR components
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
        
        # Initialize pyttsx3 for TTS
        self.tts_engine = pyttsx3.init()
        # Configure properties (optional)
        self.tts_engine.setProperty('rate', 150)  # Speed of speech
        self.tts_engine.setProperty('volume', 0.9)  # Volume (0.0 to 1.0)
        
        # Get available voices and set a preferred voice (optional)
        voices = self.tts_engine.getProperty('voices')
        if voices:  # Check if voices are available
            # Usually index 0 is male voice, 1 is female voice if available
            if len(voices) > 1:
                self.tts_engine.setProperty('voice', voices[1].id)  # Set female voice
            else:
                self.tts_engine.setProperty('voice', voices[0].id)  # Set default voice
    
    def transcribe_audio(self, audio_file_path=None, audio_array=None, sample_rate=16000):
        """
        Transcribe audio to text using Whisper
        Can accept either a file path or audio array
        """
        if audio_file_path:
            # Load audio from file
            audio_array, sample_rate = torchaudio.load(audio_file_path)
            # Convert stereo to mono if needed
            if audio_array.shape[0] > 1:
                audio_array = torch.mean(audio_array, dim=0, keepdim=True)
            audio_array = audio_array.squeeze().numpy()
        
        if audio_array is None:
            raise ValueError("Either audio_file_path or audio_array must be provided")
        
        # Process audio with Whisper
        input_features = self.whisper_processor(
            audio_array, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features
        
        # Generate token ids
        predicted_ids = self.whisper_model.generate(input_features)
        
        # Decode token ids to text
        transcription = self.whisper_processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription
    
    def text_to_speech(self, text, output_path=None):
        """
        Convert text to speech using pyttsx3
        If output_path is provided, save to file
        Otherwise, just play the audio
        """
        if not output_path:
            # Just play the text without saving
            def speak_text():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            # Run in a separate thread to avoid blocking
            thread = threading.Thread(target=speak_text)
            thread.daemon = True
            thread.start()
            return None
        else:
            # Create temporary file if needed
            temp_file = None
            if not output_path.endswith('.wav'):
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_path = temp_file.name
                temp_file.close()
            else:
                temp_path = output_path
            
            # Save speech to file
            self.tts_engine.save_to_file(text, temp_path)
            self.tts_engine.runAndWait()
            
            # If using temp file, convert format if needed
            if temp_file:
                # For simplicity, just copy the file
                # In a real app, you might want to convert from wav to mp3/ogg
                with open(temp_path, 'rb') as src, open(output_path, 'wb') as dst:
                    dst.write(src.read())
                os.unlink(temp_path)  # Remove temp file
            
            return output_path

class VoiceActivityDetector:
    def __init__(self, threshold=0.01, min_silence_duration=0.5):
        self.threshold = threshold
        self.min_silence_duration = min_silence_duration
        self.sample_rate = 16000
    
    def detect_voice(self, audio_array):
        """
        Detect voice activity in audio array
        Returns array of (start_time, end_time) in seconds
        """
        # Calculate energy of signal
        energy = np.abs(audio_array)
        
        # Find intervals with energy above threshold
        is_speech = energy > self.threshold
        
        # Convert to time ranges
        min_silence_samples = int(self.min_silence_duration * self.sample_rate)
        
        # Find speech segments
        speech_segments = []
        current_segment = None
        
        for i, speech in enumerate(is_speech):
            if speech and current_segment is None:
                # Start of speech
                current_segment = i
            elif not speech and current_segment is not None:
                # End of speech
                if i - current_segment > min_silence_samples:
                    speech_segments.append((
                        current_segment / self.sample_rate,
                        i / self.sample_rate
                    ))
                current_segment = None
        
        # Handle case where speech continues until the end
        if current_segment is not None:
            speech_segments.append((
                current_segment / self.sample_rate,
                len(is_speech) / self.sample_rate
            ))
        
        return speech_segments