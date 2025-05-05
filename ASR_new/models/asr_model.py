# models/asr_model.py

import torch
import whisper

class HindiASRModel:
    def __init__(self, model_size="small"):
        self.model = whisper.load_model(model_size)
    
    def transcribe(self, audio_path: str, language: str = "hi") -> str:
        result = self.model.transcribe(audio_path, language=language)
        return result['text']
