from ASR.models.asr_model import HindiASRModel
from transformers import MarianMTModel, MarianTokenizer
import torch
import numpy as np
import joblib
import soundfile
import numpy as np
import librosa
import os
from sklearn.model_selection import train_test_split
import glob



asr = HindiASRModel(model_size="medium")

int2emotion_new = {
    "anger": "angry",
    "sad": "sad",
    "happy": "happy",
    "neutral": "calm"
}

# we allow only these four emotions
AVAILABLE_EMOTIONS_NEW = set(int2emotion_new.values())

def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
    Features supported:
        - MFCC (mfcc)
        - Chroma (chroma)
        - MEL Spectrogram Frequency (mel)
        - Contrast (contrast)
        - Tonnetz (tonnetz)
    e.g:
    `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
            
        result = np.array([])
        
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
            
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
            
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
            
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
            
    return result


model_filename = "./ESR/result/random_forest_model.joblib" # achieved around 0.75 accuracy with high precision and recall for this model
loaded_classifier = joblib.load(model_filename)
print(f"ESR Loaded model from {model_filename}")


model_name = "Helsinki-NLP/opus-mt-hi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"Model loaded on {device}")


def translate_hindi_to_english(text):
    """
    Translate Hindi text to English using the pretrained model
    
    Args:
        text (str): Hindi text to be translated
        
    Returns:
        str: Translated English text
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs)
    
    translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)    
    return translated_text[0]



def infer(audio_path):
    hindi_text = asr.transcribe(audio_path)
    print("Hindi text : ",hindi_text)
    X = []
    features = extract_feature(audio_path, mfcc=True, chroma=True, mel=True)
    X.append(features)
    loaded_prediction = loaded_classifier.predict(X)
    print("Emotion Predicted :", loaded_prediction[0])

    english_translation = translate_hindi_to_english(hindi_text)
    print("English Translation : ",english_translation)
