# utils/inference.py

def infer_and_display(model, audio_path):
    print(f"🔊 Transcribing: {audio_path}")
    transcription = model.transcribe(audio_path)
    print(f"📝 Transcription: {transcription}")
    return transcription
