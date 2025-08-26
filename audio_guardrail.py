from transformers import pipeline
import soundfile as sf
import librosa

# Load ASR pipeline once globally
asr = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=-1)

def transcribe_audio(file_path):
    # Read audio file
    audio_input, sample_rate = sf.read(file_path)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
        
    transcription = asr(audio_input)
    return transcription['text']

def analyze_audio_file(filepath, guardrail_bot):
    print(f"\n🎤 Transcribing audio file: {filepath}")
    transcript = transcribe_audio(filepath)
    print(f"📝 Transcript preview: {transcript[:200]}...\n")

    # Use existing guardrail bot's text detection on transcript
    score, is_malicious, details = guardrail_bot.get_security_score_for_segment(transcript)
    print(f"🎯 Threat score: {score:.3f}")
    if is_malicious:
        print(f"🚨 Detected malicious content: {details}")
    else:
        print("✅ No malicious content detected")

    return score, is_malicious
