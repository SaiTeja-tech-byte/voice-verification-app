import streamlit as st
import sounddevice as sd
import numpy as np
import librosa
import os
from sklearn.metrics.pairwise import cosine_similarity
import time

# Configuration
SAMPLE_RATE = 16000
DURATION = 2
N_MFCC = 20
REFERENCE_PATH = "reference_audio.npy"

# Utility functions
def record_audio(duration=DURATION, sr=SAMPLE_RATE):
    st.info("Recording... Please speak")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    return audio

def extract_mfcc(audio, sr=SAMPLE_RATE):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = librosa.util.fix_length(mfcc, size=63, axis=1)
    return mfcc.flatten()

def save_reference(audio):
    np.save(REFERENCE_PATH, audio)
    st.success("✅ Reference voice saved.")

def load_reference():
    if os.path.exists(REFERENCE_PATH):
        return np.load(REFERENCE_PATH)
    else:
        st.warning("⚠️ No reference voice found. Please register first.")
        return None

def compare_voices(ref_audio, live_audio):
    ref_emb = extract_mfcc(ref_audio)
    live_emb = extract_mfcc(live_audio)
    similarity = cosine_similarity([ref_emb], [live_emb])[0][0]
    return similarity

def mock_deepfake_check(audio):
    # Placeholder: Use a real model for actual detection
    return np.random.rand() < 0.3  # Randomly say 30% chance it's AI

# Streamlit UI
st.set_page_config(page_title="Live Voice Verification & Deepfake Detection")
st.title("🎙️ Voice Verification & Deepfake Detection")
tabs = st.tabs(["📝 Register Voice", "🔍 Live Interview Monitor"])

# Tab 1: Register
with tabs[0]:
    st.header("Step 1: Register Candidate Voice")
    input_method = st.radio("Choose input method:", ["Record via Mic", "Upload .wav File"])

    if input_method == "Record via Mic":
        if st.button("🎤 Record Voice"):
            audio = record_audio()
            save_reference(audio)
    else:
        uploaded_file = st.file_uploader("Upload reference .wav file", type=["wav"])
        if uploaded_file:
            y, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
            save_reference(y)

# Tab 2: Live Detection
with tabs[1]:
    st.header("Step 2: Live Interview Monitoring")
    ref_audio = load_reference()

    if ref_audio is not None:
        if st.button("🎧 Start Live Detection"):
            live_audio = record_audio()
            similarity = compare_voices(ref_audio, live_audio)
            st.write(f"🧑‍💼 Speaker Match Similarity: **{similarity:.2f}**")

            if similarity >= 0.85:
                st.success("✅ Speaker Verified - Same person")
            else:
                st.error("⚠️ Possible Impersonation - Voice does not match")

            is_fake = mock_deepfake_check(live_audio)
            if is_fake:
                st.error("🤖 Detected: Possible AI-generated voice")
            else:
                st.success("🧠 Voice seems human (Not detected as AI)")
