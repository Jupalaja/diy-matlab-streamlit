import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import logging
import warnings

logging.getLogger('matplotlib').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

st.title('Audio File Uploader and Waveform Plotter')
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])


def convert_to_mono(song):
    mono_song = librosa.to_mono(song.T)
    return mono_song


def resample_audio(song, original_sr, target_sr):
    resampled_song = librosa.resample(song, orig_sr=original_sr, target_sr=target_sr)
    return resampled_song


if uploaded_file is not None:
    song, samplerate = sf.read(uploaded_file)
    st.audio(uploaded_file, format='audio/wav')

    duration = song.shape[0] / samplerate
    st.write(f"Sample Rate: {samplerate} Hz")
    st.write(f"Duration: {duration:.2f} seconds")

    mono_song = convert_to_mono(song)
    target_sr = 8000
    resampled_song = resample_audio(mono_song, samplerate, target_sr)
    new_duration = resampled_song.shape[0] / target_sr
    time = np.linspace(0., new_duration, resampled_song.shape[0])

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(time, resampled_song, color='tab:blue')
    ax.set(xlabel='Time (s)', ylabel='Amplitude')
    ax.set_title("Cleaned Waveform of the Audio File")
    ax.grid()
    st.pyplot(fig)
