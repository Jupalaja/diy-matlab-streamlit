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


def filter_input_audio(uploaded_file):
    song, samplerate = sf.read(uploaded_file)
    st.audio(uploaded_file, format='audio/wav')
    duration = song.shape[0] / samplerate

    st.write(f"Original Sample Rate: {samplerate} Hz")
    st.write(f"Duration: {duration:.2f} seconds")

    mono_song = convert_to_mono(song)
    target_sr = 8000
    resampled_song = resample_audio(mono_song, samplerate, target_sr)
    new_duration = resampled_song.shape[0] / target_sr
    time = np.linspace(0., new_duration, resampled_song.shape[0])

    # Plot the waveform
    st.header("Filtered signal")
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(time, resampled_song, color='tab:blue')
    ax.set(xlabel='Time (s)', ylabel='Amplitude')
    ax.set_title("Cleaned Waveform of the Audio File")
    ax.grid()
    st.pyplot(fig)

    return resampled_song, target_sr


def get_spectrogram(song, target_sr):
    # Compute Spectrogram
    window_time = 0.064  # Window time in seconds
    window_size = int(window_time * target_sr)  # Convert window time to samples
    hop_length = window_size // 2  # 50% overlap
    n_fft = 512  # Number of FFT points
    window = np.hamming(window_size)  # Hamming window

    # Compute STFT
    D = librosa.stft(song, n_fft=n_fft, hop_length=hop_length, win_length=window_size, window=window)

    # Convert amplitude to dB
    S_db = librosa.amplitude_to_db(abs(D), ref=np.max)

    # Plot spectrogram
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        S_db, sr=target_sr, hop_length=hop_length, x_axis='time', y_axis='log', ax=ax, cmap='viridis', vmin=-50, vmax=0
    )

    st.text(" ")
    st.header("Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label='dB')
    ax.set_title('Wide-Band Spectrogram')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim([20, np.min([target_sr / 2, 8000])])
    st.pyplot(fig)
    return S_db


if uploaded_file is not None:
    resampled_song, target_sr = filter_input_audio(uploaded_file)

    spectrogram_data = get_spectrogram(resampled_song, target_sr)




