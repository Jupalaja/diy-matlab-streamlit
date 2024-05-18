import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import logging
import warnings

# Configure logging to reduce verbosity
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Suppress specific Matplotlib warnings
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# Title of the app
st.title('Audio File Uploader and Waveform Plotter')

# Upload audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Reading the uploaded file
    data, samplerate = sf.read(uploaded_file)
    st.audio(uploaded_file, format='audio/wav')

    # Display some audio info
    duration = data.shape[0] / samplerate
    st.write(f"Sample Rate: {samplerate} Hz")
    st.write(f"Duration: {duration:.2f} seconds")

    # get the samples of duration
    time = np.linspace(0., duration, data.shape[0])

    # Plot waveform using matplotlib and librosa
    fig, ax = plt.subplots()
    ax.plot(time, data, color='tab:blue')
    ax.set(xlabel='Time (s)', ylabel='Amplitude')
    ax.set_title("Waveform of the Audio File")
    ax.grid()

    # Display plot in Streamlit
    st.pyplot(fig)
