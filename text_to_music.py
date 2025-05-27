from audiocraft.models import MusicGen 
import streamlit as st
import os
import torch
import torchaudio
import numpy as np
import base64

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    return model

def generate_music_tensors(description, duration: int):
    print("Description: ", description)
    print("Duration: ", duration)

    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=120,
        temperature=0.8,
        duration=duration
    )

    # Generate the music tensor
    output = model.generate(
        descriptions=[description],
        progress=True
    )

    return output[0]

def save_audio(samples: torch.Tensor):
    sample_rate = 32000
    save_path = "audio_output/"
    os.makedirs(save_path, exist_ok=True)

    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()

    if samples.dim() == 2:
        samples = samples[None, ...]

    # Get list of existing audio files and increment filename dynamically
    existing_files = os.listdir(save_path)
    existing_indices = [
        int(f.split("_")[1].split(".")[0]) for f in existing_files if f.startswith("audio_") and f.endswith(".wav")
    ]
    next_index = max(existing_indices) + 1 if existing_indices else 0

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{next_index + idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

    # Return the last generated audio path
    return os.path.join(save_path, f"audio_{next_index}.wav")

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()

    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_icon=":musical_notes:",
    page_title="Music Gen"
)

def main():
    st.title("🎵 Text to Music Generation 🎧")

    with st.expander("See Explanation"):
        st.write("This is a music generation app using Facebook's MusicGen model.")

    text_area = st.text_area("Enter your description")
    time_slider = st.slider("Select time duration (in seconds)", 2, 20, 5)

    if text_area and time_slider:
        st.json(
            {
                "Your Description": text_area,
                "Selected Time Duration": time_slider
            }
        )

        st.subheader("Generated Music")

        music_tensors = generate_music_tensors(text_area, time_slider)
        print("Music Tensors:", music_tensors)

        # Save generated audio and get the latest saved file
        audio_filepath = save_audio(music_tensors)

        # Display the generated audio
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/wav')
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
