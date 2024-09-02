import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

def load_eeg_embedding(file_path):
    """Loads the EEG embedding tensor from a file."""
    try:
        eeg_embedding = torch.load(file_path, map_location='cpu')
        return eeg_embedding
    except Exception as e:
        print(f"Error loading EEG embedding: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Path to the EEG embedding file
    eeg_embedding_file = "recording_0/audio_embeddings/audio_embedding_0.pt"

    # Load the EEG embedding
    eeg_embedding = load_eeg_embedding(eeg_embedding_file)
    print(eeg_embedding)
    print(eeg_embedding.shape)
