import os
import torch
from pydub import AudioSegment
from pathlib import Path
from bm import play
from bm.events import Sound
from math import floor

# Function to save a tensor to a file
def save_tensor(tensor, filename):
    torch.save(tensor, filename)

# Function to save a list of dictionaries to a JSON file
def save_metadata(metadata, filename):
    import json
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)

# Function to segment the audio based on start time and duration
def save_audio_segment(full_audio_path, start_time, duration, save_path, sound_event):
    audio = AudioSegment.from_wav(full_audio_path)
    start_ms = floor(start_time * 1000)  # Convert to milliseconds
    duration_ms = floor(duration * 1000)  # Convert to milliseconds
    
    # Check if the calculated segment is valid
    if start_ms < 0 or start_ms >= len(audio):
        print(f"Invalid start time: {start_ms} ms for file {full_audio_path}")
        print(sound_event.start)
        print(sound_event.offset)
        return False
    
    if duration_ms <= 0 or start_ms + duration_ms > len(audio):
        print(f"Invalid duration: {duration_ms} ms for file {full_audio_path}")
        print(sound_event.start)
        print(sound_event.offset)
        print(sound_event.duration)
        print(len(audio))
        return False
    
    # Extract the audio segment
    segment = audio[start_ms:start_ms + duration_ms]
    
    # Save the segment if it's non-empty
    if len(segment) > 0:
        segment.export(save_path, format="wav")
        return True
    else:
        print(f"Segment is empty for file {full_audio_path}")
        return False

def process_and_save_batches(loader, output_dir, solver):
    metadata = []
    
    for i, batch in enumerate(loader):

        #print(batch)

        subject_id = batch.subject_index.item()
        recording_id = batch.recording_index.item()
        save_dir_eeg_embeddings = Path(output_dir) / f"subject_{subject_id}" / f"recording_{recording_id}" / "eeg_embeddings"
        save_dir_audio_embeddings = Path(output_dir) / f"subject_{subject_id}" / f"recording_{recording_id}" / "audio_embeddings"
        save_dir_audio_wav = Path(output_dir) / f"subject_{subject_id}" / f"recording_{recording_id}" / "audio_wavs"

        save_dir_eeg_embeddings.mkdir(parents=True, exist_ok=True)
        save_dir_audio_embeddings.mkdir(parents=True, exist_ok=True)
        save_dir_audio_wav.mkdir(parents=True, exist_ok=True)

        # meg = batch.meg.unsqueeze(0)
        # batch.meg = meg
        # inputs = dict(meg=meg)
        # eeg_embedding = model(inputs, batch)
        batch.meg = batch.meg.unsqueeze(0)
        #batch.to(solver.device)
        eeg_embedding, _, _, _ = solver._process_batch(batch)

        # Save MEG data
        eeg_save_path = save_dir_eeg_embeddings / f"eeg_embedding_{i}.pt"
        save_tensor(eeg_embedding, eeg_save_path)
        #print(batch.meg.shape)

        # Save features
        features_save_path = save_dir_audio_embeddings / f"audio_embedding_{i}.pt"
        save_tensor(batch.features, features_save_path)
        #print(batch.features.shape)

        # Save audio segment
        #print(batch._event_lists)

        # Retrieve the Sound event
        sound_event = None
        for event in batch._event_lists[0]:
            if isinstance(event, Sound):
                sound_event = event
                break

        if sound_event:
            try:
                start_time = sound_event.offset
                if start_time < 0:
                    print('Invalid start_time')

                duration = sound_event.duration
                if duration < 0.1:
                    print('Duration invalid')
                audio_file = sound_event.filepath

                audio_segment_save_path = save_dir_audio_wav / f"audio_segment_{i}.wav"
                save_audio_segment(audio_file, start_time, duration, audio_segment_save_path, sound_event)
                
                metadata.append({
                    "audio_id": f"audio_segment_{i}",
                    "meg_signal": str(eeg_save_path),
                    "features": str(features_save_path),
                    "filepath": str(audio_segment_save_path)
                })
            except Exception as e:
                print(f"Error processing event list for batch {i}: {e}")



    # Save metadata to JSON
    metadata_save_path = Path(output_dir) / "metadata.json"
    save_metadata(metadata, metadata_save_path)

if __name__ == '__main__':

    args = ['dset.selections=[brennan2019]']
    solver = play.get_solver_from_args(args)

    solver.restore()

    loaders = solver.loaders
    train_loader = loaders['train'].dataset
    valid_loader = loaders['valid'].dataset
    test_loader = loaders['test'].dataset

    # model = solver.model
    # model.to('cpu')

    # torch.save(train_loader, 'train_loader.pth')
    # torch.save(valid_loader, 'valid_loader.pth')
    # torch.save(test_loader, 'test_loader.pth')

    # # Load the train loader
    # loaded_train_loader = torch.load('train_loader.pth')
    
    # Define the output directory where everything will be saved
    output_dir_train = "./train_data"
    output_dir_valid = "./valid_data"
    output_dir_test = "./test_data"

    # Process and save each batch in the loader
    process_and_save_batches(train_loader, output_dir_train, solver)
    #process_and_save_batches(valid_loader, output_dir_valid)
    #process_and_save_batches(test_loader, output_dir_test)
