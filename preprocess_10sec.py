import os
import mne
import h5py
import numpy as np
from pathlib import Path

# Define the directories
data_dir = Path(r"\\NRC-065359\Abbotsford") #data path
output_dir = Path(r"F:\Zara\EEG_Data_10sec") #output directory

# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Function to process and save data from a FIF file
def process_fif_file(file_path, output_dir, window_size=10):
    raw = mne.io.read_raw_fif(file_path, preload=True)
    
    # Extract signals
    signal_data = raw.get_data()
    sfreq = raw.info['sfreq']  # Sampling frequency
    n_samples_per_window = int(window_size * sfreq)  # Number of samples in a 10-second window
    
    # Extract annotations
    annotations = raw.annotations

    # Prepare windows and save to H5 files
    n_windows = signal_data.shape[1] // n_samples_per_window  # Total number of 10-second windows
    for i in range(n_windows):
        start_sample = i * n_samples_per_window
        end_sample = start_sample + n_samples_per_window
        window_signal = signal_data[:, start_sample:end_sample]  # Extract 10-second window of signal data
        
        # Extract annotations for the current window
        start_time = start_sample / sfreq  # Convert start sample index to time
        end_time = end_sample / sfreq  # Convert end sample index to time
        window_annotations = []
        if annotations:
            for annot in annotations:
                annot_start = annot['onset']
                annot_end = annot_start + annot['duration']
                # Check if annotation falls within the current window
                if annot_start >= start_time and annot_end <= end_time:
                    window_annotations.append([str(annot['onset'] - start_time), str(annot['duration']), str(annot['description'])])
        
        # Save to H5 file
        output_file_name = f"{file_path.stem}_{int(start_time)}.h5"
        output_file_path = output_dir / output_file_name
        with h5py.File(output_file_path, 'w') as h5file:
            h5file.create_dataset('signal', data=window_signal)
            h5file.create_dataset('annotations', data=np.array(window_annotations, dtype=h5py.string_dtype()))
            h5file.create_dataset('original_file', data=np.string_(file_path.name))
            h5file.create_dataset('start_time', data=start_time)
            h5file.create_dataset('end_time', data=end_time)
        
        print(f"Processed and saved {output_file_path}")

# Loop through all FIF files in the directory
for fif_file in data_dir.glob("*.fif"):
    process_fif_file(fif_file, output_dir)

print("Processing complete.")
