import os
import mne
import h5py
import numpy as np
from pathlib import Path

# Define the directories
data_dir = Path(r"\\NRC-065359\Abbotsford") #data path
output_dir = Path(r"F:\Zara\EEG_Data") #output directory

# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Function to process and save data from a FIF file
def process_fif_file(file_path, output_dir):
    raw = mne.io.read_raw_fif(file_path, preload=True)
    
    # Extract signals
    signal_data = raw.get_data()
    
    # Extract annotations
    annotations = raw.annotations

    # Prepare data to save
    annotations_data = []
    if annotations:
        for annot in annotations:
            annotations_data.append([str(annot['onset']), str(annot['duration']), str(annot['description'])])
    
    # Save to H5 file
    output_file_path = output_dir / (file_path.stem + '.h5')
    with h5py.File(output_file_path, 'w') as h5file:
        h5file.create_dataset('signal', data=signal_data)
        h5file.create_dataset('annotations', data=np.array(annotations_data, dtype=h5py.string_dtype()))
    
    print(f"Processed and saved {file_path} to {output_file_path}")

# Loop through all FIF files in the directory
for fif_file in data_dir.glob("*.fif"):
    process_fif_file(fif_file, output_dir)

print("Processing complete.")
