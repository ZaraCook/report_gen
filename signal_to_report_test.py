import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, load_metric
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

# Define directories
data_dir = Path(r"F:\Zara\EEG_Data_10sec") #replace with your data path
hf_token = "hf_token" #replace with yoru huggingface token
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
output_dir = Path(r"F:\Zara\report_gen\results") #output directory
output_dir.mkdir(parents=True, exist_ok=True)

annotations_dir = output_dir / "annotations"
annotations_dir.mkdir(parents=True, exist_ok=True)

debug_info_path = output_dir / "debug_info.txt"

# Define CNN for feature extraction
class EEGFeatureExtractor(nn.Module):
    def __init__(self, input_length):
        super(EEGFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(64 * (input_length // 4), 256)  # Calculate based on data length after convolutions and pooling

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc(x))
        return x

# Function to load data from H5 files in smaller batches
def load_data_in_batches(data_dir, batch_size=10):
    signals = []
    annotations = []
    filenames = []
    max_length = 0
    files = list(data_dir.glob("*.h5"))
    for i in tqdm(range(0, len(files), batch_size), desc="Loading data in batches"):
        batch_files = files[i:i + batch_size]
        batch_signals = []
        batch_annotations = []
        for file in batch_files:
            with h5py.File(file, 'r') as h5file:
                signal_data = h5file['signal'][:].astype(np.float32)
                annotation_data = h5file['annotations'][:]
                start_time = h5file['start_time'][()]
                # Adjust annotation timestamps relative to the start time of the segment
                adjusted_annotations = []
                for annot in annotation_data:
                    onset = float(annot[0]) + start_time
                    duration = float(annot[1])
                    description = annot[2]
                    adjusted_annotations.append([onset, duration, description])
                batch_signals.append(signal_data)
                batch_annotations.append(adjusted_annotations)
                filenames.append(file.stem)
                if signal_data.shape[1] > max_length:
                    max_length = signal_data.shape[1]
        signals.extend(batch_signals)
        annotations.extend(batch_annotations)
    return signals, annotations, filenames, max_length

# Function to pad signals in smaller batches and save them
def pad_and_save_signals(signals, max_length, batch_size, output_dir, debug_file):
    for i in tqdm(range(0, len(signals), batch_size), desc="Padding and saving signals"):
        batch_file_path = output_dir / f"padded_signals_batch_{i//batch_size}.npy"
        if batch_file_path.exists():
            continue  # Skip if this batch has already been processed and saved

        batch_signals = signals[i:i + batch_size]
        padded_signals = []
        expected_channels = 25

        for idx, signal in enumerate(batch_signals):
            debug_file.write(f"Original signal {i + idx} shape: {signal.shape}\n")
            if signal.shape[0] < expected_channels:
                pad_width = expected_channels - signal.shape[0]
                signal = np.pad(signal, ((0, pad_width), (0, 0)), 'constant')
                debug_file.write(f"Signal at index {i + idx} had {signal.shape[0]} channels, padded to {expected_channels} channels.\n")
            pad_width = max_length - signal.shape[1]
            if pad_width > 0:
                padded_signal = np.pad(signal, ((0, 0), (0, pad_width)), 'constant')
            else:
                padded_signal = signal[:, :max_length]
            padded_signals.append(padded_signal)
            debug_file.write(f"Padded signal {i + idx} shape: {padded_signal.shape}\n")
        
        padded_signals = np.stack(padded_signals).astype(np.float32)
        np.save(batch_file_path, padded_signals)
        del padded_signals  # Free memory

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

# Add a padding token to the tokenizer if it doesn't have one
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Function to tokenize data
def tokenize_function(signals, annotations, chunk_size=100):
    for start in range(0, len(signals), chunk_size):
        end = start + chunk_size
        chunk_signals = signals[start:end]
        chunk_annotations = annotations[start:end]
        
        tokenized_signals = tokenizer(list(map(str, chunk_signals.tolist())), padding="max_length", truncation=True)
        tokenized_annotations = tokenizer(list(map(str, chunk_annotations)), padding="max_length", truncation=True)
        
        yield {
            "input_ids": tokenized_signals["input_ids"], 
            "attention_mask": tokenized_signals["attention_mask"], 
            "labels": tokenized_annotations["input_ids"]
        }

# Prepare the dataset
signals, annotations, filenames, max_length = load_data_in_batches(data_dir)

# Write debug information to file
batch_size = 1000  # Adjust based on your available memory
with open(debug_info_path, 'w') as debug_file:
    pad_and_save_signals(signals, max_length, batch_size, output_dir, debug_file)

# Define custom metrics
def compute_custom_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Convert IDs to text for comparison
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    correct_annotations = 0
    correct_timestamps = 0
    total_annotations = len(decoded_labels)
    
    all_annotations = []

    for idx, (pred_text, label_text) in enumerate(zip(decoded_preds, decoded_labels)):
        pred_annotations = eval(pred_text)  # Assuming the annotations are saved as a list of dictionaries
        label_annotations = eval(label_text)

        file_annotations = {
            "file": filenames[idx],
            "predicted_annotations": pred_annotations,
            "actual_annotations": label_annotations
        }
        all_annotations.append(file_annotations)

        if not label_annotations and not pred_annotations:
            correct_annotations += 1  # Correct if both are empty
            continue

        for pred_annot, label_annot in zip(pred_annotations, label_annotations):
            if pred_annot['description'] == label_annot['description']:
                correct_annotations += 1
                if abs(pred_annot['onset'] - label_annot['onset']) <= 3:  # Tolerance of 3 seconds
                    correct_timestamps += 1

    annotation_accuracy = correct_annotations / total_annotations if total_annotations > 0 else 1
    timestamp_accuracy = correct_timestamps / correct_annotations if correct_annotations > 0 else 1

    # Save the annotations to a file
    with open(annotations_dir / "predicted_annotations.json", "w") as f:
        json.dump(all_annotations, f, indent=4)
    
    return {
        "annotation_accuracy": annotation_accuracy,
        "timestamp_accuracy": timestamp_accuracy,
    }

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",  # Updated to the new key
    save_strategy="epoch",
    save_total_limit=3,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    report_to="tensorboard"
)

# Load the saved padded signals and annotations for training
padded_signal_files = sorted(output_dir.glob("padded_signals_batch_*.npy"))

for file in tqdm(padded_signal_files, desc="Processing padded signal files"):
    signals = np.load(file)
    for data_chunk in tokenize_function(signals, annotations, chunk_size=10):
        dataset = Dataset.from_dict(data_chunk)

        # Split the dataset into training, validation, and testing sets
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.125, random_state=42)  # 0.125 of 80% is 10%

        # Define the trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_custom_metrics
        )

        # Train the model
        trainer.train()

# Save the model
model.save_pretrained(output_dir / "fine-tuned-llama3.1-8b")
tokenizer.save_pretrained(output_dir / "fine-tuned-llama3.1-8b")

# Evaluate the model
results = trainer.evaluate(test_dataset)
print(results)

# Save the results
with open(output_dir / "test_results.txt", "w") as f:
    f.write(str(results))
