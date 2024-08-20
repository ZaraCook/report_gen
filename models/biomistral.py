import os
import mne
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# Hugging Face token
hf_token = "hf_token" #replace with yoru huggingface token

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B", token=hf_token)

'''
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", token=hf_token)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", token=hf_token)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", token=hf_token)

tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b", token=hf_token)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b", token=hf_token)
'''

# Define a function to process a single FIF file
def process_fif_file(file_path):
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load the raw data
    raw = mne.io.read_raw_fif(file_path, preload=True)
    
    # Extract annotations
    annotations = []
    if raw.annotations:
        for annot in raw.annotations:
            annotations.append(f"Onset: {annot['onset']}, Description: {annot['description']}")
    else:
        annotations.append("No annotations found.")

    annotations_summary = "\n".join(annotations)

    # Generate the "Description" section using annotations
    description_prompt = (
        f"You are a medical doctor analyzing clinical EEG reports. Here is the report you will analyze: \n\n{annotations_summary}. \n\nQuestion: What does the report tell you about the EEG patient?"
    )
    inputs_description = tokenizer(description_prompt, return_tensors="pt")
    description_ids = model.generate(inputs_description["input_ids"], max_new_tokens=200, num_beams=4, early_stopping=True, temperature=0.7, top_p=0.9, num_return_sequences=1)
    description = tokenizer.decode(description_ids[0], skip_special_tokens=True).replace(description_prompt, "").strip()

    # Ensure the description is a complete paragraph
    if not description.endswith('.'):
        description = description.rsplit('.', 1)[0] + '.'

    # Generate the "Impression" section
    impression_prompt = (
        f"You are a medical doctor analyzing clinical EEG reports. Here is the report you will analyze: \n\n{annotations_summary}.\n\nQuestion: Is there any seizure activity present?"
    )
    inputs_impression = tokenizer(impression_prompt, return_tensors="pt")
    impression_ids = model.generate(inputs_impression["input_ids"], max_new_tokens=200, num_beams=4, early_stopping=True)
    impression = tokenizer.decode(impression_ids[0], skip_special_tokens=True).replace(impression_prompt, "").strip()

    # Remove specific lines from the impression
    if "Please let me know if you want me to make any changes." in impression:
        impression = impression.split("Please let me know if you want me to make any changes.")[0].strip()

    # Ensure the impression is a complete paragraph
    if not impression.endswith('.'):
        impression = impression.rsplit('.', 1)[0] + '.'

    # Construct the final report with scan ID, date, and annotations
    scan_id = os.path.basename(file_path)
    report_date = datetime.now().strftime("%Y-%m-%d")
    report = (
        f"Scan ID: {scan_id}\nDate: {report_date}\n\n"
        f"Annotations:\n{annotations_summary}\n\n"
        f"Description:\n{description}\n\nImpression:\n{impression}\n"
    )
    
    return report

# Correct the file path
file_path = r"\\NRC-065359\Abbotsford\000b9ee7-9f19-412d-bf22-360cd7c2f325_raw.fif"  # Use a raw string or escape backslashes

# Process a single FIF file
report = process_fif_file(file_path)

# Print the report to stdout
print(report)

# Save the report to a file
with open('report_biomistral.txt', 'w') as f:
    f.write(report)
