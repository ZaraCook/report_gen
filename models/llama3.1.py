import os
import mne
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# Hugging Face token
hf_token = "hf_token" #replace with yoru huggingface token

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B", token=hf_token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B", token=hf_token)

# Set the padding token to eos token
tokenizer.pad_token = tokenizer.eos_token

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
    
    # Print annotations for debugging
    print("Annotations Summary:\n", annotations_summary)

    # Generate the "Description" section using annotations
    description_prompt = (
        f"Summarize the following EEG annotations into a detailed and cohesive paragraph: {annotations_summary}. "
        f"Include necessary medical insights and explanations without inferring additional details."
    )
    print("Description Prompt:\n", description_prompt)
    
    inputs_description = tokenizer(description_prompt, return_tensors="pt", padding=True, truncation=True)
    description_ids = model.generate(inputs_description["input_ids"], max_new_tokens=200, num_beams=4, early_stopping=True)
    description = tokenizer.decode(description_ids[0], skip_special_tokens=True).replace(description_prompt, "").strip()
    print("Generated Description:\n", description)

    # Ensure the description is a complete paragraph
    if not description.endswith('.'):
        description = description.rsplit('.', 1)[0] + '.'

    # Generate the "Impression" section
    impression_prompt = (
        f"Based on the summarized EEG annotations and the following description: {description}, provide a medical impression in 2-3 sentences. "
        f"Is there any seizure activity present? Include whether this EEG is normal or abnormal, referring only to the annotations and description provided. Ignore artifacts and only state your medical impression."
    )
    print("Impression Prompt:\n", impression_prompt)
    
    inputs_impression = tokenizer(impression_prompt, return_tensors="pt", padding=True, truncation=True)
    impression_ids = model.generate(inputs_impression["input_ids"], max_new_tokens=200, num_beams=4, early_stopping=True)
    impression = tokenizer.decode(impression_ids[0], skip_special_tokens=True).replace(impression_prompt, "").strip()
    print("Generated Impression:\n", impression)

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
with open('report_llama-3.1-8b.txt', 'w') as f:
    f.write(report)
