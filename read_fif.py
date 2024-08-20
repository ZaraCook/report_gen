import mne
import matplotlib.pyplot as plt
import sys

# Redirect stdout to a file
output_file_path = 'output.txt'
sys.stdout = open(output_file_path, 'w')

# Correct the file path
file_path = r"\\NRC-065359\Abbotsford\000b9ee7-9f19-412d-bf22-360cd7c2f325_raw.fif"  # Use a raw string or escape backslashes

# Load the raw data
raw = mne.io.read_raw_fif(file_path, preload=True)

# Print the basic info
print(raw.info)

# Plot the raw data to visually inspect it
raw.plot()
plt.show()  # Keep the plot open

# Print the channel names and types
print("\nChannel names and types:")
for ch_name in raw.info['ch_names']:
    ch_type = mne.io.pick.channel_type(raw.info, raw.info['ch_names'].index(ch_name))
    print(f"{ch_name}: {ch_type}")

# Plot sensor locations
raw.plot_sensors(kind='topomap')
plt.show()  # Keep the plot open

# Extract and print events, if any
try:
    events = mne.find_events(raw)
    print("\nEvents:")
    print(events)
except ValueError:
    print("\nNo events found.")

# Access and print annotations
if raw.annotations:
    print("\nAnnotations:")
    print(raw.annotations)
else:
    print("\nNo annotations found.")

# Print head shape points and digitization information, if available
if 'dig' in raw.info:
    print("\nDigitization points (head shape, fiducials, etc.):")
    for point in raw.info['dig']:
        print(point)

# Print details of any bad channels
if raw.info['bads']:
    print("\nBad channels:")
    print(raw.info['bads'])
else:
    print("\nNo bad channels marked.")

# Print highpass and lowpass filter settings
print(f"\nHighpass filter: {raw.info['highpass']} Hz")
print(f"Lowpass filter: {raw.info['lowpass']} Hz")

# Restore stdout
sys.stdout.close()
sys.stdout = sys.__stdout__
