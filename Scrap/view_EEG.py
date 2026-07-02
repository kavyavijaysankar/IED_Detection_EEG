import mne

# Load the Kural EDF file
file_path = 'Kural_Dataset/Recordings/S03.edf'  # Update this path to your EDF file
raw = mne.io.read_raw_edf(file_path, preload=True)

# 1. Basic info: See the sampling rate and channel names
print(raw.info)
for ann in raw.annotations:
    print(f"{ann['description']}  at  {ann['onset']:.3f}s")

# 2. Interactive Browser
# This opens a window to scroll through the EEG
raw.plot(block=True)