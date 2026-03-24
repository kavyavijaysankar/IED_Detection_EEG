import mne

# Load the Kural EDF file
file_path = '/Users/kavya/Documents/GitHub/Epilepsy-EEG-FDA/Kural (2020) Dataset/S01.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# 1. Basic info: See the sampling rate and channel names
print(raw.info)

# 2. Interactive Browser
# This opens a window to scroll through the EEG
raw.plot(block=True)