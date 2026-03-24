import mne
import skfda
import numpy as np
import matplotlib.pyplot as plt
from skfda.representation.basis import BSpline

# --- 1. Load the .edf File ---
# Replace 'your_file.edf' with your actual filename
file_path = '/Users/kavya/Desktop/uni/research project/Kural EEG Dataset/S01.edf'
raw = mne.io.read_raw_edf(file_path, preload=True)

# Let's pick a channel and a small window for the test (e.g., 2 seconds)
# Epilepsy research often looks at F3, F4, T3, T4, etc.
raw.filter(1, 40)  # Basic bandpass filter to remove drift and line noise
data, times = raw.get_data(picks=[0], start=0, stop=int(raw.info['sfreq'] * 2), return_times=True)

# --- 2. Convert to skfda Grid Object ---
# skfda expects (n_samples, n_points)
fd_grid = skfda.FDataGrid(data, grid_points=times)

# --- 3. Define the B-spline Basis ---
# For IEDs, we want a dense knot placement. 
# Let's try 50 basis functions for a 2-second window.
n_basis = 80 
basis = BSpline(n_basis=n_basis, order=4)

# --- 4. Smoothing (The "Fitting" Step) ---
# This converts the discrete grid into a continuous functional object
# smoothing_parameter (lambda) controls the stiffness of the spline
fd_basis = fd_grid.to_basis(basis, smoothing_parameter=0.0001)

# --- 5. Visualization ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(times, data[0], color='gray', alpha=0.3, label='Raw EEG Data')
fd_basis.plot(axes=ax, color='red', linewidth=2, label='B-spline Fit')
ax.set_title("B-spline Fitting of EEG Signal")
ax.legend()
plt.show()