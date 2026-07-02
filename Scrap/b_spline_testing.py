import mne
import skfda
import matplotlib.pyplot as plt
from skfda.representation.basis import BSpline

# Load .edf File
file_path = 'Kural_Dataset/Recordings/S04.edf' 
raw = mne.io.read_raw_edf(file_path, preload=True)

# pick a channel and a small window
raw.filter(1, 40)  # Basic bandpass filter to remove drift and line noise
data, times = raw.get_data(picks=[0], start=0, stop=int(raw.info['sfreq'] * 2), return_times=True)

# Convert to skfda Grid Object
# skfda expects (n_samples, n_points)
fd_grid = skfda.FDataGrid(data, grid_points=times)

# Define the B-spline Basis
n_basis = 80 
basis = BSpline(n_basis=n_basis, order=4)

# Smoothing

fd_basis = fd_grid.to_basis(basis, smoothing_parameter=0.0001)

# visualization
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(times, data[0], color='gray', alpha=0.3, label='Raw EEG Data')
fd_basis.plot(axes=ax, color='red', linewidth=2, label='B-spline Fit')
ax.set_title("B-spline Fitting of EEG Signal")
ax.legend()
plt.show()