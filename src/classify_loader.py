"""EEG classification loader: load EDFs, window around transient marker, build FDataGrid."""

from pathlib import Path

import mne
import numpy as np
import pandas as pd
from skfda import FDataGrid


def load_manifest(csv_path):
    df = pd.read_csv(csv_path)
    usable = df[df["load_error"].isna() & (df["annotation_status"] == "ok")].copy()
    return usable.reset_index(drop=True)


def load_edf(file_id, edf_dir):
    """Open EDF header only (no data preloaded), assert 500 Hz, drop EKG channel."""
    raw = mne.io.read_raw_edf(
        str(Path(edf_dir) / f"{file_id}.edf"), preload=False, verbose=False
    )
    assert raw.info["sfreq"] == 500.0, f"{file_id}: expected 500 Hz, got {raw.info['sfreq']}"
    ekg_chs = [ch for ch in raw.ch_names if not ch.startswith("E ")]
    assert len(ekg_chs) == 1, f"{file_id}: expected 1 non-EEG channel, found {ekg_chs}"
    raw.drop_channels(ekg_chs)
    return raw


def build_dataset(manifest, edf_dir):
    """
    Build skfda FDataGrid of the most-active EEG channel per recording.
    Returns (fd, labels, certainty, half_width_s). Time axis is seconds relative to transient.
    """
    sfreq = 500.0
    half_width_s = min(manifest["pre_transient_s"].min(), manifest["post_transient_s"].min()) - 1 / sfreq
    half_samples = int(round(half_width_s * sfreq))
    peak_half = int(round(0.1 * sfreq))  # ±0.1 s window for channel selection

    curves = []
    for _, row in manifest.iterrows():
        raw = load_edf(row["file_id"], edf_dir)
        onset = int(round(row["transient_onset_s"] * sfreq))
        # Single disk read: only the window we need, not the full recording
        data = raw.get_data(start=onset - half_samples, stop=onset + half_samples)
        # data: (n_channels, 2*half_samples); onset is at column index half_samples
        ch = np.argmax(np.ptp(data[:, half_samples - peak_half : half_samples + peak_half], axis=1))
        curves.append(data[ch])  # ponytail: swap data[ch] → data.T for multichannel

    n = 2 * half_samples
    assert all(len(c) == n for c in curves), "Mismatched curve lengths"

    t = (np.arange(n) - half_samples) / sfreq
    fd = FDataGrid(data_matrix=np.stack(curves), grid_points=t)
    return fd, manifest["label_binary"].to_numpy(), manifest["certainty"].to_numpy(), half_width_s
