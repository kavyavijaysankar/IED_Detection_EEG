"""
eeg_loader.py
-------------
Loads all EDF files from a directory, merges with a demographics CSV,
and returns a tidy summary DataFrame plus a dict of MNE raw objects.

Usage
-----
    from eeg_loader import load_eeg_dataset

    df, raws = load_eeg_dataset(
        edf_dir='path/to/edf/folder',
        demographics_csv='path/to/demographics.csv'
    )

    # df  -> one row per participant, all metadata + annotation properties
    # raws -> {'S01': <Raw>, 'S02': <Raw>, ...}  keyed by file_id

Columns in df
-------------
    file_id               : e.g. 'S01'  (matches EDF filename stem)
    sample_id             : e.g. 'S1'   (as in demographics CSV)
    label                 : 'Epileptic' or 'Non_Epileptic'
    label_binary          : 1 = Epileptic, 0 = Non_Epileptic
    gender                : 'M' or 'F'
    age                   : int
    sfreq_hz              : sampling frequency from EDF header
    duration_s            : total epoch duration in seconds
    n_channels_total      : all channels including EKG
    n_eeg_channels        : EEG channels only (EKG excluded)
    n_annotations_total   : all annotation markers in the file
    n_transient_markers   : markers with 'Annotation' in description
    annotation_status     : 'ok', 'missing', or 'multiple (N)'
    transient_onset_s     : time of sharp transient in seconds
    pre_transient_s       : seconds before transient (background window)
    post_transient_s      : seconds after transient (recovery window)
    frac_position         : transient onset / duration  (0=start, 1=end)
    all_annotation_labels : pipe-separated list of all annotation strings
    load_error            : None if ok, error string if file failed to load
"""

import os
import numpy as np
import pandas as pd
import mne

mne.set_log_level('WARNING')


def load_eeg_dataset(edf_dir: str, demographics_csv: str):
    """
    Parameters
    ----------
    edf_dir : str
        Path to folder containing all EDF files.
    demographics_csv : str
        Path to the demographics CSV with columns:
        Samples, Gender, Age, Gold Standard

    Returns
    -------
    summary_df : pd.DataFrame
        One row per EDF file. See module docstring for column descriptions.
    raws : dict
        Keys are file_id strings (e.g. 'S01'), values are MNE Raw objects.
        Files that failed to load are absent from this dict.
    """

    # --- load and clean demographics ---
    demo = pd.read_csv(demographics_csv)
    demo['file_id'] = demo['Samples'].apply(lambda s: 'S' + s[1:].zfill(2))
    demo = demo.rename(columns={
        'Samples': 'sample_id',
        'Gender': 'gender',
        'Age': 'age',
        'Gold Standard': 'label'
    })
    demo['label_binary'] = (demo['label'] == 'Epileptic').astype(int)

    records = []
    raws = {}

    edf_files = sorted([f for f in os.listdir(edf_dir) if f.endswith('.edf')])

    for fname in edf_files:
        file_id = fname.replace('.edf', '')
        fpath = os.path.join(edf_dir, fname)

        try:
            raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
        except Exception as e:
            records.append({'file_id': file_id, 'load_error': str(e)})
            continue

        sfreq = raw.info['sfreq']
        duration = raw.times[-1]
        eeg_chs = [ch for ch in raw.ch_names if 'EKG' not in ch]
        all_anns = list(raw.annotations)
        transient_anns = [a for a in all_anns if 'Annotation' in a['description']]
        n_transients = len(transient_anns)

        if n_transients == 1:
            t_onset = transient_anns[0]['onset']
            annotation_status = 'ok'
        elif n_transients == 0:
            t_onset = np.nan
            annotation_status = 'missing'
        else:
            t_onset = transient_anns[0]['onset']  # use first, flag it
            annotation_status = f'multiple ({n_transients})'

        pre = t_onset if not np.isnan(t_onset) else np.nan
        post = (duration - t_onset) if not np.isnan(t_onset) else np.nan
        frac = (t_onset / duration) if not np.isnan(t_onset) else np.nan

        records.append({
            'file_id': file_id,
            'sfreq_hz': sfreq,
            'duration_s': round(duration, 4),
            'n_channels_total': len(raw.ch_names),
            'n_eeg_channels': len(eeg_chs),
            'n_annotations_total': len(all_anns),
            'n_transient_markers': n_transients,
            'annotation_status': annotation_status,
            'transient_onset_s': round(t_onset, 4) if not np.isnan(t_onset) else np.nan,
            'pre_transient_s': round(pre, 4) if not np.isnan(pre) else np.nan,
            'post_transient_s': round(post, 4) if not np.isnan(post) else np.nan,
            'frac_position': round(frac, 4) if not np.isnan(frac) else np.nan,
            'all_annotation_labels': ' | '.join(a['description'] for a in all_anns),
            'load_error': None
        })

        raws[file_id] = raw

    summary_df = pd.DataFrame(records)

    summary_df = summary_df.merge(
        demo[['file_id', 'sample_id', 'gender', 'age', 'label', 'label_binary']],
        on='file_id',
        how='left'
    )

    col_order = [
        'file_id', 'sample_id', 'label', 'label_binary', 'gender', 'age',
        'sfreq_hz', 'duration_s', 'n_channels_total', 'n_eeg_channels',
        'n_annotations_total', 'n_transient_markers', 'annotation_status',
        'transient_onset_s', 'pre_transient_s', 'post_transient_s', 'frac_position',
        'all_annotation_labels', 'load_error'
    ]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    return summary_df, raws


if __name__ == '__main__':
    # example: point these at your local paths
    EDF_DIR = '/Users/kavya/Desktop/uni/research project/Kural EEG Dataset1/Recordings'
    DEMO_CSV = '/Users/kavya/Desktop/uni/research project/Kural EEG Dataset1/EEG data demographics - Sheet1.csv'

    df, raws = load_eeg_dataset(EDF_DIR, DEMO_CSV)

    print(f"Loaded {len(raws)} files successfully")
    print(f"DataFrame shape: {df.shape}")
    print(df.head())

    # check for any issues
    bad = df[df['annotation_status'] != 'ok']
    if len(bad) > 0:
        print(f"\nFiles with annotation issues:\n{bad[['file_id', 'label', 'annotation_status']]}")
    else:
        print("\nAll annotations look clean.")