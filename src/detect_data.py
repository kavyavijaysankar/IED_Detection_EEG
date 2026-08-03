"""Detection preprocessing: recording-level split + full-recording multichannel loading.

Distinct from classify_loader (which windows tightly around the known marker). Detection needs
whole recordings across a fixed channel set, since the marker location is unknown at test time.
"""
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SF = 500.0

# Standard 10-20 channels present in ALL 100 recordings. 3 (epileptic) recordings lack the
# inferior-temporal chain (F9/F10 T9/T10 P9/P10), so those are excluded for a consistent montage.
# ponytail: 19-channel common set; revisit if inferior-temporal coverage proves to matter for temporal IEDs.
CH19 = ['E C3-Ref', 'E C4-Ref', 'E Cz-Ref', 'E F3-Ref', 'E F4-Ref', 'E F7-Ref', 'E F8-Ref',
        'E FP1-Ref', 'E FP2-Ref', 'E Fz-Ref', 'E O1-Ref', 'E O2-Ref', 'E P3-Ref', 'E P4-Ref',
        'E P7-Ref', 'E P8-Ref', 'E Pz-Ref', 'E T7-Ref', 'E T8-Ref']


def load_recording_path(edf_path, reference='recorded'):
    """Full recording as (19, T) microvolt array over CH19, from an EDF at an arbitrary path.

    Use this for external/test EDFs (e.g. the supervisor's data). The file must contain the 19 standard
    10-20 channels named as in Kural (`E Fp1-Ref`, ...) at 500 Hz.
    reference: 'recorded' (as-stored) | 'average' (common-average); 'bipolar' deferred.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    assert raw.info['sfreq'] == SF, f"{edf_path}: expected {SF} Hz, got {raw.info['sfreq']}"
    raw.pick(CH19)
    X = raw.get_data() * 1e6
    if reference == 'average':
        X = X - X.mean(axis=0, keepdims=True)
    elif reference == 'bipolar':
        raise NotImplementedError("bipolar montage not implemented yet")
    elif reference != 'recorded':
        raise ValueError(f"unknown reference: {reference}")
    return X


def load_recording(file_id, edf_dir, reference='recorded'):
    """Full recording (19, T) µV for a `file_id` under `edf_dir`. See load_recording_path."""
    return load_recording_path(Path(edf_dir) / f"{file_id}.edf", reference)


def load_dataset(manifest, edf_dir, reference='recorded'):
    """List of per-recording dicts: fid, X(19,T), mk (marker sample), epi (bool), cert, dur (s)."""
    out = []
    for _, r in manifest.iterrows():
        X = load_recording(r['file_id'], edf_dir, reference)
        out.append(dict(fid=r['file_id'], X=X, mk=int(round(r['transient_onset_s'] * SF)),
                        epi=bool(r['label_binary']), cert=r['certainty'], dur=X.shape[1] / SF))
    return out


def stratified_split(manifest, n_test=10, seed=0):
    """Recording-level train/test split, stratified on `certainty`.

    `certainty` (negative / unclear_positive / clear_positive) encodes both epileptic-vs-not and the
    clear/unclear sub-split, so stratifying on it covers both. Whole recordings go in or out.
    Returns (train_df, test_df).
    """
    train, test = train_test_split(manifest, test_size=n_test, random_state=seed,
                                   stratify=manifest['certainty'])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def window_around(x, center, half):
    """Leakage-safe fixed window x[center-half : center+half]; None if it would exceed bounds.

    Enforces the plan's rule that events within `half` samples of a recording edge are undetectable.
    """
    lo, hi = center - half, center + half
    if lo < 0 or hi > len(x):
        return None
    return x[lo:hi]


def _selfcheck():
    root = Path(__file__).resolve().parent.parent
    man = pd.read_csv(root / 'Kural_Dataset' / 'eeg_summary.csv')
    man = man[man['load_error'].isna() & (man['annotation_status'] == 'ok')].reset_index(drop=True)

    tr, te = stratified_split(man, n_test=10, seed=0)
    assert len(te) == 10 and len(tr) == len(man) - 10, "wrong split sizes"
    assert not (set(tr['file_id']) & set(te['file_id'])), "train/test overlap"
    tr2, te2 = stratified_split(man, n_test=10, seed=0)
    assert list(te['file_id']) == list(te2['file_id']), "split not deterministic"

    rec_dir = root / 'Kural_Dataset' / 'Recordings'
    X = load_recording(te['file_id'].iloc[0], rec_dir)
    assert X.shape[0] == 19, "expected 19 channels"
    Xa = load_recording(te['file_id'].iloc[0], rec_dir, 'average')
    assert np.allclose(Xa.mean(0), 0, atol=1e-9), "average reference must zero-sum per sample"

    assert window_around(np.arange(100), 5, 10) is None, "edge window should be rejected"
    assert len(window_around(np.arange(100), 50, 10)) == 20, "window length wrong"

    print(f"selfcheck OK  |  train={len(tr)} test={len(te)}  |  "
          f"test certainty={te['certainty'].value_counts().to_dict()}")


if __name__ == '__main__':
    _selfcheck()
