"""Detection preprocessing: recording-level split + full-recording multichannel loading.

Distinct from classify_loader (which windows tightly around the known marker). Detection needs
whole recordings across a fixed channel set, since the marker location is unknown at test time.
"""
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from detect_config import Config

# Standard 10-20 channels present in ALL 100 recordings. 3 (epileptic) recordings lack the
# inferior-temporal chain (F9/F10 T9/T10 P9/P10), so those are excluded for a consistent montage.
# ponytail: 19-channel common set; revisit if inferior-temporal coverage proves to matter for temporal IEDs.
CH19 = ['E C3-Ref', 'E C4-Ref', 'E Cz-Ref', 'E F3-Ref', 'E F4-Ref', 'E F7-Ref', 'E F8-Ref',
        'E FP1-Ref', 'E FP2-Ref', 'E Fz-Ref', 'E O1-Ref', 'E O2-Ref', 'E P3-Ref', 'E P4-Ref',
        'E P7-Ref', 'E P8-Ref', 'E Pz-Ref', 'E T7-Ref', 'E T8-Ref']

# Same channels under standard_1020 names, SAME ORDER (so a channel index means the same thing in both).
# Needed because interpolation is spatial: it requires electrode positions, and a montage is matched by name.
STD19 = [{'FP1': 'Fp1', 'FP2': 'Fp2'}.get(c[2:-4], c[2:-4]) for c in CH19]


def bad_channels(X, cfg):
    """Split channel indices into (soft, hard, over_cap) — see Config for the rationale of each threshold.

      soft  noisy enough to distort a common average, but still real data: dropped from the average
            reference computation, KEPT in the analysis.
      hard  broken: removed from the analysis and interpolated. Either wildly noisy (MAD >
            bad_hard_factor x the median across channels) or flat/disconnected (peak-to-peak < bad_flat_uv).

    over_cap: more than cfg.max_interpolate channels are hard-bad, so interpolating them would be fiction.
    The recording is NOT refused — all of them are demoted to soft (excluded from the average, left in the
    analysis) and the flag is reported, which returns a caveated number instead of nothing.

    Note the asymmetry: 'flat' is tested on peak-to-peak, NOT on MAD. A low MAD means a quiet baseline, not
    a dead channel — Kural's midline channels sit near its recording reference and have tiny MAD while
    carrying perfectly good signal, so a MAD-based floor would interpolate them away.
    """
    mad = np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1)
    ratio = mad / max(float(np.median(mad)), 1e-12)
    hard = set(np.where((ratio > cfg.bad_hard_factor) | (np.ptp(X, axis=1) < cfg.bad_flat_uv))[0].tolist())
    soft = set(np.where(ratio > cfg.bad_soft_factor)[0].tolist()) - hard
    over_cap = len(hard) > cfg.max_interpolate
    if over_cap:
        soft, hard = soft | hard, set()
    return sorted(soft), sorted(hard), over_cap


def load_recording_path(edf_path, cfg=None):
    """Full recording as (19, T) microvolt array over CH19, from an EDF at an arbitrary path.

    **This is the single preprocessing entry point** — every path into the cascade (load_recording,
    load_dataset, predict.py) comes through here, so anything applied to the signal before L1 belongs here
    and nowhere else. It is driven entirely by `cfg`, and `cfg` is what gets frozen into the saved model, so
    an external run cannot preprocess differently from training.

    Use this for external/test EDFs (e.g. the supervisor's data). The file must contain the 19 standard
    10-20 channels named as in Kural (`E Fp1-Ref`, ...); any sample rate is accepted and resampled to
    cfg.sfreq (clinical files are commonly 250/256/512 Hz).

    cfg.reference: 'recorded' (as-stored) | 'average' (common-average); 'bipolar' deferred.
    """
    cfg = cfg or Config()
    if cfg.reference == 'bipolar':
        raise NotImplementedError("bipolar montage not implemented yet")
    if cfg.reference not in ('recorded', 'average'):
        raise ValueError(f"unknown reference: {cfg.reference}")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick(CH19)                              # pick() also reorders to CH19, so indices are canonical
    raw.rename_channels(dict(zip(CH19, STD19)))
    raw.set_montage('standard_1020', match_case=False, verbose=False)   # positions, for interpolation
    if cfg.bandpass:                            # filter BEFORE resampling (mne's recommended order), so
        lo, hi = cfg.bandpass                   # the anti-aliasing step then has nothing left to remove
        raw.filter(lo, hi, verbose=False)       # zero-phase, so spike timing and shape are preserved
    if raw.info['sfreq'] != cfg.sfreq:          # guarded, so same-rate files are provably untouched
        raw.resample(cfg.sfreq, verbose=False)  # mne low-passes before decimating (anti-aliasing)

    soft, hard, over_cap = bad_channels(raw.get_data() * 1e6, cfg)
    if cfg.reference == 'average':              # average over the GOOD channels only, so a bad electrode
        keep = [STD19[i] for i in range(len(STD19)) if i not in set(soft) | set(hard)]
        raw.set_eeg_reference(ref_channels=keep, verbose=False)
    if hard:                                    # interpolate AFTER re-referencing, so the reconstruction is
        raw.info['bads'] = [STD19[i] for i in hard]   # consistent with the channels it is built from
        raw.interpolate_bads(reset_bads=False, verbose=False)
    return raw.get_data() * 1e6, {'soft': soft, 'hard': hard, 'over_cap': over_cap}


def load_recording(file_id, edf_dir, cfg=None):
    """(X, bads) for a `file_id` under `edf_dir`. See load_recording_path."""
    return load_recording_path(Path(edf_dir) / f"{file_id}.edf", cfg)


def load_dataset(manifest, edf_dir, cfg=None):
    """List of per-recording dicts: fid, X(19,T), mk (marker sample), epi (bool), cert, dur (s).

    `mk` and `dur` are derived from cfg.sfreq, NOT from a hard-coded rate — they used to assume 500 Hz, so
    resampling without this change would put every ground-truth marker at the wrong sample and look exactly
    like "preprocessing destroyed the signal".
    """
    cfg = cfg or Config()
    out = []
    for _, r in manifest.iterrows():
        X, bads = load_recording(r['file_id'], edf_dir, cfg)
        out.append(dict(fid=r['file_id'], X=X, mk=int(round(r['transient_onset_s'] * cfg.sfreq)),
                        epi=bool(r['label_binary']), cert=r['certainty'], dur=X.shape[1] / cfg.sfreq,
                        bad_soft=bads['soft'], bad_hard=bads['hard'], bad_over_cap=bads['over_cap']))
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
    from dataclasses import replace
    cfg = Config()
    fid0 = te['file_id'].iloc[0]
    X, bads = load_recording(fid0, rec_dir, cfg)
    assert X.shape[0] == 19, "expected 19 channels"
    assert set(bads) == {'soft', 'hard', 'over_cap'}, "loader must report the bad-channel split"

    # bad_channels: the two thresholds, the flat test, and the over-cap fallback. Synthetic so every
    # branch is exercised — on Kural itself nothing is hard-bad, so none of this runs on real data.
    rng = np.random.default_rng(0)
    B = rng.normal(0, 8, (19, 4000))
    B[3] *= 4.0                       # noisy: ~4x -> soft only
    B[7] *= 20.0                      # wildly noisy -> hard
    B[11] = 1e-4                      # flat/disconnected -> hard (ptp test, not MAD)
    soft, hard, over = bad_channels(B, cfg)
    assert soft == [3], f"expected channel 3 soft, got {soft}"
    assert hard == [7, 11], f"expected 7 (noisy) and 11 (flat) hard, got {hard}"
    assert not over, "2 hard channels is at the cap, not over it"
    B2 = B.copy(); B2[15] = 1e-4      # a third hard channel -> over the cap
    soft2, hard2, over2 = bad_channels(B2, cfg)
    assert over2 and hard2 == [] and set(soft2) == {3, 7, 11, 15}, \
        f"over the cap: interpolate none, demote all to soft; got {soft2}/{hard2}/{over2}"
    # a quiet-but-live channel must NOT be called dead: low MAD, healthy peak-to-peak (the Kural midline case)
    B3 = B.copy(); B3[5] = rng.normal(0, 8, 4000) * 0.02
    assert 5 not in bad_channels(B3, cfg)[1], "a quiet channel with real range is not a dead channel"

    # average reference is taken over the GOOD channels only, so it zero-sums over exactly those
    Xa, ba = load_recording(fid0, rec_dir, replace(cfg, reference='average'))
    keep = [i for i in range(19) if i not in set(ba['soft']) | set(ba['hard'])]
    assert np.allclose(Xa[keep].mean(0), 0, atol=1e-9), "average must zero-sum over the unflagged channels"

    # preprocessing is driven by cfg alone. Written rate-explicitly so it does not silently pass or fail
    # when the DEFAULT sfreq changes (it went 500 -> 250 in Phase 1).
    fid = te['file_id'].iloc[0]
    ref = replace(cfg, reference='recorded')      # isolate rate/filter effects from the reference
    raw500 = load_recording(fid, rec_dir, replace(ref, sfreq=500.0, bandpass=None))[0]  # native, untouched
    bp500 = load_recording(fid, rec_dir, replace(ref, sfreq=500.0))[0]                  # filtered only
    bp250 = load_recording(fid, rec_dir, replace(ref, sfreq=250.0))[0]                  # filtered+resampled
    assert bp500.shape == raw500.shape, "filtering must not change the sample count"
    assert bp250.shape[1] == raw500.shape[1] // 2, "resampling to half the rate should halve the samples"
    assert not np.allclose(bp500, raw500), "the bandpass should actually change the signal"

    # the bandpass must remove what it says it removes, and keep what it says it keeps
    lo, hi = cfg.bandpass
    f = np.fft.rfftfreq(raw500.shape[1], 1.0 / 500.0)
    pw = lambda A, a, b: float((np.abs(np.fft.rfft(A, axis=1))[:, (f >= a) & (f < b)] ** 2).sum())
    assert pw(bp500, hi + 15, 250) < 0.02 * pw(raw500, hi + 15, 250), "stopband not attenuated"
    assert pw(bp500, 4, 30) > 0.7 * pw(raw500, 4, 30), "passband should be largely preserved"

    assert window_around(np.arange(100), 5, 10) is None, "edge window should be rejected"
    assert len(window_around(np.arange(100), 50, 10)) == 20, "window length wrong"

    print(f"selfcheck OK  |  train={len(tr)} test={len(te)}  |  "
          f"test certainty={te['certainty'].value_counts().to_dict()}")


if __name__ == '__main__':
    _selfcheck()
