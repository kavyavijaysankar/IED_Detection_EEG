"""Detection preprocessing: recording-level split + full-recording multichannel loading.
"""
import re
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from detect_config import Config

# The canonical channel index space, in standard_1020 spelling. The first N_REQUIRED are the standard
# 10-20 array and every recording must have them; the rest are the extended inferior-temporal chain,
# which a file may simply not carry (3 Kural recordings do not). An absent electrode is left as a zero
# row rather than reconstructed — see load_recording_path.
STD_NAMES = ['C3', 'C4', 'Cz', 'F3', 'F4', 'F7', 'F8', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2',
             'P3', 'P4', 'P7', 'P8', 'Pz', 'T7', 'T8',
             'F9', 'F10', 'P9', 'P10', 'T9', 'T10']
N_REQUIRED = 19

_ALIAS = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'} # old 10-20 nomenclature


def electrode(name):
    """EDF channel name -> canonical electrode label ('EEG Fp1-REF', 'E FP1-Ref', 'Fp1-A1' -> 'FP1')."""
    s = re.split(r'[-_]', name.upper().replace(' ', ''))[0]
    s = re.sub(r'^(EEG|ECG|EKG|E)', '', s)
    return _ALIAS.get(s, s)


ELECTRODES = [n.upper() for n in STD_NAMES]

# left/right homologous pairs as ELECTRODES indices; the midline (Fz, Cz, Pz) has no mirror
HOMOLOGOUS = [(ELECTRODES.index(l), ELECTRODES.index(r)) for l, r in
              (('FP1', 'FP2'), ('F3', 'F4'), ('F7', 'F8'), ('C3', 'C4'),
               ('P3', 'P4'), ('P7', 'P8'), ('T7', 'T8'), ('O1', 'O2'),
               ('F9', 'F10'), ('P9', 'P10'), ('T9', 'T10'))]


def _is_bipolar(name):
    """True for a bipolar derivation ('F7-T3'), False for a referential channel ('Fp1-A1')."""
    parts = re.split(r'[-_]', name.upper().replace(' ', ''))
    return len(parts) > 1 and _ALIAS.get(parts[1], parts[1]) in ELECTRODES


def match_channels(ch_names):
    """(the file's own names for the electrodes it carries, their ELECTRODES indices), both ascending.

    The standard 19 are required; the extended electrodes are optional and simply absent when the file
    has none. Raises on a missing standard electrode, an ambiguous name or a bipolar montage.
    """
    found = {}
    for n in ch_names:
        e = electrode(n)
        if e in ELECTRODES:
            found.setdefault(e, []).append(n)
    bipolar = [n for v in found.values() for n in v if _is_bipolar(n)]
    if bipolar:
        raise ValueError(f"bipolar montage ({bipolar[:4]}...); the model needs referential channels and "
                         f"re-references them itself. Please export referential.")
    dup = {e: v for e, v in found.items() if len(v) > 1}
    if dup:
        raise ValueError(f"ambiguous channels — one electrode named twice, possibly under different "
                         f"references: {dup}")
    missing = [e for e in ELECTRODES[:N_REQUIRED] if e not in found]
    if missing:
        raise ValueError(f"missing electrodes {missing}. File contains: {list(ch_names)}")
    present = [i for i, e in enumerate(ELECTRODES) if e in found]
    return [found[ELECTRODES[i]][0] for i in present], present


def electrode_positions():
    """(n, 3) electrode coordinates in mm, in ELECTRODES order, from the standard_1020 montage."""
    pos = mne.channels.make_standard_montage('standard_1020').get_positions()['ch_pos']
    return np.array([pos[c] for c in STD_NAMES]) * 1000.0


def bad_channels(X, cfg):
    """Split channel indices into (soft, hard, over_cap). Soft - noisy enough to distort a common average, but still real data: dropped from the average reference computation, kept in the analysis. Hard: removed from the analysis and interpolated.

    over_cap: more than cfg.max_interpolate channels are hard, so none are interpolated and all are treated as soft.
    """
    mad = np.median(np.abs(X - np.median(X, axis=1, keepdims=True)), axis=1)
    ratio = mad / max(float(np.median(mad)), 1e-12)
    hard = set(np.where((ratio > cfg.bad_hard_factor) | (np.ptp(X, axis=1) < cfg.bad_flat_uv))[0].tolist())
    soft = set(np.where(ratio > cfg.bad_soft_factor)[0].tolist()) - hard
    over_cap = len(hard) > cfg.max_interpolate
    if over_cap:
        soft, hard = soft | hard, set()
    return sorted(soft), sorted(hard), over_cap, ratio


def load_recording_path(edf_path, cfg=None):
    cfg = cfg or Config()
    if cfg.reference == 'bipolar':
        raise NotImplementedError("bipolar montage not implemented yet")
    if cfg.reference not in ('recorded', 'average'):
        raise ValueError(f"unknown reference: {cfg.reference}")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    names, present = match_channels(raw.ch_names)
    raw.pick(names) # pick() also reorders, so indices are canonical
    std = [STD_NAMES[i] for i in present]
    raw.rename_channels(dict(zip(raw.ch_names, std)))
    assert raw.ch_names == std, f"channel order not canonical after pick: {raw.ch_names}"
    raw.set_montage('standard_1020', match_case=False, verbose=False)   # positions, for interpolation
    if cfg.bandpass: # filter BEFORE resampling (mne's recommended order), so
        lo, hi = cfg.bandpass # the anti-aliasing step then has nothing left to remove
        raw.filter(lo, hi, verbose=False) # zero-phase, so spike timing and shape are preserved
    if raw.info['sfreq'] != cfg.sfreq: # guarded, so same-rate files are provably untouched
        raw.resample(cfg.sfreq, verbose=False) # mne low-passes before decimating (anti-aliasing)

    soft, hard, over_cap, ratio = bad_channels(raw.get_data() * 1e6, cfg)   # indices into `present`
    if cfg.reference == 'average': # average over the GOOD channels only, so a bad electrode
        keep = [std[i] for i in range(len(std)) if i not in set(soft) | set(hard)]
        raw.set_eeg_reference(ref_channels=keep, verbose=False)
    if hard: # interpolate after re-referencing, so the reconstruction is
        raw.info['bads'] = [std[i] for i in hard] # consistent with the channels it is built from
        raw.interpolate_bads(reset_bads=False, verbose=False)

    # Expand to the canonical index space. An electrode the file does not carry stays a ZERO row: it is
    # never reconstructed, and a flat channel yields no L1 candidates (sharpness 0 < l1_threshold), so it
    # cannot join an event, satisfy min_channels or enter a spatial feature. Same guarantee as an
    # interpolated channel, with no plumbing — the only place that has to know is background().
    X = np.zeros((len(ELECTRODES), raw.n_times))
    X[present] = raw.get_data() * 1e6
    mad_ratio = np.full(len(ELECTRODES), np.nan)
    mad_ratio[present] = ratio
    return X, {'soft': [present[i] for i in soft], 'hard': [present[i] for i in hard],
               'over_cap': over_cap, 'mad_ratio': mad_ratio,
               'absent': sorted(set(range(len(ELECTRODES))) - set(present))}


def load_recording(file_id, edf_dir, cfg=None):
    """(X, bads) for a `file_id` under `edf_dir`. See load_recording_path."""
    return load_recording_path(Path(edf_dir) / f"{file_id}.edf", cfg)


def load_dataset(manifest, edf_dir, cfg=None):
    cfg = cfg or Config()
    out = []
    for _, r in manifest.iterrows():
        X, bads = load_recording(r['file_id'], edf_dir, cfg)
        out.append(dict(fid=r['file_id'], X=X, mk=int(round(r['transient_onset_s'] * cfg.sfreq)),
                        epi=bool(r['label_binary']), cert=r['certainty'], dur=X.shape[1] / cfg.sfreq,
                        bad_soft=bads['soft'], bad_hard=bads['hard'], bad_over_cap=bads['over_cap'],
                        absent=bads['absent']))
    return out


def stratified_split(manifest, n_test=10, seed=0):
    """Recording-level train/test split, stratified on certainty.
    Returns (train_df, test_df); n_test=0 puts everything in train and returns an empty test frame.
    """
    if n_test == 0:
        return manifest.reset_index(drop=True), manifest.iloc[0:0].reset_index(drop=True)
    train, test = train_test_split(manifest, test_size=n_test, random_state=seed,
                                   stratify=manifest['certainty'])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def window_around(x, center, half):
    """Leakage-safe fixed window x[center-half : center+half]; None if it would exceed bounds.
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
    tr0, te0 = stratified_split(man, n_test=0, seed=0)
    assert len(tr0) == len(man) and len(te0) == 0, "n_test=0 should train on everything"

    assert electrode('E FP1-Ref') == electrode('EEG Fp1-REF') == electrode('Fp1-A1') == 'FP1'
    assert electrode('EEG T3-LE') == 'T7' and electrode('T6') == 'P8', "old 10-20 names must alias"
    assert electrode('FC3') == 'FC3' and electrode('TP7') == 'TP7', "extended array must not collide"
    assert electrode('P EKG') not in ELECTRODES, "a non-EEG channel must match no electrode"
    kural = [f"E {e.replace('FP', 'Fp').replace('Z', 'z')}-Ref" for e in ELECTRODES]
    assert match_channels(kural) == (kural, list(range(len(ELECTRODES)))), \
        "Kural's own names must still match, in order"
    old = {'T7': 'T3', 'T8': 'T4', 'P7': 'T5', 'P8': 'T6'}
    clinical = [f"EEG {old.get(e, e).title()}-REF" for e in ELECTRODES]
    assert match_channels(list(reversed(clinical)) + ['P EKG', 'EEG A1-REF']) == \
        (clinical, list(range(len(ELECTRODES)))), \
        "must return every electrode in canonical order whatever the file's order, ignoring extras"
    assert _is_bipolar('F7-T3') and _is_bipolar('EEG C3-Cz'), "bipolar derivations must be spotted"
    assert not _is_bipolar('Fp1-A1') and not _is_bipolar('E FP1-Ref'), "referential ones must not be"
    chain = [f"{a}-{b}" for a, b in zip(ELECTRODES, ELECTRODES[1:] + ELECTRODES[:1])]
    for bad_list, word in ((clinical[1:], 'missing'), (clinical + ['Fp1-A2'], 'ambiguous'),
                           (chain, 'bipolar')):
        try:
            match_channels(bad_list)
            raise AssertionError(f"{word} electrode should raise")
        except ValueError as ex:
            assert word in str(ex), f"unhelpful {word} error: {ex}"

    # the extended electrodes are OPTIONAL: a file without them loads, one without a standard 10-20
    # electrode is still refused. This is what lets the 3 Kural recordings that lack them through.
    core = clinical[:N_REQUIRED]
    assert match_channels(core) == (core, list(range(N_REQUIRED))), \
        "a standard-19 file must load, with the extended electrodes simply absent"
    for one in ELECTRODES[N_REQUIRED:]:
        names = [n for n in clinical if electrode(n) != one]
        assert len(match_channels(names)[1]) == len(ELECTRODES) - 1, f"{one} should be optional"

    rec_dir = root / 'Kural_Dataset' / 'Recordings'
    from dataclasses import replace
    cfg = Config()
    fid0 = te['file_id'].iloc[0]
    X, bads = load_recording(fid0, rec_dir, cfg)
    assert X.shape[0] == len(ELECTRODES), f"expected {len(ELECTRODES)} channels, got {X.shape[0]}"
    assert set(bads) == {'soft', 'hard', 'over_cap', 'mad_ratio', 'absent'}, \
        "loader must report the bad-channel split and which electrodes the file lacked"

    rng = np.random.default_rng(0)
    B = rng.normal(0, 8, (19, 4000))
    B[3] *= 4.0                       # noisy: ~4x -> soft only
    B[7] *= 20.0                      # wildly noisy -> hard
    B[11] = 1e-4                      # flat/disconnected -> hard (ptp test, not MAD)
    soft, hard, over, ratio = bad_channels(B, cfg)
    assert soft == [3], f"expected channel 3 soft, got {soft}"
    assert hard == [7, 11], f"expected 7 (noisy) and 11 (flat) hard, got {hard}"
    assert not over, "2 hard channels is at the cap, not over it"
    assert len(ratio) == 19 and ratio[7] > cfg.bad_hard_factor > ratio[3] > cfg.bad_soft_factor, \
        "mad_ratio must report each channel's MAD against the median"
    B2 = B.copy(); B2[15] = 1e-4      # a third hard channel -> over the cap
    soft2, hard2, over2, _ = bad_channels(B2, cfg)
    assert over2 and hard2 == [] and set(soft2) == {3, 7, 11, 15}, \
        f"over the cap: interpolate none, demote all to soft; got {soft2}/{hard2}/{over2}"
    # a quiet-but-live channel must NOT be called dead: low MAD, healthy peak-to-peak (the Kural midline case)
    B3 = B.copy(); B3[5] = rng.normal(0, 8, 4000) * 0.02
    assert 5 not in bad_channels(B3, cfg)[1], "a quiet channel with real range is not a dead channel"

    # average reference is taken over the good channels only, so it zero-sums over exactly those
    Xa, ba = load_recording(fid0, rec_dir, replace(cfg, reference='average'))
    keep = [i for i in range(len(ELECTRODES))
            if i not in set(ba['soft']) | set(ba['hard']) | set(ba['absent'])]
    assert np.allclose(Xa[keep].mean(0), 0, atol=1e-9), "average must zero-sum over the unflagged channels"

    # preprocessing is driven by cfg alone. Written rate-explicitly so it does not silently pass or fail
    # when the default sfreq changes (it went 500 -> 250 in Phase 1).
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

    pos = electrode_positions()
    ix = {e: i for i, e in enumerate(ELECTRODES)}
    d = lambda a, b: float(np.linalg.norm(pos[ix[a]] - pos[ix[b]]))
    assert pos.shape == (len(ELECTRODES), 3), f"expected {len(ELECTRODES)} 3-D electrode positions"
    assert d('FP1', 'O1') > d('FP1', 'F3'), "front-to-back must exceed a neighbouring pair"
    assert abs(d('C3', 'CZ') - d('C4', 'CZ')) < 3.0, "left/right must mirror (the template is not exact)"
    assert 20 < d('F7', 'T7') < 90, f"adjacent electrodes should be tens of mm apart, got {d('F7','T7'):.0f}"
    assert d('T9', 'T7') < d('T9', 'T8'), "T9 sits below T7, not across the head"
    assert len(HOMOLOGOUS) == 11 and len({i for p in HOMOLOGOUS for i in p}) == 22, \
        "11 disjoint L/R pairs (8 standard + the 3 extended)"
    for li, ri in HOMOLOGOUS:
        assert pos[li][0] < 0 < pos[ri][0], \
            f"{ELECTRODES[li]}/{ELECTRODES[ri]} are not a left/right pair"
        assert abs(abs(pos[li][0]) - abs(pos[ri][0])) < 8.0, "homologues should mirror across the midline"

    # A file missing extended electrodes loads with zero rows in their place: never reconstructed, and
    # inert downstream because a flat channel's sharpness is 0 and so yields no L1 candidate.
    from detect_stage1 import channel_stat
    short = [f for f in man['file_id']
             if len(match_channels(mne.io.read_raw_edf(str(rec_dir / f"{f}.edf"), preload=False,
                                                       verbose=False).ch_names)[1]) < len(ELECTRODES)]
    assert short, "expected some Kural recordings to lack the extended electrodes"
    Xs_, bs_ = load_recording(short[0], rec_dir, cfg)
    assert bs_['absent'] and not (set(bs_['absent']) & set(bs_['hard'])), \
        "an absent electrode must never be interpolated"
    assert np.all(Xs_[bs_['absent']] == 0), "absent electrodes must be left as zero rows"
    assert channel_stat(Xs_, cfg)[bs_['absent']].max() < cfg.l1_threshold, \
        "a zero row must never reach the L1 threshold"
    present = [i for i in range(len(ELECTRODES)) if i not in set(bs_['absent'])]
    from detection import DetectionPipeline
    assert DetectionPipeline.background(Xs_) == DetectionPipeline.background(Xs_[present]), \
        "background amplitude must ignore the absent channels, not average them in"

    assert window_around(np.arange(100), 5, 10) is None, "edge window should be rejected"
    assert len(window_around(np.arange(100), 50, 10)) == 20, "window length wrong"

    print(f"selfcheck OK  |  train={len(tr)} test={len(te)}  |  "
          f"{len(ELECTRODES)} electrodes, {len(short)} recordings missing some  |  "
          f"test certainty={te['certainty'].value_counts().to_dict()}")


if __name__ == '__main__':
    _selfcheck()
