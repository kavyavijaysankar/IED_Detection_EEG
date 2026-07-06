# Epilepsy-EEG-FDA

### Dataset — `Kural_Dataset/`

- `Recordings/` — S01.edf … S100.edf
- `Demographics.csv` - file number, sex, age, gold standard (epileptic, non-epileptic)
- `eeg_summary.csv` - metadata, transient and signal properties, extra info, etc added through EDA. Also used in classification notebooks.
- `Supplementary_material_1–6.pdf` 

### Source — `src/`

- `EEG_loader.py` - used for EDA
- `classify_loader.py` - used for classification

### Notebooks — `notebooks/`

- `EDA.ipynb` - exploratory data analysis
- `classification_registered.ipynb` - 2s window classification with registration 
- `classification.ipynb` - larger window classification without registration

### `Scrap/` - trial and error

### Misc

- `imagestyle.mplstyle`
- `figures/` - not all saved yet.
