# IED detection — pilot run

Scans EDF recordings for interictal epileptiform discharges and stores results in two CSVs.

## Install (once)

Needs **Python 3.11**.

    python3.11 -m venv venv
    venv/bin/pip install -r requirements.txt

On Windows, use `venv\Scripts\python.exe` wherever `venv/bin/python` appears below.

## Run

    venv/bin/python pilot.py

Enter the path of the folder containing the recordings in the terminal or command line and press enter. Results will be stored in 2 CSVs in a results folder. A file that fails is recorded in the CSV with its error and does not stop the run.

The montage must be **referential, not bipolar**, and contain all 19.

