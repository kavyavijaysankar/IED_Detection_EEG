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

Sub-folders are searched too, so you can point it at one folder containing everything. Expect roughly a minute per 20-minute recording — a few hundred recordings is an overnight job. Results are written as it goes, so nothing is lost if you stop it.

**If a long run is interrupted**, start it again the same way but add `--resume`:

    venv/bin/python pilot.py --resume

It reads what is already in the CSVs and carries on from where it stopped instead of starting over.

The montage must be **referential, not bipolar**, and contain all 19.

