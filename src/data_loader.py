from pathlib import Path
import h5py
import numpy as np

from features import extract_summary_features_for_cell, extract_delta_q_features_for_cell


def _read_scalar(dataset):
    """Read a scalar or 1-element dataset from HDF5 and return a Python scalar."""
    value = dataset[()]
    if isinstance(value, np.ndarray):
        value = value.squeeze()
    return value

def get_raw_data_files(raw_dir="data/raw"):
    """
    Return all .mat files inside the raw data folder.
    """
    raw_path = Path(raw_dir)

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data folder not found: {raw_path}")

    mat_files = sorted(raw_path.glob("*.mat"))

    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {raw_path}")

    return mat_files

def load_cycle_lives(file_path):
    """
    Load clean cycle life values for all valid cells in one batch file.
    Skips cells with missing cycle life values.

    Returns a list of dictionaries with batch file, cell index, and cycle life.
    """
    records = []

    with h5py.File(file_path, "r") as f:
        batch = f["batch"]
        cycle_life_refs = batch["cycle_life"]

        for i in range(cycle_life_refs.shape[0]):
            ref = cycle_life_refs[i, 0]
            value = _read_scalar(f[ref])
            clean_value = float(value)

            if np.isnan(clean_value):
                print(f"Skipping cell {i} in {file_path.name}: cycle life is NaN")
                continue

            records.append({
                "batch_file": file_path.name,
                "cell_index": i,
                "cycle_life": int(clean_value),
            })

    return records


def inspect_mat_file(file_path):
    """
    Read and print clean cycle life values for the first few cells.
    """
    print(f"\nInspecting: {file_path.name}")

    with h5py.File(file_path, "r") as f:
        batch = f["batch"]
        cycle_life_refs = batch["cycle_life"]

        print(f"\nNumber of cells in this batch: {cycle_life_refs.shape[0]}")

        print("\nFirst 10 clean cycle life values:")

        for i in range(min(10, cycle_life_refs.shape[0])):
            ref = cycle_life_refs[i, 0]
            value = _read_scalar(f[ref])
            clean_value = int(value)

            print(f"Cell {i}: {clean_value} cycles")

if __name__ == "__main__":
    import pandas as pd

    files = get_raw_data_files()

    all_records = []

    for file in files:
        print(f"Processing {file.name}...")

        with h5py.File(file, "r") as f:
            batch = f["batch"]
            cycle_life_refs = batch["cycle_life"]

            batch_records = []

            for cell_index in range(cycle_life_refs.shape[0]):
                ref = cycle_life_refs[cell_index, 0]
                value = f[ref][()]
                cycle_life = float(value.squeeze())

                if np.isnan(cycle_life):
                    print(f"Skipping cell {cell_index}: cycle life is NaN")
                    continue

                summary_features = extract_summary_features_for_cell(
                    h5_file=f,
                    batch=batch,
                    cell_index=cell_index
                )

                delta_q_features = extract_delta_q_features_for_cell(
                    h5_file=f,
                    batch=batch,
                    cell_index=cell_index,
                    cycle_early=10,
                    cycle_late=100
                )

                features = {}
                features.update(summary_features)
                features.update(delta_q_features)

                record = {
                    "batch_file": file.name,
                    "cell_index": cell_index,
                    "cycle_life": int(cycle_life),
                }

                record.update(features)
                batch_records.append(record)

            print(f"Valid cells from this batch: {len(batch_records)}")
            all_records.extend(batch_records)

    df = pd.DataFrame(all_records)

    print("\nFinal feature table preview:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    output_path = Path("data/processed/summary_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nSaved summary feature table to: {output_path}")








