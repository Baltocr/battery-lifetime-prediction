from pathlib import Path
import h5py
import numpy as np

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
            value = f[ref][()]
            clean_value = float(value.squeeze())

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
            value = f[ref][()]
            clean_value = int(value.squeeze())

            print(f"Cell {i}: {clean_value} cycles")

if __name__ == "__main__":
    import pandas as pd

    files = get_raw_data_files()

    all_records = []

    for file in files:
        records = load_cycle_lives(file)
        all_records.extend(records)

        print(f"{file.name}: {len(records)} valid cells")

    df = pd.DataFrame(all_records)

    print("\nDataset preview:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    print("\nCycle life summary:")
    print(df["cycle_life"].describe())

    output_path = Path("data/processed/cell_cycle_lives.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nSaved cycle life table to: {output_path}")