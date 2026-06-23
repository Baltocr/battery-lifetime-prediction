from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import h5py



def plot_cycle_life_distribution(input_csv="data/processed/cell_cycle_lives.csv"):
    """
    Plot the distribution of battery cycle lives.
    """
    df = pd.read_csv(input_csv)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(df["cycle_life"], bins=20)
    ax.set_title("Distribution of Battery Cycle Life")
    ax.set_xlabel("Cycle Life")
    ax.set_ylabel("Number of Cells")

    output_path = Path("reports/figures/cycle_life_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")

def plot_delta_q_curve(raw_file="data/raw/2017-05-12_batchdata_updated_struct_errorcorrect.mat"):
    """
    Plot Delta Q curve for cell 0 between cycle 10 and cycle 100.
    """
    with h5py.File(raw_file, "r") as f:
        batch = f["batch"]
        cycles_refs = batch["cycles"]

        cell_index = 0
        cycle_10_index = 9
        cycle_100_index = 99

        cycles = f[cycles_refs[cell_index, 0]]

        qdlin_10 = f[cycles["Qdlin"][cycle_10_index, 0]][()].squeeze()
        qdlin_100 = f[cycles["Qdlin"][cycle_100_index, 0]][()].squeeze()

        delta_q = qdlin_100 - qdlin_10

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(delta_q)
    ax.axhline(0, linestyle="--")

    ax.set_title("Delta Q Curve: Cycle 100 - Cycle 10")
    ax.set_xlabel("Interpolated Voltage Index")
    ax.set_ylabel("Delta Q")

    output_path = Path("reports/figures/delta_q_curve_cell_0.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved Delta Q curve to: {output_path}")


if __name__ == "__main__":
    plot_cycle_life_distribution()
    plot_delta_q_curve()