from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


if __name__ == "__main__":
    plot_cycle_life_distribution()