import h5py
import numpy as np


def extract_summary_features_for_cell(h5_file, batch, cell_index):
    """
    Extract simple early-cycle features from the summary data for one cell.

    Uses only cycles up to cycle 100 to avoid data leakage.
    """
    summary_refs = batch["summary"]
    ref = summary_refs[cell_index, 0]
    summary = h5_file[ref]

    q_discharge = summary["QDischarge"][()].squeeze()
    ir = summary["IR"][()].squeeze()
    charge_time = summary["chargetime"][()].squeeze()
    tavg = summary["Tavg"][()].squeeze()
    tmax = summary["Tmax"][()].squeeze()

    features = {
        "q_discharge_cycle_2": q_discharge[1],
        "q_discharge_cycle_100": q_discharge[99],
        "capacity_fade_2_to_100": q_discharge[1] - q_discharge[99],
        "capacity_retention_100": q_discharge[99] / q_discharge[1],
        "ir_cycle_2": ir[1],
        "ir_cycle_100": ir[99],
        "ir_change_2_to_100": ir[99] - ir[1],
        "charge_time_cycle_2": charge_time[1],
        "tavg_cycle_100": tavg[99],
        "tmax_cycle_100": tmax[99],
    }

    return features