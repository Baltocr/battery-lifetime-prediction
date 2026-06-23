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

def extract_delta_q_features_for_cell(h5_file, batch, cell_index, cycle_early=10, cycle_late=100):
    """
    Extract Delta Q features using Qdlin from two early cycles.

    Delta Q = Qdlin(cycle_late) - Qdlin(cycle_early)

    cycle_early and cycle_late are written using human cycle numbers,
    so cycle 10 corresponds to Python index 9.
    """
    cycles_refs = batch["cycles"]
    ref = cycles_refs[cell_index, 0]
    cycles = h5_file[ref]

    early_index = cycle_early - 1
    late_index = cycle_late - 1

    qdlin_early_ref = cycles["Qdlin"][early_index, 0]
    qdlin_late_ref = cycles["Qdlin"][late_index, 0]

    qdlin_early = h5_file[qdlin_early_ref][()].squeeze()
    qdlin_late = h5_file[qdlin_late_ref][()].squeeze()

    delta_q = qdlin_late - qdlin_early

    features = {
        f"delta_q_min_{cycle_late}_{cycle_early}": float(np.min(delta_q)),
        f"delta_q_max_{cycle_late}_{cycle_early}": float(np.max(delta_q)),
        f"delta_q_mean_{cycle_late}_{cycle_early}": float(np.mean(delta_q)),
        f"delta_q_var_{cycle_late}_{cycle_early}": float(np.var(delta_q)),
    }

    return features









