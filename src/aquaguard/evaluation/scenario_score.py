import pandas as pd


def scenario_anomaly_score(residuals: pd.DataFrame) -> pd.Series:
    """
    Collapse per-node residuals into one scenario-level anomaly score per timestep:
    the most negative residual across all nodes at that moment. No leak-location
    knowledge is used - this only asks 'is anything unusual happening anywhere?'
    """
    return residuals.min(axis=1)