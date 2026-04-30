import numpy as np
from scipy import stats
import pandas as pd

def ks_drift(X_ref, X_test_df, ks_threshold=0.10):
    """
    Feature-level drift detection using the Kolmogorov-Smirnov statistic.
    """
    ks_stats = np.array([
        stats.ks_2samp(X_ref.iloc[:, f].values, X_test_df.iloc[:, f].values)[0]
        for f in range(X_ref.shape[1])
    ])
    n_drifted = np.sum(ks_stats > ks_threshold)
    return {
        'n_drifted'      : int(n_drifted),
        'drift_fraction' : n_drifted / X_ref.shape[1],
        'mean_ks_stat'   : float(np.mean(ks_stats)),
        'max_ks_stat'    : float(np.max(ks_stats)),
        'ks_stats'       : ks_stats,
    }

def psi_score(X_ref, X_test_df, n_bins=10, psi_cap=0.50):
    """
    Population Stability Index (PSI) — distribution shift monitoring.
    """
    bin_edges = np.linspace(-4, 4, n_bins + 1)
    psi_vals  = []
    for f in range(X_ref.shape[1]):
        ref_col = X_ref.iloc[:, f].values
        tst_col = X_test_df.iloc[:, f].values
        mu, std = ref_col.mean(), ref_col.std()
        if std < 1e-9:
            continue
        ref_z = np.clip((ref_col - mu) / std, -4, 4)
        tst_z = np.clip((tst_col - mu) / std, -4, 4)
        ref_cnt = np.histogram(ref_z, bins=bin_edges)[0]
        tst_cnt = np.histogram(tst_z, bins=bin_edges)[0]
        ref_smooth = np.sqrt(len(ref_z) / n_bins)
        tst_smooth = np.sqrt(len(tst_z) / n_bins)
        ref_pct = (ref_cnt + ref_smooth) / (len(ref_z) + n_bins * ref_smooth)
        tst_pct = (tst_cnt + tst_smooth) / (len(tst_z) + n_bins * tst_smooth)
        psi = float(np.sum((tst_pct - ref_pct) * np.log(tst_pct / ref_pct)))
        psi_vals.append(min(psi, psi_cap))
    mean_psi = float(np.mean(psi_vals))
    verdict  = ("STABLE"         if mean_psi < 0.10 else
                "MONITOR"        if mean_psi < 0.25 else
                "DRIFT DETECTED")
    return {'mean_psi': mean_psi, 'max_psi': float(np.max(psi_vals)),
            'psi_vals': np.array(psi_vals), 'verdict': verdict}
