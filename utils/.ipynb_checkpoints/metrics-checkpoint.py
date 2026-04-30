import numpy as np
import matplotlib.pyplot as plt

def compute_ece(y_true, y_proba_mat, n_bins=10):
    """Expected Calibration Error (ECE)."""
    confidence = np.max(y_proba_mat, axis=1)
    predicted  = np.argmax(y_proba_mat, axis=1) + 1   # back to 1-indexed
    correct    = (predicted == np.asarray(y_true))
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        mask = (confidence >= bin_edges[b]) & (confidence < bin_edges[b + 1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(correct[mask].mean() - confidence[mask].mean())
    return ece / len(y_true)

def compute_tps(coverage, sel_acc, ece):
    """Trustworthy Prediction Score (TPS)."""
    return coverage * sel_acc / (1 + ece)

def plot_reliability(ax, y_true, y_proba, title, color='#2196F3', n_bins=10):
    """Plot a reliability (calibration) diagram on ax and return ECE."""
    confidence = np.max(y_proba, axis=1)
    predicted  = np.argmax(y_proba, axis=1) + 1
    correct    = (predicted == np.asarray(y_true))
    bin_edges  = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_sizes = [], [], []
    for b in range(n_bins):
        mask = (confidence >= bin_edges[b]) & (confidence < bin_edges[b + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidence[mask].mean())
            bin_sizes.append(mask.sum())
            
    ece = sum(s * abs(a - c) for a, c, s in zip(bin_accs, bin_confs, bin_sizes)) / len(y_true)
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.75,
           color=color, label='Actual Accuracy', edgecolor='white')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
    ax.fill_between(bin_confs, bin_confs, bin_accs, alpha=0.2, color='red')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel("Mean Confidence", fontsize=10)
    ax.set_ylabel("Actual Accuracy", fontsize=10)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ece
