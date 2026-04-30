import numpy as np

def compute_global_top_k(shap_values, k=10, class_idx=None):
    """Global top-k features by mean |SHAP| value."""
    if class_idx is not None:
        importance = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)
    else:
        importance = np.mean(np.abs(shap_values), axis=(0, 2))
    return frozenset(np.argsort(importance)[::-1][:k].tolist())

def compute_ess(shap_values, sample_idx, k=10, class_idx=None):
    """Explainability Stability Score for a single sample."""
    if class_idx is not None:
        local_imp = np.abs(shap_values[sample_idx, :, class_idx])
    else:
        local_imp = np.mean(np.abs(shap_values[sample_idx]), axis=1)
    local_top_k  = frozenset(np.argsort(local_imp)[::-1][:k].tolist())
    global_top_k = compute_global_top_k(shap_values, k=k, class_idx=class_idx)
    return len(global_top_k & local_top_k) / k

def compute_ess_batch(shap_values, sample_indices, k=10, class_idx=None):
    """Compute ESS for a batch of samples."""
    global_top_k = compute_global_top_k(shap_values, k=k, class_idx=class_idx)
    scores = []
    for idx in sample_indices:
        if class_idx is not None:
            local_imp = np.abs(shap_values[idx, :, class_idx])
        else:
            local_imp = np.mean(np.abs(shap_values[idx]), axis=1)
        local_top_k = frozenset(np.argsort(local_imp)[::-1][:k].tolist())
        scores.append(len(global_top_k & local_top_k) / k)
    return np.array(scores)
