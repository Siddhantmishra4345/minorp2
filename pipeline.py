import numpy as np
from utils.metrics import compute_tps

def selective_engine(conf_lr, pred_lr, conf_fallback, pred_fallback, y_true, tau_high, tau_low):
    """
    Three-tier selective prediction engine.
    """
    if tau_low >= tau_high:
        raise ValueError(
            f"tau_low ({tau_low:.4f}) must be strictly less than "
            f"tau_high ({tau_high:.4f})."
        )

    n = len(conf_lr)
    decision   = np.empty(n, dtype=object)
    final_pred = np.empty(n, dtype=int)

    for i in range(n):
        if conf_lr[i] >= tau_high:
            decision[i]   = 'accept'
            final_pred[i] = pred_lr[i]
        elif conf_lr[i] >= tau_low:
            decision[i]   = 'defer'
            final_pred[i] = pred_fallback[i]
        else:
            decision[i]   = 'reject'
            final_pred[i] = -1

    # if y_true is provided we can compute metrics
    if y_true is not None:
        y_true_arr    = np.asarray(y_true)
        accepted_mask = decision != 'reject'
        covered       = accepted_mask.sum()
        coverage      = covered / n

        sel_acc = (final_pred[accepted_mask] == y_true_arr[accepted_mask]).mean() if covered > 0 else 0.0

        conf_covered  = np.where(decision == 'accept', conf_lr,
                        np.where(decision == 'defer',  conf_fallback, 0.0))
        correct_cov   = (final_pred[accepted_mask] == y_true_arr[accepted_mask])
        conf_cov_vals = conf_covered[accepted_mask]

        ece_sel = 0.0
        if covered > 0:
            bin_edges = np.linspace(0, 1, 11)
            for b in range(10):
                mask = (conf_cov_vals >= bin_edges[b]) & (conf_cov_vals < bin_edges[b + 1])
                if mask.sum() > 0:
                    ece_sel += mask.sum() * abs(
                        correct_cov[mask].mean() - conf_cov_vals[mask].mean()
                    )
            ece_sel /= covered

        tps = compute_tps(coverage, sel_acc, ece_sel) if covered > 0 else 0.0
    else:
        coverage = sel_acc = ece_sel = tps = 0.0

    return {
        'decision'  : decision,
        'final_pred': final_pred,
        'n_accept'  : (decision == 'accept').sum(),
        'n_defer'   : (decision == 'defer').sum(),
        'n_reject'  : (decision == 'reject').sum(),
        'coverage'  : coverage,
        'sel_acc'   : sel_acc,
        'ece_sel'   : ece_sel,
        'tps'       : tps,
    }

def evaluate_model(model, X, y_true, model_name="Model"):
    from utils.metrics import compute_ece
    y_pred   = model.predict(X)
    y_proba  = model.predict_proba(X)
    acc      = (y_pred == np.asarray(y_true)).mean()
    ece_val  = compute_ece(y_true, y_proba)
    conf     = np.max(y_proba, axis=1)
    correct  = (y_pred == np.asarray(y_true))
    overconf = np.sum((conf > 0.9) & ~correct)
    return {'accuracy': acc, 'ece': ece_val, 'confidence': conf,
            'predictions': y_pred, 'probabilities': y_proba,
            'correct': correct, 'overconfident': overconf}
