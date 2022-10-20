import numpy as np
from sklearn.metrics import log_loss

def aggregated_log_loss(
    y_true,
    y_pred
):
    """
    Calculate the aggregated logistic loss.
    
    Parameters
    ----------
    y_true
        array-like or label indicator matrix. 
        Ground truth (correct) labels for n_samples samples.

    y_pred
        array-like of float, shape = (n_samples, n_classes) or (n_samples,)
        Predicted probabilities, as returned by a classifier’s predict_proba method
        
    Returns
    -------
    mean_log_loss
        The mean log loss for each label class.
        
    Notes
    -----
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
    """
    log_losses = []
    for label in y_pred.columns:
        log_losses.append(log_loss(y_true[label], y_pred[label]))
    return np.array(log_losses).mean()