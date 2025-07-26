import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_class_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }


def compute_mae_mad(y_true, y_pred):
    diff = np.abs(np.array(y_true) - np.array(y_pred))
    return {
        "mae": np.mean(diff),
        "mad": np.median(diff)
    }

def plot_residual_hist(residuals, title, output_path):
    plt.figure()
    plt.hist(residuals, bins=100, alpha=0.75)
    plt.title(title)
    plt.xlabel("Residual (seconds)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
