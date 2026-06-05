import matplotlib.pyplot as plt
import numpy as np

def top_k_accuracy(preds, y_true, k=10):
    """
    preds: np.array shape (N, num_classes)
    y_true: np.array shape (N,)
    """
    topk = np.argpartition(preds, -k, axis=1)[:, -k:]
    hits = [1 if y_true[i] in topk[i] else 0 for i in range(len(y_true))]
    return np.mean(hits)


def mrr_at_k(preds, y_true, k=10):
    """
    Mean Reciprocal Rank @ k
    """
    N = preds.shape[0]
    rr_scores = []

    for i in range(N):
        # Ranking: highest prob, lowest rank index
        top_indices = np.argsort(preds[i])[::-1][:k]

        if y_true[i] in top_indices:
            rank = np.where(top_indices == y_true[i])[0][0] + 1
            rr_scores.append(1.0 / rank)
        else:
            rr_scores.append(0.0)

    return np.mean(rr_scores)


def int_to_axis_word(v):
    return np.uint32(v & 0xFFFF)


def confusion_matrix_np(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def print_confusion_matrix(cm):
    n = len(cm)
    print("Confusion Matrix:")
    print("    Pred:", " ".join([f"{i:5d}" for i in range(n)]))
    for i, row in enumerate(cm):
        print(f"True {i:2d}:", " ".join([f"{v:5d}" for v in row]))


        
def float_to_axis_word(f):
    """
    Convert a Python float into the exact 16-bit fixed-point binary format required by the FPGA model, 
    then pack it into a 32-bit AXI word for DMA transfer.
    """
    FX_FRAC_BITS = 4     # para ap_fixed<16,12>
    fx = int(round(f * (1 << FX_FRAC_BITS))) & 0xFFFF
    return np.uint32(fx)   # 16 LSBs, 16 MSBs = 0



def plot_confusion_matrix(cm, class_names=None, normalize=False, cmap="Blues"):
    """
    Plot a confusion matrix with matplotlib.

    Parameters
    ----------
    cm : ndarray (C x C)
        Confusion matrix.
    class_names : list of str/int
        Names of classes (length C). If None, numbers are used.
    normalize : bool
        If True, each row is normalized to sum to 1.
    cmap : str
        Matplotlib colormap.
    """

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # Normalize rows (optional)
    if normalize:
        cm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Axis labels and ticks
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix"
    )

    # Rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Write values inside cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            txt = f"{value:.2f}" if normalize else str(value)
            ax.set_xlabel("Predicted Label", fontsize=14)
            ax.set_ylabel("True Label", fontsize=14)
            ax.set_title("Confusion Matrix", fontsize=16)
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=14,
                color="white" if value > thresh else "black"
            )

    plt.tight_layout()
    plt.show()
