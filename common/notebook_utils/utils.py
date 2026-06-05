# utils.py
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Any, Hashable
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

import pandas as pd
from sklearn.utils import shuffle

def _to_numpy(a):
    """
    Convert a given object to a numpy array.

    If the object is None, return None.
    Otherwise, use numpy.array to convert the object to a numpy array.

    Parameters
    ----------
    a : Any
        The object to be converted.

    Returns
    -------
    np.ndarray
        The converted numpy array.
    """
    import numpy as _np
    if a is None: return None
    a = _np.array(a)
    return a

def _detect_task(y_true, y_proba) -> str:

    """
    Detect the type of task from the shape of y_true and y_proba.

    Parameters
    ----------
    y_true : np.ndarray
        The true labels of the data.
    y_proba : np.ndarray
        The predicted probabilities of the data.

    Returns
    -------
    str
        The type of task (binary, multilabel, multiclass).
    """
    y_true = _to_numpy(y_true)
    y_proba = _to_numpy(y_proba)
    if y_proba.ndim == 1 or y_proba.shape[1] == 1:
        return "binary"

    if y_true.ndim == 2 and y_true.shape[1] == y_proba.shape[1]:
        vals = np.unique(y_true)
        if np.all(np.isin(vals, [0, 1])): 
            return "multilabel"
    return "multiclass"

def _to_int_labels(y_true) -> np.ndarray:
    y_true = _to_numpy(y_true)
    if y_true.ndim == 2:  # one-hot
        return y_true.argmax(axis=1)
    return y_true.astype(int).ravel()

def _to_multilabel(y_true, n_classes) -> np.ndarray:
    """
    Convert true labels to multilabel format.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    n_classes : int
        Number of classes.

    Returns
    -------
    oh : np.ndarray
        One-hot encoded labels in multilabel format.
    """
    y_true = _to_numpy(y_true)
    if y_true.ndim == 1:
        oh = np.zeros((y_true.size, n_classes), dtype=int)
        oh[np.arange(y_true.size), y_true] = 1
        return oh
    return y_true.astype(int)

def report_classifier(
    history: Any,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    model_name: str = "Model",
    threshold: float = 0.5,
    figsize_curve: Tuple[int,int] = (15,3),
    cmap_cm: str = "Blues",
    show: bool = True,
    save_prefix: Optional[str] = None,
) -> Tuple[Dict[str, plt.Figure], Dict[str, float], Dict[str, Any]]:


    """
    Reports the performance of a classifier.

    Parameters
    ----------
    history : Any
        The history of the classifier (e.g. history attribute of a Keras model).
    y_true : np.ndarray
        The true labels of the data.
    y_proba : np.ndarray
        The predicted probabilities of the data.
    class_names : Optional[List[str]], optional
        The names of the classes. If None, use the range of the number of classes.
    model_name : str, optional
        The name of the model. Defaults to "Model".
    threshold : float, optional
        The threshold for the predicted probabilities. Defaults to 0.5.
    figsize_curve : Tuple[int,int], optional
        The size of the figures for the ROC/AUC curves and the precision-recall curves. Defaults to (15,3).
    cmap_cm : str, optional
        The colormap for the confusion matrix. Defaults to "Blues".
    show : bool, optional
        Whether to show the figures. Defaults to True.
    save_prefix : Optional[str], optional
        The prefix for saving the figures. If None, do not save the figures. Defaults to None.

    Returns
    -------
    Tuple[Dict[str, plt.Figure], Dict[str, float], Dict[str, Any]]
        A tuple containing three dictionaries:
            - figs: a dictionary containing the figures for the ROC/AUC curves, the precision-recall curves, and the confusion matrix.
            - aucs: a dictionary containing the AUC macro, micro, and weighted scores.
            - metrics: a dictionary containing the accuracy, AP macro score, AP error, classification report, classification report error, and f1 micro score.
    """
    figs, aucs, metrics = {}, {}, {}
    y_proba = _to_numpy(y_proba)
    task = _detect_task(y_true, y_proba)
    n_classes = 1 if y_proba.ndim == 1 else y_proba.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(n_classes if n_classes>1 else 2)]

    hist = getattr(history, "history", None)
    if isinstance(hist, dict):
        # accuracy
        if any(k in hist for k in ("accuracy","acc","binary_accuracy","sparse_categorical_accuracy")):
            fig_acc = plt.figure(figsize=figsize_curve); figs["accuracy"] = fig_acc
            k_acc = next(k for k in hist.keys() if "acc" in k and not k.startswith("val_"))
            k_vac = "val_" + k_acc
            plt.plot(hist.get(k_acc, []), label="train")
            if k_vac in hist: plt.plot(hist.get(k_vac, []), label="val")
            plt.title(f"{model_name} – accuracy"); plt.ylabel("accuracy"); plt.xlabel("epoch"); plt.legend()
            if save_prefix: fig_acc.savefig(f"{save_prefix}_accuracy.png", dpi=150, bbox_inches="tight")
        # loss
        if "loss" in hist:
            fig_loss = plt.figure(figsize=figsize_curve); figs["loss"] = fig_loss
            plt.plot(hist["loss"], label="train")
            if "val_loss" in hist: plt.plot(hist["val_loss"], label="val")
            plt.title(f"{model_name} – loss"); plt.ylabel("loss"); plt.xlabel("epoch"); plt.legend()
            if save_prefix: fig_loss.savefig(f"{save_prefix}_loss.png", dpi=150, bbox_inches="tight")

    if task == "binary":
        proba_pos = y_proba.ravel()
        y_pred = (proba_pos >= threshold).astype(int)
        y_true_int = _to_int_labels(y_true)  # 0/1
    elif task == "multiclass":
        y_pred = y_proba.argmax(axis=1)
        y_true_int = _to_int_labels(y_true)
    else:  # multilabel
        y_true_int = _to_multilabel(y_true, n_classes)
        y_pred = (y_proba >= threshold).astype(int)

    # ===== 2) ROC/AUC =====
    try:
        if task in ("binary","multiclass"):
            y_true_oh = (np.eye(n_classes)[y_true_int] if task=="multiclass"
                         else np.stack([1 - y_true_int, y_true_int], axis=1))
            y_proba_c = (y_proba if task=="multiclass"
                         else np.stack([1 - y_proba.ravel(), y_proba.ravel()], axis=1))
            aucs["macro"] = roc_auc_score(y_true_oh, y_proba_c, multi_class="ovr", average="macro")
            aucs["micro"] = roc_auc_score(y_true_oh, y_proba_c, multi_class="ovr", average="micro")
            aucs["weighted"] = roc_auc_score(y_true_oh, y_proba_c, multi_class="ovr", average="weighted")

            fig_roc = plt.figure(figsize=figsize_curve); figs["roc"] = fig_roc
            C = (n_classes if task=="multiclass" else 2)
            for i in range(C):
                fpr, tpr, _ = roc_curve(y_true_oh[:, i], y_proba_c[:, i])
                auc_i = auc(fpr, tpr)
                label = f"{class_names[i]} (AUC {auc_i:.3f})"
                plt.plot(fpr, tpr, label=label)
            plt.plot([0,1],[0,1],"k--",lw=0.8)
            plt.title(f"ROC – {model_name}")
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(fontsize=8, ncol=min(5, C))
            if save_prefix: fig_roc.savefig(f"{save_prefix}_roc.png", dpi=150, bbox_inches="tight")
        else:
            aucs["macro"] = roc_auc_score(y_true_int, y_proba, average="macro")
    except Exception as e:
        aucs["error"] = f"ROC/AUC failed: {e}"

    # ===== 3) Precision-Recall / AUPRC =====
    try:
        fig_pr = plt.figure(figsize=figsize_curve); figs["pr"] = fig_pr
        if task == "binary":
            prec, rec, _ = precision_recall_curve(y_true_int, proba_pos)
            ap = average_precision_score(y_true_int, proba_pos)
            plt.plot(rec, prec, label=f"AP={ap:.3f}")
            metrics["AP_macro"] = ap
        else:
            aps = []
            C = (n_classes if task=="multiclass" else y_true_int.shape[1])
            y_true_mat = (np.eye(n_classes)[y_true_int] if task=="multiclass" else y_true_int)
            for i in range(C):
                prec, rec, _ = precision_recall_curve(y_true_mat[:, i], y_proba[:, i])
                ap = average_precision_score(y_true_mat[:, i], y_proba[:, i])
                aps.append(ap)
                plt.plot(rec, prec, label=f"{class_names[i]} (AP={ap:.3f})")
            metrics["AP_macro"] = float(np.mean(aps))
        plt.title(f"Precision–Recall – {model_name}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(fontsize=8, ncol=min(5, C if task!='binary' else 1))
        if save_prefix: fig_pr.savefig(f"{save_prefix}_pr.png", dpi=150, bbox_inches="tight")
    except Exception as e:
        metrics["AP_error"] = f"PR/AUPRC failed: {e}"

    # ===== 4) Confusion matrix + classification_report  =====
    
    if task in ("binary","multiclass"):
        cm = confusion_matrix(y_true_int, y_pred, normalize='true')   # normalize by row (true label)
        fig_cm = plt.figure(figsize=(6,5)); figs["cm"] = fig_cm
        disp = ConfusionMatrixDisplay(confusion_matrix=cm*100, display_labels=(["neg","pos"] if task=="binary" else class_names))
        disp.plot(cmap=cmap_cm, colorbar=True, ax=plt.gca(), values_format=".1f")
        plt.gca().set_title(f"Confusion matrix – {model_name} (%)")
        plt.tight_layout()
        if save_prefix: fig_cm.savefig(f"{save_prefix}_cm.png", dpi=150, bbox_inches="tight")

        try:
            report = classification_report(
                y_true_int, y_pred,
                target_names=(["neg","pos"] if task=="binary" else class_names),
                output_dict=True, zero_division=0
            )
            metrics["classification_report"] = report
            metrics["accuracy"] = report.get("accuracy", float((y_true_int==y_pred).mean()))
        except Exception as e:
            metrics["classification_report_error"] = str(e)
    else:
        from sklearn.metrics import f1_score
        metrics["f1_micro"] = f1_score(y_true_int, y_pred, average="micro", zero_division=0)

    if show:
        plt.show()
    else:
        plt.close("all")

    return figs, aucs, metrics

def mrr_at_k(preds, y_true, k=10):
    ranks = []
    for i in range(len(y_true)):
        top_indices = np.argsort(preds[i])[::-1][:k]
        if y_true[i] in top_indices:
            r = np.where(top_indices == y_true[i])[0][0] + 1
            ranks.append(1.0 / r)
        else:
            ranks.append(0.0)
    return np.mean(ranks)


def top_k_accuracy(preds, y_true, k=10):
    top = np.argpartition(preds, -k, axis=1)[:, -k:]
    hits = [1 if y_true[i] in top[i] else 0 for i in range(len(y_true))]
    return np.mean(hits)



def preproc_multiclass(
    df: pd.DataFrame,
    label_column: str = "class",
    n_train_per_class: Optional[int] = None,
    n_test_per_class: Optional[int] = None,
    train_frac: float = 0.8,
    classes: Optional[List[Hashable]] = None,
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    Preprocess a DataFrame for multiclass classification.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be preprocessed.
    label_column : str, default="class"
        Name of the column with the class labels.
    n_train_per_class : Optional[int], default=None
        Number of samples to be used for training per class.
        If None, `train_frac` is used.
    n_test_per_class : Optional[int], default=None
        Number of samples to be used for testing per class.
        If None, `1 - train_frac` is used.
    train_frac : float, default=0.8
        Fraction of samples to be used for training.
    classes : Optional[List[Hashable]], default=None
        List of classes to be considered. If None, all unique values in
        the label column are used.
    random_state : Optional[int], default=42
        Seed used to shuffle the data.

    Returns
    -------
    df_train : pd.DataFrame
        DataFrame with the training data.
    df_test : pd.DataFrame
        DataFrame with the testing data.
    """
    if classes is None:
        classes = list(pd.unique(df[label_column]))
    df = df.copy()

    parts_train = []
    parts_test  = []

    for c in classes:
        c_df = df[df[label_column] == c]
        c_df = shuffle(c_df, random_state=random_state).reset_index(drop=True)

        n_total = len(c_df)
        if n_total == 0:
            continue

        if n_train_per_class is not None or n_test_per_class is not None:
            n_train = n_train_per_class if n_train_per_class is not None else int(n_total * train_frac)
            if n_test_per_class is None:
                n_test = max(0, n_total - n_train)
            else:
                n_test = n_test_per_class
        else:
            n_train = int(n_total * train_frac)
            n_test  = n_total - n_train

        n_train = min(n_train, n_total)
        n_test  = min(n_test,  max(0, n_total - n_train))

        c_train = c_df.iloc[:n_train]
        c_test  = c_df.iloc[n_train:n_train + n_test]

        parts_train.append(c_train)
        parts_test.append(c_test)

    df_train = pd.concat(parts_train, ignore_index=True) if parts_train else pd.DataFrame(columns=df.columns)
    df_test  = pd.concat(parts_test,  ignore_index=True) if parts_test  else pd.DataFrame(columns=df.columns)

    df_train = shuffle(df_train, random_state=random_state).reset_index(drop=True)
    df_test  = shuffle(df_test,  random_state=random_state).reset_index(drop=True)

    return df_train, df_test



# df_train, df_test = preproc_multiclass(df, label_column='class', train_frac=0.8)

# df_train, df_test = preproc_multiclass(df, label_column='class',
#                                        n_train_per_class=10_000,
#                                        n_test_per_class=900)

# df_train, df_test = preproc_multiclass(df, label_column='class',
#                                        classes=[0,1,2,3,4,5,6,7,8,9],
#                                        train_frac=0.75)

