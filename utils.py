from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

def get_f1_micro(y_pred, y_label):
    return f1_score(y_label, y_pred, average="micro")
def get_f1_macro(y_pred, y_label):
    return f1_score(y_label, y_pred, average="macro")
def get_f1(y_pred, y_label):
    return f1_score(y_label, y_pred)
def get_auc(y_pred, y_label):
    return roc_auc_score(y_label, y_pred)
def get_auc_mutil_class(y_label, y_pred, N=2):
    auc_scores = []
    for i in range(N):
        y_true_one_vs_all = np.array([1 if label == i else 0 for label in y_label])
        y_pred_one_vs_all = y_pred[:, i]  # Probability scores for class i
        auc_i = roc_auc_score(y_true_one_vs_all, y_pred_one_vs_all)
        auc_scores.append(auc_i)
    auc_mean = np.mean(auc_scores)
    print("AUC Score (One-vs-All):", auc_mean)
def get_f1_macro_detail(y_label, y_pred, N):
    f1_scores = f1_score(y_label, y_pred, labels=np.arange(N), average=None)
    print("每个类别的F1-score:", f1_scores)
def calculate_tnr_tpr(y_label, y_pred):
    # Initialize counters for TP, TN, FP, and FN
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Assuming y_label and y_pred are lists or arrays containing 0s and 1s
    for true_label, predicted_label in zip(y_label, y_pred):
        if true_label == 1:
            if predicted_label == 1:
                TP += 1
            else:
                FN += 1
        else:
            if predicted_label == 0:
                TN += 1
            else:
                FP += 1

    # Calculate TNR and TPR
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    print("TNR", TNR)
    print("TPR", TPR)
