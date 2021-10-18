from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import dcg_score, ndcg_score
import multiprocessing
import numpy as np


def MSE(preds, true):
    squaredError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)
    return sum(squaredError) / len(squaredError)


def MAE(preds, true):
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        absError.append(abs(dis))
    return sum(absError) / len(absError)


def RMSE(preds, true):
    squaredError = []
    absError = []
    for i in range(len(preds)):
        dis = true[i] - preds[i]
        squaredError.append(dis * dis)
        absError.append(abs(dis))
    from math import sqrt
    return sqrt(sum(squaredError) / len(squaredError))


def Acc(true, preds):
    return accuracy_score(true, preds)


def AUC(true, preds):
    return roc_auc_score(true, preds)


def NLL(true, preds):
    return -log_loss(true, preds, eps=1e-7)


# TopN 评价指标
def dcg_at_k(true, preds, k):
    return dcg_score(true, preds, k)


def ndcg_at_k(true, preds, k):
    true = true.toarray()
    return ndcg_score(true, preds, k)


def MRR(true, preds, k):
    true = true.toarray()
    discount = 1 / (np.arange(true.shape[1]) + 1)
    discount[k:] = 0  # 从 discount 取前 k 个值不为 0
    ranking = np.argsort(preds)[:, ::-1]
    ranked = true[np.arange(ranking.shape[0])[:, np.newaxis], ranking]
    cumulative_gains = discount.dot(ranked.T)
    mrr = np.mean(cumulative_gains)

    return mrr


def recall_at_k(preds, true):
    hits = preds & true

    return hits.sum() / true.sum()


