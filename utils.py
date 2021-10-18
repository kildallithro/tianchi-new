from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
import pandas as pd
from metrics import AUC, MAE, MSE, RMSE, ndcg_at_k, MRR
import torch


class MF_DATA(data.Dataset):
    def __init__(self, filename):
        raw_matrix = np.loadtxt(filename)
        self.users_num = int(10000)  # sample: 10000 df: 250000
        self.items_num = int(6836)  # sample: 6836 df: 364047
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


def evaluate_model(model, val_data, opt):
    true = val_data[:, 2]
    user = torch.LongTensor(val_data[:, 0]).to(opt.device)
    item = torch.LongTensor(val_data[:, 1]).to(opt.device)
    preds = model.predict(user, item).to(opt.device)

    mae = MAE(preds, true)
    mse = MSE(preds, true)
    rmse = RMSE(preds, true)
    auc = AUC(true, preds.detach().cpu().numpy())

    return mae, mse, rmse, auc


# 获取近期点击最多的文章
def get_item_topk_click(train_data, k):
    train_data = pd.DataFrame(train_data)
    topk_click = train_data[:, 1].value_counts().index[:k]

    return topk_click


def evaluate_model_matrix(model, val_data, opt):
    # 先不去除训练集中用户已经选的items
    _row = val_data[:, 0]
    _col = val_data[:, 1]
    _data = val_data[:, 2]
    true = sparse.coo_matrix((_data, (_row, _col)), shape=(10000, 6836), dtype=int)
    preds = model.recommendation()

    ndcg = ndcg_at_k(true, preds.detach().cpu().numpy(), opt.Ks)
    mrr = MRR(true, preds.detach().cpu().numpy(), opt.Ks)

    return ndcg, mrr



